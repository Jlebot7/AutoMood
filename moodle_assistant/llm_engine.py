"""
MoodleAssist-AI - Motor de Inferencia LLM con AsyncGroq
Gestiona la comunicación asíncrona con Groq, la optimización de prompts y la validación JSON.
"""

from __future__ import annotations

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from groq import AsyncGroq, RateLimitError, APIError, APIConnectionError

from .config import settings, Settings
from .models import (
    QuestionData,
    QuestionType,
    LLMResponse,
    BatchLLMResponse,
)

SYSTEM_PROMPT = """Eres MoodleAssist-AI, un asistente de élite especializado en responder con máxima precisión académica cuestionarios y actividades universitarias en Moodle.

Tu objetivo es analizar preguntas académicas en español (o el idioma del enunciado), comprender el contexto teórico, gramatical y lógico, y proporcionar la respuesta correcta con rigor absoluto.

REGLAS DE ORO:
1. LECTURA INTEGRAL Y CONTEXTO NARRATIVO GLOBAL PRIMERO:
   - En preguntas con textos largos, lecturas, actividades de completar huecos (Cloze / multianswer / gapselect) y ejercicios de gramática o idiomas:
     * DEBES LEER EL TEXTO COMPLETO DE PRINCIPIO A FIN antes de responder cualquier sub-pregunta o hueco individual.
     * Analiza la línea temporal (timeline) de la narración, los marcadores temporales (ej. 'in the 1980s', 'first', 'ago', 'since', 'today', 'now') y la coherencia lógica entre párrafos.
     * En ejercicios de tiempos verbales (ej. Simple Past vs. Present Perfect):
       - ATENCIÓN AL MARCADOR 'SINCE': Las oraciones que inician con 'Since [evento pasado]' (ej. 'Since computers were first introduced in the early 1980\\'s, technology HAS CHANGED a great deal') conectan el punto de inicio en el pasado con el presente y exigen obligatoriamente PRESENT PERFECT ('has changed') en la cláusula principal.
       - Hechos concluidos del pasado histórico: Las oraciones que describen cómo eran o funcionaban los objetos o personas en esa época pasada (ej. 'The first computers were...', 'They didn\\'t have...', 'they weren\\'t...', 'customers paid...', 'machines did very little', 'Most computers were...') DEBEN ir en SIMPLE PAST (ej. 'didn\\'t have', 'weren\\'t', 'were', 'paid', 'did'). En 'They (have, not)' usa 'didn\\'t have' (NUNCA 'haven\\'t had').
       - La era moderna y actual: Las oraciones que describen la época moderna y cambios continuos (ej. 'Times have changed', 'Computers have become', 'today\\'s games have become', 'Many computer users have also gotten on the Internet and have begun', 'We have started...', 'machines of the past have evolved') DEBEN ir en PRESENT PERFECT (ej. 'have changed', 'have become', 'have created', 'have begun', 'have started', 'have evolved').
       - Mantén el paralelismo verbal en oraciones coordinadas (ej. 'have also gotten... and have begun').

2. USO CORRECTO DE CONTRACCIONES (Affirmative vs. Negative Contractions):
   - La indicación 'USE CONTRACTIONS WHENEVER POSSIBLE' se refiere a CONTRACCIONES NEGATIVAS:
     * 'did not' -> 'didn\\'t'
     * 'were not' -> 'weren\\'t'
     * 'was not' -> 'wasn\\'t'
     * 'have not' -> 'haven\\'t'
     * 'has not' -> 'hasn\\'t'
   - ¡PROHIBIDO USAR CONTRACCIONES AFIRMATIVAS ('ve, 's) DESPUÉS DE SUSTANTIVOS O FRASES NOMINALES!
     * En inglés estándar, las contracciones afirmativas ('ve, 's) NUNCA se unen a sustantivos (comunes, propios o plurales) ni tras conjunciones coordinadas:
       - 'technology has changed' (NUNCA 'technology \\'s changed')
       - 'Times have changed' (NUNCA 'Times \\'ve changed')
       - 'Computers have become' (NUNCA 'Computers \\'ve become')
       - 'Programmers have created' (NUNCA 'Programmers \\'ve created')
       - 'today\\'s games have become' (NUNCA 'today\\'s games \\'ve become')
       - 'Many computer users have also gotten' (NUNCA 'users \\'ve also gotten')
       - 'and have begun' (NUNCA 'and \\'ve begun')
       - 'the simple machines of the past have evolved' (NUNCA 'machines \\'ve evolved')
       - 'We have started' (usa la forma completa 'have started').
     * Usa SIEMPRE las formas completas ('have' / 'has') cuando el sujeto sea un sustantivo o cuando la contracción no esté unida directamente a un pronombre personal.

3. PRECISIÓN ACADÉMICA MÁXIMA: Elige la respuesta inequívocamente correcta según el marco teórico estándar de la disciplina.
4. FORMATO ESTRICTO: DEBES responder ÚNICAMENTE con un objeto JSON válido que cumpla exactamente la estructura solicitada. NUNCA agregues texto conversacional antes o después del JSON.
5. COINCIDENCIA DE IDS Y TEXTOS:
   - Para selección múltiple / simple (multichoice, truefalse): Indica en 'selected_options' los IDs exactos o textos de las opciones correctas.
   - Para desplegables, huecos o emparejamiento (match, gapselect, multianswer): En 'matching_answers' mapea cada sub_id con el valor o texto exacto de la opción seleccionada.
   - Para texto libre o numérico (shortanswer, numerical): Proporciona la respuesta exacta sin rodeos en 'text_answer'.
6. RAZONAMIENTO GLOBAL EN 'explanation': En el campo 'explanation', resume brevemente en 1-2 líneas el contexto general del texto (ej. línea temporal identificada: pasado vs presente) y justifica las respuestas clave antes de presentar el JSON.
7. El campo 'question_id' en cada respuesta DEBE SER EXACTAMENTE IGUAL al valor del campo 'question_id' de la pregunta en la entrada.
"""

SINGLE_QUESTION_PROMPT_TEMPLATE = """Analiza la siguiente pregunta de Moodle y selecciona la respuesta correcta.

IMPORTANTE - LECTURA INTEGRAL, LÍNEA TEMPORAL Y CONTRACCIONES:
1. Lee primero todo el enunciado ('Enunciado') de principio a fin para entender el contexto global, la línea temporal y la coherencia del texto.
2. Cada hueco en el texto está etiquetado como [HUECO N: sub_id]. Relaciona cada hueco con las opciones disponibles abajo.
3. Respeta la concordancia temporal (hechos del pasado histórico en Simple Past vs cambios de la era actual en Present Perfect).
4. REGLA DE CONTRACCIONES: Las contracciones afirmativas ('ve, 's) NUNCA se usan tras sustantivos (ej. 'Computers have become', 'Times have changed', 'technology has changed'). Las contracciones son para formas negativas ('didn't', 'weren't', 'haven't').

PREGUNTA:
- ID: {question_id}
- Número: {number}
- Tipo: {qtype}
- Enunciado Completo con Huecos: {prompt}
{media_info}
{options_info}
{sub_questions_info}

RESPONDE OBLIGATORIAMENTE con un JSON con la siguiente estructura exacta:
{{
  "question_id": "{question_id}",
  "explanation": "Contexto temporal general identificado y justificación concisa",
  "selected_options": ["id_o_texto_exacto"],
  "text_answer": null,
  "matching_answers": {{
    "sub_id_1": "opcion_o_texto_elegido"
  }},
  "confidence": 0.98
}}
"""

BATCH_QUESTIONS_PROMPT_TEMPLATE = """Analiza las siguientes preguntas de Moodle y proporciona la respuesta correcta para CADA UNA de ellas.

IMPORTANTE - LECTURA INTEGRAL, LÍNEA TEMPORAL Y CONTRACCIONES:
1. En preguntas de lectura o textos con huecos (multianswer / cloze / gapselect), lee PRIMERO todo el texto completo ('full_reading_text_with_gaps') de principio a fin para comprender el contexto global.
2. Identifica la línea temporal (párrafos de hechos concluidos en el pasado vs párrafos de cambios en la era actual o presente).
3. Asegura coherencia y concordancia gramatical entre cada hueco y la totalidad de la lectura.
4. REGLA DE CONTRACCIONES: Las contracciones afirmativas ('ve, 's) NUNCA se usan tras sustantivos (ej. 'Computers have become', 'Times have changed', 'technology has changed'). Las contracciones aplican a formas negativas ('didn't', 'weren't', 'haven't').

LISTA DE PREGUNTAS:
{questions_payload}

RESPONDE OBLIGATORIAMENTE con un JSON con la siguiente estructura exacta:
{{
  "answers": [
    {{
      "question_id": "COPIAR_EL_QUESTION_ID_EXACTO",
      "explanation": "Contexto temporal general identificado y justificación concisa",
      "selected_options": ["id_o_texto_opcion"],
      "text_answer": null,
      "matching_answers": {{
        "sub_id_1": "opcion_o_texto_elegido",
        "sub_id_2": "opcion_o_texto_elegido"
      }},
      "confidence": 0.98
    }}
  ]
}}
"""


class LLMEngine:
    """Motor de inferencia LLM utilizando Groq AsyncGroq."""

    def __init__(self, cfg: Settings = settings):
        self.config = cfg
        self._client: Optional[AsyncGroq] = None

    @property
    def client(self) -> AsyncGroq:
        """Inicializa o retorna la instancia singleton del cliente AsyncGroq."""
        if self._client is None:
            if not self.config.is_groq_configured():
                raise ValueError(
                    "La clave GROQ_API_KEY no está configurada o es inválida. "
                    "Edita el archivo .env o define la variable de entorno GROQ_API_KEY."
                )
            self._client = AsyncGroq(
                api_key=self.config.groq_api_key,
                timeout=self.config.groq_timeout,
            )
        return self._client

    async def test_connection(self) -> bool:
        """Prueba rápida de conectividad y validez de la API key de Groq."""
        try:
            client = self.client
            response = await client.chat.completions.create(
                model=self.config.groq_model,
                messages=[
                    {"role": "system", "content": "You are a test ping bot. Respond with JSON: {\"status\": \"ok\"}"},
                    {"role": "user", "content": "ping"}
                ],
                response_format={"type": "json_object"},
                max_tokens=20,
                temperature=0.0
            )
            content = response.choices[0].message.content or ""
            return "status" in content or "ok" in content
        except Exception:
            return False

    def _format_media_info(self, q: QuestionData) -> str:
        """Formatea información multimedia para el prompt."""
        lines = []
        if q.media_context.has_images and q.media_context.image_descriptions:
            lines.append(f"- Imágenes descritas: {'; '.join(q.media_context.image_descriptions)}")
        if q.media_context.equations:
            lines.append(f"- Fórmulas matemáticas detectadas: {'; '.join(q.media_context.equations)}")
        if q.media_context.tables:
            lines.append(f"- Tablas de datos:\n" + "\n".join(q.media_context.tables))
        return "\n".join(lines) if lines else "- Contexto multimedia: Ninguno"

    def _format_options_info(self, q: QuestionData) -> str:
        """Formatea las opciones para multichoice o truefalse."""
        if not q.options:
            return ""
        lines = ["- Opciones disponibles:"]
        for opt in q.options:
            lines.append(f"  * ID: \"{opt.id}\" | Valor: \"{opt.value}\" | Texto: \"{opt.text}\"")
        if q.is_multiple_response:
            lines.append("  (NOTA: Esta pregunta permite múltiples respuestas correctas. Selecciona todas las aplicables)")
        return "\n".join(lines)

    def _format_sub_questions_info(self, q: QuestionData) -> str:
        """Formatea sub-preguntas para match o gapselect con su número de hueco correlativo."""
        if not q.sub_questions:
            return ""
        lines = ["- Elementos / Espacios desplegables a responder (¡Relaciónalos con su posición en el enunciado con huecos anterior!):"]
        for idx, sq in enumerate(q.sub_questions):
            gap_num = idx + 1
            lines.append(f"  * [HUECO {gap_num}] Sub-ID: \"{sq.sub_id}\" | Contexto local: \"{sq.prompt}\"")
            if sq.options:
                opt_strs = [f"[\"{o.id}\" (valor=\"{o.value}\"): \"{o.text}\"]" for o in sq.options]
                lines.append(f"    Opciones posibles: {', '.join(opt_strs)}")
        return "\n".join(lines)

    def _clean_json_response(self, text: str) -> str:
        """Limpia posibles bloques markdown de código envolventes."""
        text = text.strip()
        # Remover ```json y ``` si están presentes
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    async def _execute_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Ejecuta la llamada a Groq con manejo de backoff exponencial y auto-fallback ante modelos 404."""
        fallback_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        last_error = None
        for attempt in range(1, self.config.groq_max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.groq_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.config.groq_temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Respuesta vacía recibida de Groq.")
                return content
            except RateLimitError as rle:
                last_error = rle
                wait_time = (2 ** attempt) + 1.0
                await asyncio.sleep(wait_time)
            except (APIConnectionError, APIError) as apie:
                last_error = apie
                err_str = str(apie).lower()
                if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                    for fb in fallback_models:
                        if fb != self.config.groq_model:
                            self.config.groq_model = fb
                            break
                await asyncio.sleep(1.0 * attempt)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                    for fb in fallback_models:
                        if fb != self.config.groq_model:
                            self.config.groq_model = fb
                            break
                await asyncio.sleep(1.0)

        raise RuntimeError(f"Fallo al invocar Groq tras {self.config.groq_max_retries} intentos: {last_error}")

    async def solve_single_question(self, question: QuestionData) -> LLMResponse:
        """Resuelve una pregunta individual utilizando el modelo de Groq."""
        user_prompt = SINGLE_QUESTION_PROMPT_TEMPLATE.format(
            question_id=question.question_id,
            number=question.number,
            qtype=question.qtype.value,
            prompt=question.prompt,
            media_info=self._format_media_info(question),
            options_info=self._format_options_info(question),
            sub_questions_info=self._format_sub_questions_info(question),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        raw_response = await self._execute_with_retry(messages)
        cleaned_json = self._clean_json_response(raw_response)

        try:
            data = json.loads(cleaned_json)
            if "answers" in data and isinstance(data["answers"], list) and len(data["answers"]) > 0:
                data = data["answers"][0]
            
            if "question_id" not in data or not data["question_id"]:
                data["question_id"] = question.question_id

            # Reconstruir matching_answers si vinieron en selected_options
            if not data.get("matching_answers") and data.get("selected_options") and question.sub_questions:
                rebuilt = {}
                for opt_str in data["selected_options"]:
                    for sq in question.sub_questions:
                        if opt_str.startswith(sq.sub_id):
                            val = opt_str[len(sq.sub_id):].lstrip("_: -")
                            rebuilt[sq.sub_id] = val
                            break
                if rebuilt:
                    data["matching_answers"] = rebuilt

            return LLMResponse.model_validate(data)
        except Exception as e:
            return LLMResponse(
                question_id=question.question_id,
                explanation=f"Error al deserializar respuesta JSON: {str(e)}. Contenido: {cleaned_json[:150]}",
                selected_options=[],
                text_answer=None,
                matching_answers=None,
                confidence=0.0
            )

    async def solve_batch_questions(self, questions: List[QuestionData]) -> List[LLMResponse]:
        """Resuelve múltiples preguntas en una sola petición optimizada para velocidad y coherencia."""
        if not questions:
            return []

        payload_items = []
        for q in questions:
            sub_q_formatted = []
            for s_idx, sq in enumerate(q.sub_questions):
                sub_q_formatted.append({
                    "sub_id": sq.sub_id,
                    "gap_number": s_idx + 1,
                    "prompt": sq.prompt,
                    "available_options": [
                        {"id": opt.id, "value": opt.value, "text": opt.text}
                        for opt in sq.options
                    ]
                })

            payload_items.append({
                "question_id": q.question_id,
                "number": q.number,
                "qtype": q.qtype.value,
                "full_reading_text_with_gaps": q.prompt,
                "media": q.media_context.model_dump(),
                "options": [opt.model_dump() for opt in q.options],
                "sub_questions": sub_q_formatted,
                "is_multiple_response": q.is_multiple_response
            })

        questions_json_str = json.dumps(payload_items, ensure_ascii=False, indent=2)
        user_prompt = BATCH_QUESTIONS_PROMPT_TEMPLATE.format(
            questions_payload=questions_json_str
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = await self._execute_with_retry(messages)
            cleaned_json = self._clean_json_response(raw_response)
            data = json.loads(cleaned_json)

            q_map = {q.question_id: q for q in questions}
            num_map = {q.number.strip().lower(): q for q in questions}
            digit_map = {}
            for q in questions:
                d = "".join(filter(str.isdigit, q.number))
                if d:
                    digit_map[d] = q

            if "answers" in data and isinstance(data["answers"], list):
                parsed_answers = []
                for idx, ans_dict in enumerate(data["answers"]):
                    raw_qid = str(ans_dict.get("question_id", "")).strip()
                    target_q = None
                    if raw_qid in q_map:
                        target_q = q_map[raw_qid]
                    elif raw_qid.lower() in num_map:
                        target_q = num_map[raw_qid.lower()]
                    else:
                        raw_digits = "".join(filter(str.isdigit, raw_qid))
                        if raw_digits in digit_map:
                            target_q = digit_map[raw_digits]

                    if not target_q and len(data["answers"]) == len(questions) and idx < len(questions):
                        target_q = questions[idx]
                    elif not target_q and len(questions) == 1:
                        target_q = questions[0]

                    if target_q:
                        ans_dict["question_id"] = target_q.question_id

                    ans_obj = LLMResponse.model_validate(ans_dict)
                    ans_obj = self._rebuild_matching_answers(ans_obj, q_map)
                    parsed_answers.append(ans_obj)
                return parsed_answers

            if "question_id" in data:
                if len(questions) == 1:
                    data["question_id"] = questions[0].question_id
                ans_obj = LLMResponse.model_validate(data)
                return [self._rebuild_matching_answers(ans_obj, q_map)]

            answers = []
            for k, v in data.items():
                if isinstance(v, dict):
                    raw_k = str(k).strip()
                    target_q = q_map.get(raw_k) or num_map.get(raw_k.lower())
                    if not target_q and len(questions) == 1:
                        target_q = questions[0]
                    v["question_id"] = target_q.question_id if target_q else raw_k
                    ans_obj = LLMResponse.model_validate(v)
                    answers.append(self._rebuild_matching_answers(ans_obj, q_map))
            if answers:
                return answers

        except Exception:
            pass

        tasks = [self.solve_single_question(q) for q in questions]
        return await asyncio.gather(*tasks)

    def _rebuild_matching_answers(self, ans: LLMResponse, questions_map: Dict[str, QuestionData]) -> LLMResponse:
        """Reconstruye matching_answers si el modelo las retornó dentro de selected_options."""
        q = questions_map.get(ans.question_id)
        if q and q.sub_questions and not ans.matching_answers and ans.selected_options:
            rebuilt = {}
            for opt_str in ans.selected_options:
                for sq in q.sub_questions:
                    if opt_str.startswith(sq.sub_id):
                        val = opt_str[len(sq.sub_id):].lstrip("_: -")
                        rebuilt[sq.sub_id] = val
                        break
            if rebuilt:
                ans.matching_answers = rebuilt
        return ans

