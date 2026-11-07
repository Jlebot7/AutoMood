"""
MoodleAssist-AI - Motor de Inferencia LLM con AsyncGroq
Gestiona la comunicación asíncrona con Groq, la optimización de prompts y la validación JSON.
"""

from __future__ import annotations

import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional
from groq import AsyncGroq, RateLimitError, APIError, APIConnectionError

logger = logging.getLogger("AutoMood.LLM")

from .config import settings, Settings
from .models import (
    QuestionData,
    QuestionType,
    LLMResponse,
    BatchLLMResponse,
)
from .local_llm import LocalLLMEngine

SYSTEM_PROMPT = """Eres MoodleAssist-AI, un asistente de élite especializado en responder con máxima precisión académica cuestionarios y actividades universitarias en Moodle.

Tu objetivo es analizar preguntas académicas en español (o el idioma del enunciado), comprender el contexto teórico, gramatical y lógico, y proporcionar la respuesta correcta con rigor absoluto.

REGLAS DE ORO:
1. LECTURA INTEGRAL Y LÍNEA TEMPORAL (Timeline & Narrative Context):
   - En textos largos, lecturas y actividades de completar huecos (Cloze / multianswer / gapselect):
     * DEBES LEER EL TEXTO COMPLETO DE PRINCIPIO A FIN antes de responder cualquier sub-pregunta o hueco individual.
     * Analiza la línea temporal de la narración, marcadores temporales (ej. 'in the 1980s', 'first', 'ago', 'since', 'today', 'now') y la coherencia lógica entre párrafos.
     * En ejercicios de tiempos verbales (Simple Past vs. Present Perfect):
       - Marcador 'SINCE': Las oraciones con 'Since [evento pasado]' (ej. 'Since computers were first introduced..., technology HAS CHANGED...') exigen PRESENT PERFECT en la cláusula principal.
       - Hechos concluidos del pasado histórico usan SIMPLE PAST (ej. 'were', 'didn't have', 'weren't', 'paid', 'did'). En 'They (have, not)' usa 'didn't have'.
       - Era moderna y cambios continuos usan PRESENT PERFECT (ej. 'have changed', 'have become', 'have created', 'have begun', 'have evolved').
       - Mantén paralelismo verbal en oraciones coordinadas (ej. 'have also gotten... and have begun').

2. USO CORRECTO DE CONTRACCIONES (Affirmative vs. Negative):
   - 'USE CONTRACTIONS WHENEVER POSSIBLE' aplica a CONTRACCIONES NEGATIVAS: 'didn\'t', 'weren\'t', 'wasn\'t', 'haven\'t', 'hasn\'t'.
   - ¡PROHIBIDO USAR CONTRACCIONES AFIRMATIVAS ('ve, 's) TRAS SUSTANTIVOS O FRASES NOMINALES!
     * En inglés estándar, las contracciones afirmativas ('ve, 's) NUNCA se unen a sustantivos: usa 'Computers have become', 'Times have changed', 'technology has changed', 'Programmers have created'.
     * Usa SIEMPRE las formas completas ('have' / 'has') cuando el sujeto sea un sustantivo o tras conjunciones.

3. ANÁLISIS DE EJEMPLOS Y CONSIGNAS DE FORMATO (In-Context Examples & Guidelines):
   - Si el enunciado, la instrucción o la lectura contiene un ejemplo modelo (ej. 'Example:', 'e.g.', 'Por ejemplo:', 'Modelo:', 'Sample:', '[0] Answer 0'):
     * ANALIZA EL EJEMPLO COMO GUÍA DEFINITIVA: examina su morfología, si incluye o no sujeto, mayúsculas/minúsculas y contracciones.
     * REPLICA ESTRICTAMENTE EL MISMO PATRÓN Y FORMATO en todas las respuestas y huecos restantes.
   - Cumple con rigor total cualquier directiva de formato (ej. 'Write only the verb', 'Use contractions', 'Responda en una sola palabra').

4. REGLA ESTRICTA CONTRA LA REPETICIÓN DE ORACIONES EN HUECOS EN LÍNEA:
   - ¡PROHIBIDO ESCRIBIR LA ORACIÓN COMPLETA EN HUECOS EN LÍNEA (Cloze / fill-in-the-blank)!
   - Si el enunciado dice: "You [HUECO 11: sub11] eat more vegetables to stay healthy." y la palabra faltante es el modal 'should':
     * En matching_answers el valor DEBE SER ÚNICAMENTE: "sub11": "should".
     * ¡JAMÁS DEVUELVAS "You should eat more vegetables to stay healthy."!
     * Escribir la oración completa duplicará las palabras en el formulario ("You You should eat more vegetables...") e invalidará el examen.
   - Para desplegables (dropdowns):
     * DEBES seleccionar EXCLUSIVAMENTE una de las opciones listadas en 'Opciones: [...]'.
     * NUNCA inventes o conjugues verbos fuera de la lista dada (ej. no pongas 'ran' si las opciones son 'could', 'should', 'would').

5. EJERCICIOS DE DICTADO Y AUDIO (Dictation & Listening):
   - Si la pregunta incluye una o varias transcripciones de audio ('🎧 Transcripción de audio (Whisper) ...'):
     * Analiza las etiquetas de archivo y el contenido de cada audio. Si hay audios de dictado (ej. 'ttsreader', oraciones numeradas), asócialos con la sección correspondiente.
     * En ejercicios de DICTADO (donde se solicita transcribir oraciones completas dictadas): transcribe fielmente en el hueco correspondiente cada una de las oraciones dictadas que escuchaste, con mayúscula inicial y ortografía correcta.
     * En ejercicios de comprensión auditiva con opciones: basa tus respuestas con exactitud en lo que dice el audio correspondiente.

6. PRECISIÓN ACADÉMICA MÁXIMA: Elige la respuesta inequívocamente correcta según el marco teórico estándar de la disciplina.
7. FORMATO ESTRICTO: DEBES responder ÚNICAMENTE con un objeto JSON válido que cumpla exactamente la estructura solicitada. NUNCA agregues texto conversacional antes o después del JSON.
8. COINCIDENCIA DE IDS Y TEXTOS:
   - Para selección múltiple / simple (multichoice, truefalse): Indica en 'selected_options' los IDs exactos o textos de las opciones correctas.
   - Para desplegables, huecos o emparejamiento (match, gapselect, multianswer): En 'matching_answers' mapea CADA sub_id con su respuesta:
     * Si el hueco es un desplegable con opciones: coloca el valor o texto exacto de la opción seleccionada.
     * Si el hueco es un campo de texto libre (fill-in-the-blank): coloca la palabra o frase exacta requerida para rellenar el espacio.
     * DEBES incluir TODOS los sub_id listados sin omitir ninguno.
   - Para texto libre o numérico (shortanswer, numerical): Proporciona la respuesta exacta sin rodeos en 'text_answer'.
9. RAZONAMIENTO GLOBAL EN 'explanation': En el campo 'explanation', resume brevemente en 1-2 líneas el contexto general del texto (ej. línea temporal identificada: pasado vs presente, o patrón del ejemplo seguido) y justifica las respuestas clave antes de presentar el JSON.
10. El campo 'question_id' en cada respuesta DEBE SER EXACTAMENTE IGUAL al valor del campo 'question_id' de la pregunta en la entrada.
"""

SINGLE_QUESTION_PROMPT_TEMPLATE = """Analiza la siguiente pregunta de Moodle y selecciona la respuesta correcta.

IMPORTANTE - LECTURA INTEGRAL, LÍNEA TEMPORAL, EJEMPLOS Y CONTRACCIONES:
1. Lee primero todo el enunciado ('Enunciado') de principio a fin para entender el contexto global, la línea temporal y la coherencia del texto.
2. DETECCIÓN DE EJEMPLOS Y CONSIGNAS: Si el enunciado contiene un ejemplo modelo (ej. 'Example:', 'e.g.', 'Modelo:', '[0]'), examina su morfología y estilo, y replica exactamente ese mismo patrón y formato en todas tus respuestas.
3. ¡REGLA DE HUECOS EN LÍNEA!: Para huecos dentro de oraciones escribe ÚNICAMENTE la palabra o verbo modal faltante (ej. 'should', 'could', 'didn't have'). ¡ESTÁ PROHIBIDO escribir la oración completa!
4. ¡REGLA DE DESPLEGABLES!: Selecciona ÚNICAMENTE una de las opciones listadas en 'Opciones: [...]'. No inventes opciones fuera de la lista.
5. ¡EJERCICIOS DE DICTADO!: Para huecos de dictado (donde se transcriben oraciones completas escuchadas en audio), escribe fielmente la oración dictada correspondiente.
6. Cada hueco en el texto está etiquetado como [HUECO N: sub_id]. Relaciona cada hueco con las opciones disponibles abajo.
7. Respeta la concordancia temporal (hechos del pasado histórico en Simple Past vs cambios de la era actual en Present Perfect).
8. REGLA DE CONTRACCIONES: Las contracciones afirmativas ('ve, 's) NUNCA se usan tras sustantivos (ej. 'Computers have become', 'Times have changed', 'technology has changed'). Las contracciones son para formas negativas ('didn't', 'weren't', 'haven't').

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

IMPORTANTE - LECTURA INTEGRAL, LÍNEA TEMPORAL, EJEMPLOS Y CONTRACCIONES:
1. En preguntas de lectura o textos con huecos (multianswer / cloze / gapselect), lee PRIMERO todo el texto completo ('full_reading_text_with_gaps') de principio a fin para comprender el contexto global.
2. DETECCIÓN DE EJEMPLOS Y CONSIGNAS: Si en el enunciado o lectura se incluye un ejemplo modelo (ej. 'Example:', 'e.g.', 'Sample:', 'Modelo:', 'Answer 0'), analízalo con prioridad para determinar exactamente la regla gramatical, si incluye o no sujeto, y el patrón de formato esperado por el calificador automático. Replica ese patrón con exactitud.
3. ¡REGLA DE HUECOS EN LÍNEA!: Para huecos dentro de oraciones escribe ÚNICAMENTE la palabra o verbo modal faltante (ej. 'should', 'could', 'didn't have'). ¡ESTÁ PROHIBIDO escribir la oración completa!
4. ¡REGLA DE DESPLEGABLES!: Selecciona ÚNICAMENTE una de las opciones listadas en available_options. No inventes opciones fuera de la lista.
5. ¡EJERCICIOS DE DICTADO!: Para huecos de dictado (donde se transcriben oraciones completas escuchadas en audio), escribe fielmente la oración dictada correspondiente.
6. Identifica la línea temporal (párrafos de hechos concluidos en el pasado vs párrafos de cambios en la era actual o presente).
7. Asegura coherencia y concordancia gramatical entre cada hueco y la totalidad de la lectura.
8. REGLA DE CONTRACCIONES: Las contracciones afirmativas ('ve, 's) NUNCA se usan tras sustantivos (ej. 'Computers have become', 'Times have changed', 'technology has changed'). Las contracciones aplican a formas negativas ('didn't', 'weren't', 'haven't').

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
        self.local_engine: Optional[LocalLLMEngine] = None
        if getattr(self.config, "enable_local_fallback", True):
            self.local_engine = LocalLLMEngine(self.config)

    @property
    def client(self) -> AsyncGroq:
        """Inicializa o retorna la instancia singleton del cliente AsyncGroq."""
        if self._client is None:
            if not self.config.is_groq_configured():
                if self.local_engine and getattr(self.config, "enable_local_fallback", True):
                    # Fallback local disponible si Groq no está configurado
                    return None
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
            if client is None:
                return False
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
        if q.media_context.audio_transcriptions:
            for idx, trans in enumerate(q.media_context.audio_transcriptions, 1):
                lines.append(f"- 🎧 Transcripción de audio (Whisper) #{idx}:\n\"\"\"\n{trans}\n\"\"\"")
        elif q.media_context.has_audio:
            lines.append("- 🎧 Archivo de audio detectado en la pregunta (sin transcripción disponible).")

        if q.media_context.ocr_texts:
            for idx, ocr in enumerate(q.media_context.ocr_texts, 1):
                lines.append(f"- 🖼️ Texto extraído de imagen (OCR) #{idx}:\n\"\"\"\n{ocr}\n\"\"\"")
        elif q.media_context.has_images and q.media_context.image_descriptions:
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
        """Formatea sub-preguntas para match, gapselect o multianswer con su número de hueco correlativo de forma ultra compacta."""
        if not q.sub_questions:
            return ""
        lines = [
            "- Elementos a responder en matching_answers (asócialos con la posición [HUECO N: sub_id] en el enunciado):",
            "  (¡REGLA FUNDAMENTAL: Para huecos dentro de oraciones escribe ÚNICAMENTE la palabra o verbo modal faltante. NUNCA escribas la oración completa!)"
        ]
        has_embedded_gaps = "[HUECO " in q.prompt or "[HUECO_" in q.prompt
        for idx, sq in enumerate(q.sub_questions):
            gap_num = idx + 1
            if sq.options:
                opts = [f'"{o.text.strip() or o.value.strip()}"' for o in sq.options if (o.text.strip() or o.value.strip())]
                lines.append(f'  * [HUECO {gap_num}] "{sq.sub_id}": [{", ".join(opts)}]')
            else:
                if not has_embedded_gaps and sq.prompt and len(sq.prompt.strip()) > 0:
                    ctx = sq.prompt.strip()
                    if len(ctx) > 80:
                        ctx = ctx[:77] + "..."
                    lines.append(f'  * [HUECO {gap_num}] "{sq.sub_id}": (Texto libre) | Contexto: "{ctx}"')
                else:
                    lines.append(f'  * [HUECO {gap_num}] "{sq.sub_id}": (Texto libre)')
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
        """Ejecuta la llamada a Groq con manejo de backoff exponencial, cascada de modelos ante 413 y auto-fallback al modelo local."""
        # Si Groq no está configurado pero el fallback local está activo
        if not self.config.is_groq_configured() and self.local_engine:
            logger.info("GROQ_API_KEY no configurada. Ejecutando inferencia con modelo local...")
            return await self.local_engine.generate_response(messages)

        client = self.client
        if client is None:
            if self.local_engine and getattr(self.config, "enable_local_fallback", True):
                return await self.local_engine.generate_response(messages)
            raise ValueError("Cliente Groq no disponible y fallback local deshabilitado.")

        cascade_models = [self.config.groq_model]
        for candidate in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            if candidate not in cascade_models:
                cascade_models.append(candidate)

        last_error = None

        # Intentar en cascada con los modelos disponibles de Groq si hay 413 (límite de tokens TPM) o 404
        for model_name in cascade_models:
            for attempt in range(1, self.config.groq_max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=self.config.groq_temperature,
                    )
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError(f"Respuesta vacía recibida de Groq con {model_name}.")
                    return content
                except RateLimitError as rle:
                    last_error = rle
                    err_str = str(rle).lower()
                    if "413" in err_str or "request too large" in err_str or "tokens" in err_str:
                        logger.warning(
                            "Límite de tokens en Groq con '%s' (413: %s). Cascada al siguiente modelo de Groq...",
                            model_name, rle
                        )
                        break
                    wait_time = (2 ** attempt) + 1.0
                    await asyncio.sleep(wait_time)
                except (APIConnectionError, APIError) as apie:
                    last_error = apie
                    err_str = str(apie).lower()
                    if "413" in err_str or "request too large" in err_str or getattr(apie, "status_code", 0) == 413:
                        logger.warning(
                            "Límite de tokens en Groq con '%s' (413). Cascada al siguiente modelo de Groq...",
                            model_name
                        )
                        break
                    if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                        logger.warning("Modelo '%s' no encontrado en Groq (404). Cascada al siguiente...", model_name)
                        break
                    await asyncio.sleep(1.0 * attempt)
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "413" in err_str or "request too large" in err_str or "tokens" in err_str:
                        logger.warning(
                            "Límite de tokens con modelo '%s'. Cascada al siguiente modelo de Groq...",
                            model_name
                        )
                        break
                    if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                        break
                    await asyncio.sleep(1.0)

        # Si todos los modelos de Groq fallaron o superaron cuota, recurrir al motor local
        if self.local_engine and getattr(self.config, "enable_local_fallback", True):
            logger.warning(
                "Todos los modelos de Groq fallaron o alcanzaron límites de cuota (%s). Activando fallback a modelo de IA local...",
                last_error
            )
            return await self.local_engine.generate_response(messages)

        raise RuntimeError(f"Fallo al invocar Groq tras cascada de modelos: {last_error}")

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
            has_embedded_gaps = "[HUECO " in q.prompt or "[HUECO_" in q.prompt
            for s_idx, sq in enumerate(q.sub_questions):
                sub_item: Dict[str, Any] = {
                    "sub_id": sq.sub_id,
                    "gap_number": s_idx + 1,
                }
                if sq.options:
                    sub_item["available_options"] = [
                        opt.text.strip() or opt.value.strip()
                        for opt in sq.options
                        if (opt.text.strip() or opt.value.strip())
                    ]
                elif not has_embedded_gaps and sq.prompt:
                    sub_item["prompt"] = sq.prompt[:80]
                sub_q_formatted.append(sub_item)

            sanitized_media = {
                "has_images": q.media_context.has_images,
                "image_descriptions": q.media_context.image_descriptions,
                "ocr_texts": q.media_context.ocr_texts,
                "has_audio": q.media_context.has_audio,
                "audio_transcriptions": q.media_context.audio_transcriptions,
                "equations": q.media_context.equations,
                "tables": q.media_context.tables,
            }

            payload_items.append({
                "question_id": q.question_id,
                "number": q.number,
                "qtype": q.qtype.value,
                "full_reading_text_with_gaps": q.prompt,
                "media": sanitized_media,
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

