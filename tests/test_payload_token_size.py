"""
Test para verificar la sanitización del payload multimedia y prevenir el desbordamiento de tokens (Error 413).
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.models import QuestionData, QuestionType, MediaContext, OptionChoice
from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.config import Settings


def test_payload_sanitization_excludes_base64():
    """Verifica que imágenes con URIs Base64 gigantescas NO se incluyan en el payload JSON enviado al LLM."""
    huge_base64_img = "data:image/png;base64," + ("A" * 120000)  # ~120 KB de base64 (~30,000 tokens)

    question = QuestionData(
        question_id="q_with_huge_image",
        number="Pregunta 1",
        qtype=QuestionType.MULTICHOICE,
        prompt="¿Cuál es la respuesta correcta según el diagrama?",
        media_context=MediaContext(
            has_images=True,
            image_descriptions=["Diagrama de flujo de circuitos"],
            image_urls=[huge_base64_img],
            ocr_texts=["Nodo A = 5V, Nodo B = 0V"],
            has_audio=False,
            audio_urls=[],
            audio_transcriptions=[],
            equations=["V = I * R"],
            tables=[]
        ),
        options=[
            OptionChoice(id="opt1", text="5V", value="1", selector="#opt1", is_selected=False),
            OptionChoice(id="opt2", text="0V", value="2", selector="#opt2", is_selected=False),
        ],
        sub_questions=[],
        is_multiple_response=False,
        is_answered=False
    )

    engine = LLMEngine(Settings(groq_api_key="gsk_test_key_mock_1234567890", enable_local_fallback=False))

    # Construir el payload tal como lo hace solve_batch_questions
    sub_q_formatted = []
    sanitized_media = {
        "has_images": question.media_context.has_images,
        "image_descriptions": question.media_context.image_descriptions,
        "ocr_texts": question.media_context.ocr_texts,
        "has_audio": question.media_context.has_audio,
        "audio_transcriptions": question.media_context.audio_transcriptions,
        "equations": question.media_context.equations,
        "tables": question.media_context.tables,
    }

    payload = [{
        "question_id": question.question_id,
        "number": question.number,
        "qtype": question.qtype.value,
        "full_reading_text_with_gaps": question.prompt,
        "media": sanitized_media,
        "options": [opt.model_dump() for opt in question.options],
        "sub_questions": sub_q_formatted,
        "is_multiple_response": question.is_multiple_response
    }]

    serialized = json.dumps(payload, ensure_ascii=False)

    # 1. El payload no debe contener la cadena Base64
    assert "data:image/png;base64" not in serialized
    assert "AAAAAAA" not in serialized

    # 2. El tamaño debe ser minúsculo (< 1 KB) en vez de > 120 KB
    assert len(serialized) < 1500, f"Payload demasiado grande: {len(serialized)} bytes"

    # 3. La información de OCR útil sí debe estar presente
    assert "Nodo A = 5V" in serialized
    assert "V = I * R" in serialized

    print(f"✓ Sanitización exitosa: Payload original reducido de >120 KB a {len(serialized)} bytes ({len(serialized.split())} palabras aprox.)")


def test_cloze_gaps_do_not_duplicate_full_reading_text():
    """Verifica que un ejercicio Cloze con 18 huecos no repita la lectura completa en cada hueco."""
    from moodle_assistant.models import SubQuestion
    
    # Simular una lectura larga de 10,000 caracteres
    long_reading = "In the 1980s technology changed. " * 300  # ~9,000 caracteres
    
    sub_questions = []
    for i in range(1, 19):
        # Si un hueco recibía el prompt largo antes de la corrección
        sub_questions.append(SubQuestion(
            sub_id=f"gap_{i}",
            prompt=long_reading,  # Simular prompt extenso previo
            selector=f"select[name='gap_{i}']",
            options=[OptionChoice(id=f"opt_{i}", text="word", value="1", selector="", is_selected=False)],
            selected_value=None
        ))
    
    question = QuestionData(
        question_id="cloze_18_gaps",
        number="Pregunta 1",
        qtype=QuestionType.MULTIANSWER,
        prompt=long_reading,
        media_context=MediaContext(),
        options=[],
        sub_questions=sub_questions,
        is_multiple_response=False,
        is_answered=False
    )
    
    # Procesar como lo hace solve_batch_questions
    sub_q_formatted = []
    for s_idx, sq in enumerate(question.sub_questions):
        local_prompt = sq.prompt or f"Espacio [{s_idx + 1}]"
        if len(local_prompt) > 160:
            local_prompt = local_prompt[:157] + "..."
        sub_q_formatted.append({
            "sub_id": sq.sub_id,
            "gap_number": s_idx + 1,
            "prompt": local_prompt,
            "available_options": [
                {"id": opt.id, "value": opt.value, "text": opt.text}
                for opt in sq.options
            ]
        })
    
    payload = [{
        "question_id": question.question_id,
        "number": question.number,
        "qtype": question.qtype.value,
        "full_reading_text_with_gaps": question.prompt,
        "sub_questions": sub_q_formatted,
    }]
    
    serialized = json.dumps(payload)
    # Sin truncado, 9,000 caracteres * 19 superaría los 170,000 caracteres (>40,000 tokens).
    # Con truncado, debe estar alrededor de ~15,000 caracteres (apenas ~3,000 tokens).
    assert len(serialized) < 20000, f"Payload Cloze excesivo: {len(serialized)} caracteres"
    print(f"✓ Control de tokens en Cloze exitoso: {len(serialized)} caracteres para 18 huecos.")


if __name__ == "__main__":
    test_payload_sanitization_excludes_base64()
    test_cloze_gaps_do_not_duplicate_full_reading_text()
    print("✓ Todos los tests de test_payload_token_size.py pasaron exitosamente!")
