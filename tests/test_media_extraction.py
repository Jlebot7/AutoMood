"""
Tests de extracción DOM y enriquecimiento de prompts para Audio y OCR.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.dom_parser import DOMParser, JS_EXTRACTOR_SCRIPT
from moodle_assistant.models import QuestionData, QuestionType, MediaContext
from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.config import Settings


SAMPLE_MULTIMEDIA_HTML = """
<div id="q_audio_1" class="que multichoice notyetanswered">
    <div class="info">
        <h3 class="no">Pregunta 1</h3>
        <div class="state">Sin responder aún</div>
    </div>
    <div class="formulation">
        <p>Listen to the conversation and choose the correct answer:</p>
        <audio controls="controls">
            <source src="https://campus.edu/pluginfile.php/999/quiz/intro/dialogue.mp3" type="audio/mp3">
        </audio>
        <p>Or download file: <a href="https://campus.edu/pluginfile.php/999/quiz/intro/backup_dialogue.wav">Audio backup</a></p>
        <p><img src="https://campus.edu/pix/icon.png" class="icon" alt="UI Icon" /></p>
        <p><img src="https://campus.edu/pluginfile.php/999/quiz/intro/scene_chart.png" alt="Scene Chart" /></p>
        <div class="answer">
            <div class="r0">
                <input type="radio" name="q_audio_1_choice" value="0" id="q_audio_1_opt0">
                <label for="q_audio_1_opt0">Option A</label>
            </div>
            <div class="r1">
                <input type="radio" name="q_audio_1_choice" value="1" id="q_audio_1_opt1">
                <label for="q_audio_1_opt1">Option B</label>
            </div>
        </div>
    </div>
</div>
"""


def test_html_parsing_multimedia_extraction():
    """Verifica que el parser estático de HTML extraiga audio e imágenes de contenido filtrando iconos."""
    questions = DOMParser.parse_html_string(SAMPLE_MULTIMEDIA_HTML)
    assert len(questions) == 1
    q = questions[0]

    assert q.question_id == "q_audio_1"
    assert q.media_context.has_audio is True
    assert "https://campus.edu/pluginfile.php/999/quiz/intro/dialogue.mp3" in q.media_context.audio_urls
    assert "https://campus.edu/pluginfile.php/999/quiz/intro/backup_dialogue.wav" in q.media_context.audio_urls

    assert q.media_context.has_images is True
    assert "https://campus.edu/pluginfile.php/999/quiz/intro/scene_chart.png" in q.media_context.image_urls
    # El icono de /pix/icon.png debe haber sido descartado
    assert "https://campus.edu/pix/icon.png" not in q.media_context.image_urls
    print("✓ Extracción de audio y filtrado de iconos en HTML pasó exitosamente!")


def test_raw_json_multimedia_parsing():
    """Verifica que parse_raw_extracted_list procese correctamente los campos de audio y OCR."""
    raw_payload = [
        {
            "question_id": "q100",
            "number": "Pregunta 5",
            "qtype": "multichoice",
            "prompt": "Listen to the recording.",
            "media_context": {
                "has_audio": True,
                "audio_urls": ["http://moodle.com/audio.mp3"],
                "audio_transcriptions": ["Speaker 1: Where is the train? Speaker 2: At platform 3."],
                "has_images": True,
                "image_urls": ["http://moodle.com/schedule.png"],
                "ocr_texts": ["Departure: 14:30 Platform: 3"],
                "equations": [],
                "tables": []
            },
            "options": [
                {"id": "opt1", "text": "Platform 3", "value": "1", "selector": "#opt1", "is_selected": False}
            ],
            "sub_questions": []
        }
    ]

    questions = DOMParser.parse_raw_extracted_list(raw_payload)
    assert len(questions) == 1
    q = questions[0]
    assert q.media_context.has_audio is True
    assert len(q.media_context.audio_transcriptions) == 1
    assert "Speaker 1: Where is the train?" in q.media_context.audio_transcriptions[0]
    assert len(q.media_context.ocr_texts) == 1
    assert "Platform: 3" in q.media_context.ocr_texts[0]
    print("✓ Parsing de payload crudo con transcripción de audio y OCR pasó exitosamente!")


def test_llm_media_info_formatting():
    """Verifica que LLMEngine formatee las transcripciones de Whisper y el texto OCR en el prompt."""
    cfg = Settings(groq_api_key="gsk_dummy_key_1234567890")
    engine = LLMEngine(cfg)

    q = QuestionData(
        question_id="q_test_listening",
        number="Pregunta 1",
        qtype=QuestionType.MULTICHOICE,
        prompt="According to the conversation, what time does the train depart?",
        media_context=MediaContext(
            has_audio=True,
            audio_urls=["http://test.edu/dialogue.mp3"],
            audio_transcriptions=["Speaker A: The train leaves at quarter past five."],
            has_images=True,
            image_urls=["http://test.edu/schedule.jpg"],
            ocr_texts=["Train Schedule: Depart 17:15 Destination: London"],
        ),
        options=[],
        sub_questions=[]
    )

    formatted = engine._format_media_info(q)
    assert "🎧 Transcripción de audio (Whisper)" in formatted
    assert "The train leaves at quarter past five." in formatted
    assert "🖼️ Texto extraído de imagen (OCR)" in formatted
    assert "Depart 17:15" in formatted
    print("✓ Formateo de contexto multimedia para LLM pasó exitosamente!")


if __name__ == "__main__":
    test_html_parsing_multimedia_extraction()
    test_raw_json_multimedia_parsing()
    test_llm_media_info_formatting()
    print("✓ Todos los tests de test_media_extraction.py pasaron exitosamente!")
