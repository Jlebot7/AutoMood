"""
Pruebas unitarias para llm_engine.py
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.models import (
    QuestionData,
    QuestionType,
    OptionChoice,
    MediaContext,
    LLMResponse,
)
from moodle_assistant.config import Settings


def test_clean_json_response():
    engine = LLMEngine(Settings(groq_api_key="mock_key"))
    
    # Caso 1: markdown con ```json
    markdown_wrapped = "```json\n{\"question_id\": \"q1\", \"selected_options\": [\"opt1\"], \"confidence\": 0.9, \"explanation\": \"ok\"}\n```"
    cleaned = engine._clean_json_response(markdown_wrapped)
    assert cleaned.startswith("{")
    assert json.loads(cleaned)["question_id"] == "q1"

    # Caso 2: texto sin markdown
    pure_json = "{\"status\": \"ok\"}"
    assert engine._clean_json_response(pure_json) == pure_json


def test_prompt_formatting():
    engine = LLMEngine(Settings(groq_api_key="mock_key"))
    q = QuestionData(
        question_id="q_math",
        number="Pregunta 3",
        qtype=QuestionType.MULTICHOICE,
        prompt="¿Cuál es la integral de 2x dx?",
        media_context=MediaContext(
            has_images=False,
            equations=["\\int 2x dx = x^2 + C"]
        ),
        options=[
            OptionChoice(id="o1", text="x^2 + C", value="0", selector="#o1"),
            OptionChoice(id="o2", text="2x^2 + C", value="1", selector="#o2"),
        ]
    )

    media_info = engine._format_media_info(q)
    assert "\\int 2x dx = x^2 + C" in media_info

    opts_info = engine._format_options_info(q)
    assert "x^2 + C" in opts_info
    assert "o1" in opts_info


if __name__ == "__main__":
    test_clean_json_response()
    test_prompt_formatting()
    print("✓ Todos los tests de llm_engine.py pasaron exitosamente!")

