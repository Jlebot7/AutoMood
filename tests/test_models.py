"""
Pruebas unitarias para models.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Agregar directorio raíz al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.models import (
    QuestionData,
    QuestionType,
    OptionChoice,
    SubQuestion,
    MediaContext,
    LLMResponse,
    BatchLLMResponse,
    ActionExecutionResult,
)


def test_question_data_model():
    q = QuestionData(
        question_id="q1",
        number="Pregunta 1",
        qtype=QuestionType.MULTICHOICE,
        prompt="¿Cuál es la complejidad temporal de búsqueda binaria?",
        options=[
            OptionChoice(id="opt1", text="O(n)", value="0", selector="#opt1"),
            OptionChoice(id="opt2", text="O(log n)", value="1", selector="#opt2"),
            OptionChoice(id="opt3", text="O(n^2)", value="2", selector="#opt3"),
        ],
        is_multiple_response=False,
    )
    assert q.question_id == "q1"
    assert q.qtype == QuestionType.MULTICHOICE
    assert len(q.options) == 3
    assert q.options[1].text == "O(log n)"


def test_llm_response_validation():
    json_payload = {
        "question_id": "q1",
        "explanation": "La búsqueda binaria divide el espacio a la mitad en cada paso, por lo tanto es logarítmica.",
        "selected_options": ["opt2"],
        "text_answer": None,
        "matching_answers": None,
        "confidence": 0.99,
    }
    resp = LLMResponse.model_validate(json_payload)
    assert resp.question_id == "q1"
    assert resp.confidence == 0.99
    assert resp.selected_options == ["opt2"]


def test_batch_llm_response_validation():
    json_batch = {
        "answers": [
            {
                "question_id": "q1",
                "explanation": "Respuesta correcta A",
                "selected_options": ["A"],
                "confidence": 0.95
            },
            {
                "question_id": "q2",
                "explanation": "Respuesta correcta B",
                "selected_options": ["B"],
                "confidence": 0.90
            }
        ]
    }
    batch = BatchLLMResponse.model_validate(json_batch)
    assert len(batch.answers) == 2
    assert batch.answers[0].question_id == "q1"


def test_matching_subquestion_model():
    sq = SubQuestion(
        sub_id="sub_0",
        prompt="Python",
        selector="select[name='sub_0']",
        options=[
            OptionChoice(id="o1", text="Interpretado", value="1", selector="select option[value='1']"),
            OptionChoice(id="o2", text="Compilado", value="2", selector="select option[value='2']"),
        ]
    )
    assert sq.sub_id == "sub_0"
    assert len(sq.options) == 2


if __name__ == "__main__":
    test_question_data_model()
    test_llm_response_validation()
    test_batch_llm_response_validation()
    test_matching_subquestion_model()
    print("✓ Todos los tests de models.py pasaron exitosamente!")
