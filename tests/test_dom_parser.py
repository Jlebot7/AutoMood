"""
Pruebas unitarias para dom_parser.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.dom_parser import DOMParser
from moodle_assistant.models import QuestionType

SAMPLE_MOODLE_HTML = """
<div id="q1" class="que multichoice notyetanswered">
    <div class="info">
        <h3 class="no">Pregunta 1</h3>
        <div class="state">Sin responder aún</div>
        <div class="grade">Puntúa como 1,00</div>
    </div>
    <div class="content">
        <div class="formulation">
            <div class="qtext">
                Marcar con bandera la pregunta. ¿Cuál es el puerto estándar del protocolo HTTPS?
            </div>
            <div class="answer">
                <div class="r0">
                    <input type="radio" name="q1:1_answer" id="q1:1_answer0" value="0">
                    <label for="q1:1_answer0">a. 80</label>
                </div>
                <div class="r1">
                    <input type="radio" name="q1:1_answer" id="q1:1_answer1" value="1">
                    <label for="q1:1_answer1">b. 443</label>
                </div>
                <div class="r0">
                    <input type="radio" name="q1:1_answer" id="q1:1_answer2" value="2">
                    <label for="q1:1_answer2">c. 8080</label>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="q2" class="que match notyetanswered">
    <div class="info">
        <h3 class="no">Pregunta 2</h3>
    </div>
    <div class="content">
        <div class="formulation">
            <div class="qtext">Relaciona cada lenguaje con su paradigma principal:</div>
            <div class="answer">
                <table>
                    <tr>
                        <td class="text">Haskell</td>
                        <td class="control">
                            <select name="q2_sub0" id="q2_sub0">
                                <option value="0">Elegir...</option>
                                <option value="1">Funcional</option>
                                <option value="2">Orientado a Objetos</option>
                            </select>
                        </td>
                    </tr>
                    <tr>
                        <td class="text">Java</td>
                        <td class="control">
                            <select name="q2_sub1" id="q2_sub1">
                                <option value="0">Elegir...</option>
                                <option value="1">Funcional</option>
                                <option value="2">Orientado a Objetos</option>
                            </select>
                        </td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
</div>
"""


def test_clean_boilerplate():
    raw = "Marcar con bandera la pregunta Puntúa como 1,00 ¿Qué es un árbol AVL?"
    cleaned = DOMParser.clean_moodle_boilerplate(raw)
    assert "Marcar con bandera" not in cleaned
    assert "Puntúa como" not in cleaned
    assert "¿Qué es un árbol AVL?" in cleaned


def test_parse_html_string():
    questions = DOMParser.parse_html_string(SAMPLE_MOODLE_HTML)
    assert len(questions) == 2

    # Pregunta 1: multichoice
    q1 = questions[0]
    assert q1.question_id == "q1"
    assert q1.qtype == QuestionType.MULTICHOICE
    assert len(q1.options) == 3
    assert "HTTPS" in q1.prompt
    assert "Marcar con bandera" not in q1.prompt
    assert q1.options[1].value == "1"

    # Pregunta 2: match
    q2 = questions[1]
    assert q2.question_id == "q2"
    assert q2.qtype == QuestionType.MATCH
    assert len(q2.sub_questions) == 2
    assert len(q2.sub_questions[0].options) == 2
    assert q2.sub_questions[0].options[0].text == "Funcional"


def test_parse_raw_extracted_list():
    raw_list = [
        {
            "question_id": "q10",
            "number": "Pregunta 10",
            "qtype": "shortanswer",
            "prompt": "Capital de Francia",
            "media_context": {"has_images": False, "image_descriptions": [], "equations": [], "tables": []},
            "options": [{"id": "inp1", "text": "", "value": "", "selector": "#inp1", "is_selected": False}],
            "sub_questions": [],
            "is_multiple_response": False,
            "is_answered": False,
        }
    ]
    parsed = DOMParser.parse_raw_extracted_list(raw_list)
    assert len(parsed) == 1
    assert parsed[0].qtype == QuestionType.SHORTANSWER
    assert parsed[0].prompt == "Capital de Francia"


if __name__ == "__main__":
    test_clean_boilerplate()
    test_parse_html_string()
    test_parse_raw_extracted_list()
    print("✓ Todos los tests de dom_parser.py pasaron exitosamente!")

