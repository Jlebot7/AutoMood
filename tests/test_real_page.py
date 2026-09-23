"""
Prueba de integración con la página real de Moodle descargada en testpage/
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

def test_real_moodle_cloze_page():
    html_file = root_dir / "testpage" / "Activity 2. Present Perfect Tense Short answer _ Campus.html"
    if not html_file.is_file():
        print("Página de test no encontrada, saltando prueba.")
        return

    content = html_file.read_text(encoding="utf-8", errors="ignore")
    questions = DOMParser.parse_html_string(content)

    assert len(questions) >= 1
    q = next(item for item in questions if item.question_id == "question-123244-1")
    assert q.qtype == QuestionType.MULTIANSWER
    assert len(q.sub_questions) == 18

    # Verificar que la opción con value="0" NO se descartó
    first_sq = q.sub_questions[0]
    vals = [opt.value for opt in first_sq.options]
    assert "0" in vals
    assert "1" in vals
    assert "2" in vals


if __name__ == "__main__":
    test_real_moodle_cloze_page()
    print("✓ Test de integración con página real de Moodle pasó exitosamente!")

