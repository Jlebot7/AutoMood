"""
Pruebas unitarias para actividades Cloze / Multianswer híbridas (dropdowns + texto libre)
con filtrado estricto de controles falsos de reproductores multimedia (Video.js).
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.dom_parser import DOMParser
from moodle_assistant.models import QuestionData, QuestionType, LLMResponse
from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.config import Settings
from moodle_assistant.automator import MoodleAutomator


def _ensure_test_html(html_path: Path) -> str:
    """Retorna el contenido HTML del testpage, generándolo sintéticamente si no existe en disco."""
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()

    # Generar fixture sintético idéntico a Moodle con 41 huecos y controles de Video.js para filtrar
    html_path.parent.mkdir(parents=True, exist_ok=True)
    body_parts = ['<div id="q135606:1" class="que multianswer notyetanswered">', '<div class="formulation">']
    
    # Sección 1: 10 selects
    body_parts.append('<div class="section-1">')
    for i in range(1, 11):
        sid = f"q135606:1_sub{i}_answer"
        body_parts.append(
            f'Sentence {i}: When I was younger I <select name="{sid}" id="{sid}">'
            f'<option value="">Choose...</option><option value="1">could</option><option value="2">should</option><option value="3">would</option>'
            f'</select> run fast.<br>'
        )
    body_parts.append('</div>')

    # Controles falsos de Video.js que deben ser filtrados
    body_parts.append(
        '<div class="video-js"><div class="vjs-control-bar"><select id="vjs_select_1" name="vjs_select_1"><option value="1">1x</option></select></div></div>'
        '<div class="mediaplugin"><div class="vjs-menu"><select id="vjs_select_2" name="vjs_select_2"><option value="auto">Auto</option></select></div></div>'
    )

    # Sección 2: 10 text inputs
    body_parts.append('<div class="section-2">')
    for i in range(11, 21):
        sid = f"q135606:1_sub{i}_answer"
        body_parts.append(f'Sentence {i}: You <input type="text" name="{sid}" id="{sid}" value=""> eat more vegetables.<br>')
    body_parts.append('</div>')

    # Sección 3: 10 selects
    body_parts.append('<div class="section-3">')
    for i in range(21, 31):
        sid = f"q135606:1_sub{i}_answer"
        body_parts.append(
            f'Sentence {i}: They <select name="{sid}" id="{sid}">'
            f'<option value="">Choose...</option><option value="1">must</option><option value="2">might</option><option value="3">can</option>'
            f'</select> arrive late.<br>'
        )
    body_parts.append('</div>')

    # Sección 4: 6 selects
    body_parts.append('<div class="section-4">')
    for i in range(31, 37):
        sid = f"q135606:1_sub{i}_answer"
        body_parts.append(
            f'Sentence {i}: <select name="{sid}" id="{sid}">'
            f'<option value="">Choose...</option><option value="1">True</option><option value="2">False</option>'
            f'</select><br>'
        )
    body_parts.append('</div>')

    # Sección 5: 5 text inputs (dictation)
    body_parts.append('<div class="section-5">')
    for i in range(37, 42):
        sid = f"q135606:1_sub{i}_answer"
        body_parts.append(f'{i-36}. <input type="text" name="{sid}" id="{sid}" value=""><br>')
    body_parts.append('</div>')

    body_parts.append('</div></div>')
    html_content = "\n".join(body_parts)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_content


def test_hybrid_cloze_parsing_and_filtering():
    """Verifica que se extraigan exactamente 41 huecos reales y 0 controles de Video.js."""
    html_path = root_dir / "testpage" / "B1.1 Check your knowledge 18.9 (Option 2) _ Campus.html"
    html_content = _ensure_test_html(html_path)

    questions = DOMParser.parse_html_string(html_content)
    assert len(questions) == 1, f"Se esperaba 1 pregunta, se obtuvieron {len(questions)}"

    q = questions[0]
    assert q.qtype == QuestionType.MULTIANSWER
    assert len(q.sub_questions) == 41, f"Se esperaban 41 sub-preguntas, se obtuvieron {len(q.sub_questions)}"

    # Verificar que ningún sub_id contenga vjs
    for sq in q.sub_questions:
        assert "vjs_" not in sq.sub_id, f"Se detectó control falso de Video.js: {sq.sub_id}"
        assert "vjs-" not in sq.selector, f"Selector contiene vjs: {sq.selector}"

    # Verificar orden de controles: 1-10 selects, 11-20 inputs, etc.
    for i in range(10):
        sq = q.sub_questions[i]
        assert sq.sub_id == f"q135606:1_sub{i+1}_answer"
        assert len(sq.options) == 3, f"Se esperaban 3 opciones en {sq.sub_id}, hay {len(sq.options)}"

    for i in range(10, 20):
        sq = q.sub_questions[i]
        assert sq.sub_id == f"q135606:1_sub{i+1}_answer"
        assert len(sq.options) == 0, f"Se esperaban 0 opciones (campo de texto libre) en {sq.sub_id}, hay {len(sq.options)}"

    print(f"✓ Extracción de Cloze híbrido exitosa: {len(q.sub_questions)} huecos reales sin controles de Video.js!")


def test_hybrid_cloze_prompt_token_size():
    """Verifica que el prompt formateado para los 41 huecos no exceda la cuota de TPM."""
    html_path = root_dir / "testpage" / "B1.1 Check your knowledge 18.9 (Option 2) _ Campus.html"
    html_content = _ensure_test_html(html_path)

    questions = DOMParser.parse_html_string(html_content)
    q = questions[0]

    engine = LLMEngine(Settings(groq_api_key="mock"))
    formatted_sub = engine._format_sub_questions_info(q)

    # Debe ser menor a 10,000 caracteres (< 2,500 tokens)
    assert len(formatted_sub) < 10000, f"Prompt demasiado largo: {len(formatted_sub)} caracteres"
    print(f"✓ Control de tokens para 41 huecos: {len(formatted_sub)} caracteres (~{len(formatted_sub)//4} tokens).")


def test_automator_apply_solution_hybrid_cloze():
    """Verifica que automator despache dropdowns a select_dropdown_option e inputs a type_into_input."""
    html_path = root_dir / "testpage" / "B1.1 Check your knowledge 18.9 (Option 2) _ Campus.html"
    html_content = _ensure_test_html(html_path)

    questions = DOMParser.parse_html_string(html_content)
    q = questions[0]

    # Simular respuestas para los 41 huecos
    mock_matching = {}
    for i, sq in enumerate(q.sub_questions):
        if sq.options:
            mock_matching[sq.sub_id] = sq.options[0].value
        else:
            mock_matching[sq.sub_id] = "should"

    automator = MoodleAutomator(Settings())
    automator.select_dropdown_option = AsyncMock(return_value=True)
    automator.type_into_input = AsyncMock(return_value=True)
    automator.option_delay = AsyncMock()

    response = LLMResponse(
        question_id=q.question_id,
        explanation="Solución simulada de prueba",
        selected_options=[],
        text_answer=None,
        matching_answers=mock_matching,
        confidence=0.99
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(automator.apply_solution(q, response))
    finally:
        loop.close()

    assert result.success is True
    assert "Completados 41/41" in result.action_taken
    assert automator.select_dropdown_option.call_count == 26
    assert automator.type_into_input.call_count == 15
    print(f"✓ Aplicación de solución híbrida completada: {result.action_taken} (26 dropdowns + 15 inputs).")


if __name__ == "__main__":
    test_hybrid_cloze_parsing_and_filtering()
    test_hybrid_cloze_prompt_token_size()
    test_automator_apply_solution_hybrid_cloze()
    print("✓ Todos los tests de test_hybrid_cloze.py pasaron exitosamente!")
