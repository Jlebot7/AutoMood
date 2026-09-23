import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import nodriver as uc
from moodle_assistant.automator import MoodleAutomator
from moodle_assistant.models import QuestionData, LLMResponse, QuestionType

async def main():
    test_html_path = (root_dir / "testpage" / "Activity 2. Present Perfect Tense Short answer _ Campus.html").resolve()
    url = f"file:///{str(test_html_path).replace('\\', '/')}"
    print(f"--- Iniciando Simulación E2E con nodriver ---")
    print(f"URL: {url}")
    
    automator = MoodleAutomator()
    browser = await uc.start(headless=True)
    automator.browser = browser
    
    try:
        tab = await browser.get(url)
        automator._tab = tab
        await asyncio.sleep(2)
        
        # 1. Extracción de preguntas
        print("\n1. Probando automator.extract_questions()...")
        questions = await automator.extract_questions()
        print(f"Preguntas detectadas: {len(questions)}")
        assert len(questions) == 1, f"Se esperaba 1 pregunta, se obtuvieron {len(questions)}"
        q = questions[0]
        print(f"Pregunta ID: {q.question_id}, Tipo: {q.qtype}, Subpreguntas: {len(q.sub_questions)}")
        assert q.qtype == QuestionType.MULTIANSWER
        assert len(q.sub_questions) == 18, f"Se esperaban 18 subpreguntas, se obtuvieron {len(q.sub_questions)}"
        
        # 2. Construir respuesta con las opciones válidas para cada subpregunta
        print("\n2. Generando respuestas para los 18 desplegables...")
        matching_answers = {}
        for sq in q.sub_questions:
            # Seleccionar la última opción disponible para cada desplegable
            chosen_opt = sq.options[-1]
            matching_answers[sq.sub_id] = chosen_opt.text
            print(f"   Sub {sq.sub_id} -> eligiendo: '{chosen_opt.text}' (val={chosen_opt.value})")

        response = LLMResponse(
            question_id=q.question_id,
            explanation="Respuestas simuladas para el test E2E de Present Perfect",
            matching_answers=matching_answers,
            confidence=0.99
        )
        
        # 3. Aplicar soluciones en el DOM
        print("\n3. Aplicando solución con automator.apply_solution()...")
        res = await automator.apply_solution(q, response)
        print(f"Resultado de apply_solution: success={res.success}, action='{res.action_taken}'")
        assert res.success is True, f"Fallo al aplicar solución: {res.error_message}"
        assert "18/18" in res.action_taken, f"Se esperaba 18/18, pero fue: {res.action_taken}"
        
        # 4. Verificar directamente en el DOM que los 18 desplegables están seleccionados
        print("\n4. Verificando estado de los 18 desplegables en el DOM...")
        verification = await tab.evaluate("""
        (() => {
            const selects = document.querySelectorAll("div.que select");
            const details = [];
            let allSelected = true;
            for (let i = 0; i < selects.length; i++) {
                const s = selects[i];
                const selected = s.selectedIndex > 0 && s.value !== "";
                if (!selected) allSelected = false;
                details.push({
                    name: s.name,
                    selectedIndex: s.selectedIndex,
                    value: s.value,
                    selectedText: s.options[s.selectedIndex] ? s.options[s.selectedIndex].text : ""
                });
            }
            return { total: selects.length, allSelected: allSelected, details: details };
        })()
        """)
        
        # Parse verification
        details_list = []
        for item in verification:
            if item[0] == 'total':
                total = item[1]['value']
            elif item[0] == 'allSelected':
                all_selected = item[1]['value']
                
        print(f"Total desplegables verificados: {total}")
        print(f"¿Todos seleccionados correctamente (>0 y con valor)? {all_selected}")
        assert all_selected is True, "¡Error! No todos los desplegables quedaron seleccionados en el DOM."

        # 5. Probar autosave
        print("\n5. Probando trigger_moodle_autosave()...")
        autosave_ok = await automator.trigger_moodle_autosave()
        print(f"trigger_moodle_autosave ejecutado: {autosave_ok}")

        # 6. Probar click_next_page
        print("\n6. Probando click_next_page()...")
        # En una página file:/// local, hacer clic en submit puede navegar a la URL del form
        # Verificamos que el script localice el botón
        can_find_btn = await tab.evaluate("""
        (() => {
            const btn = document.querySelector("#mod_quiz-next-nav");
            return !!btn;
        })()
        """)
        print(f"Botón mod_quiz-next-nav existe en DOM: {can_find_btn}")
        assert can_find_btn is True

        print("\n" + "="*60)
        print("✓ TODAS LAS PRUEBAS E2E PASARON CON ÉXITO ROTUNDO!")
        print("="*60)
        
    finally:
        browser.stop()

if __name__ == "__main__":
    asyncio.run(main())
