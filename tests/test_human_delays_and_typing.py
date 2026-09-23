import asyncio
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import nodriver as uc
from moodle_assistant.automator import MoodleAutomator
from moodle_assistant.config import settings

async def main():
    print("--- Test de Delays Humanizados y Tecleo Natural ---")
    
    # Reducir tiempos en prueba para velocidad del test
    settings.typing_speed_min = 0.01
    settings.typing_speed_max = 0.03
    settings.option_delay_min = 0.2
    settings.option_delay_max = 0.4
    settings.human_delay_min = 0.1
    settings.human_delay_max = 0.2

    automator = MoodleAutomator(cfg=settings)
    browser = await uc.start(headless=True)
    automator.browser = browser

    try:
        tab = await browser.get("about:blank")
        automator._tab = tab

        # Crear formulario de prueba con input y select
        await tab.evaluate("""
        (() => {
            document.body.innerHTML = `
                <div class="que">
                    <input id="q1_text" type="text" value="" />
                    <select id="q1_select">
                        <option value="">Selecciona...</option>
                        <option value="1">Opción Uno</option>
                        <option value="2">Opción Dos</option>
                        <option value="3">Opción Tres</option>
                    </select>
                </div>
            `;
            window.inputEventCount = 0;
            window.changeEventCount = 0;
            
            const inp = document.getElementById("q1_text");
            inp.addEventListener("input", (e) => {
                window.inputEventCount++;
            });
            inp.addEventListener("change", (e) => {
                window.changeEventCount++;
            });

            const sel = document.getElementById("q1_select");
            sel.addEventListener("change", (e) => {
                window.selectChangeCount = (window.selectChangeCount || 0) + 1;
            });
        })()
        """)

        # 1. Probar tecleo natural en input
        print("\n1. Probando tecleo humanizado en input de texto...")
        test_text = "Hello, world!"
        t0 = time.time()
        ok_type = await automator.type_into_input("#q1_text", test_text)
        t_duration = time.time() - t0
        print(f"   Resultado de escritura: {ok_type} en {t_duration:.2f}s")
        assert ok_type is True

        res = await tab.evaluate("""
        (() => {
            const inp = document.getElementById("q1_text");
            return {
                val: inp.value,
                inputCount: window.inputEventCount,
                changeCount: window.changeEventCount
            };
        })()
        """)
        val = res[0][1]['value']
        input_count = res[1][1]['value']
        change_count = res[2][1]['value']
        print(f"   Valor escrito en DOM: '{val}'")
        print(f"   Eventos 'input' capturados: {input_count} (esperados >= {len(test_text)})")
        print(f"   Eventos 'change' capturados: {change_count}")
        assert val == test_text, f"Se esperaba '{test_text}', se obtuvo '{val}'"
        assert input_count >= len(test_text), f"No se dispararon los eventos de input por carácter: {input_count}"
        assert change_count >= 1, "No se disparó el evento change final"

        # 2. Probar selección de dropdown con delays
        print("\n2. Probando selección humanizada en dropdown (<select>)...")
        t0 = time.time()
        ok_sel = await automator.select_dropdown_option("#q1_select", "Opción Dos")
        t_sel_duration = time.time() - t0
        print(f"   Resultado selección dropdown: {ok_sel} en {t_sel_duration:.2f}s")
        assert ok_sel is True

        sel_res = await tab.evaluate("""
        (() => {
            const sel = document.getElementById("q1_select");
            return {
                val: sel.value,
                selectedIndex: sel.selectedIndex,
                selectedText: sel.options[sel.selectedIndex].text,
                changeCount: window.selectChangeCount || 0
            };
        })()
        """)
        sel_val = sel_res[0][1]['value']
        sel_idx = sel_res[1][1]['value']
        sel_txt = sel_res[2][1]['value']
        sel_changes = sel_res[3][1]['value']
        print(f"   Valor seleccionado en DOM: '{sel_val}' (Index {sel_idx}, Texto: '{sel_txt}')")
        print(f"   Eventos change en select: {sel_changes}")
        assert sel_val == "2"
        assert sel_txt == "Opción Dos"
        assert sel_changes >= 1

        # 3. Probar option_delay
        print("\n3. Probando option_delay()...")
        t0 = time.time()
        await automator.option_delay()
        dt = time.time() - t0
        print(f"   Tiempo de delay entre opciones: {dt:.2f}s")
        assert dt >= 0.15

        print("\n✓ TODAS LAS PRUEBAS DE DELAYS Y TECLEO NATURAL PASARON EXITOSAMENTE!")

    finally:
        try:
            automator.stop()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
