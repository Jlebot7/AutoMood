"""
Pruebas unitarias para el sanitizador de texto de huecos (sanitize_gap_text).
Verifica que las oraciones completas repetidas por alucinación se reduzcan
exclusivamente al modal/palabra requerida, mientras que las oraciones de dictado se preservan.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.automator import MoodleAutomator


def test_sanitize_atomic_values():
    """Valores de 1 o 2 palabras deben preservarse sin alteración."""
    assert MoodleAutomator.sanitize_gap_text("should") == "should"
    assert MoodleAutomator.sanitize_gap_text("could") == "could"
    assert MoodleAutomator.sanitize_gap_text("didn't have") == "didn't have"
    assert MoodleAutomator.sanitize_gap_text("has changed") == "has changed"
    assert MoodleAutomator.sanitize_gap_text("'would'") == "would"
    assert MoodleAutomator.sanitize_gap_text('"must"') == "must"
    print("✓ test_sanitize_atomic_values pasó correctamente.")


def test_sanitize_echoed_sentences_modal_extraction():
    """Si la IA devuelve la oración completa, extraer solo el verbo modal."""
    sq_prompt_11 = "[HUECO 11] You [HUECO 11] eat more vegetables to stay healthy."
    raw_11 = "You should eat more vegetables to stay healthy."
    clean_11 = MoodleAutomator.sanitize_gap_text(raw_11, sq_prompt=sq_prompt_11, gap_num=11)
    assert clean_11 == "should", f"Se esperaba 'should', se obtuvo '{clean_11}'"

    sq_prompt_12 = "[HUECO 12] He [HUECO 12] drive a car when he was only 15."
    raw_12 = "He could drive a car when he was only 15."
    clean_12 = MoodleAutomator.sanitize_gap_text(raw_12, sq_prompt=sq_prompt_12, gap_num=12)
    assert clean_12 == "could", f"Se esperaba 'could', se obtuvo '{clean_12}'"

    sq_prompt_15 = "[HUECO 15] We [HUECO 15] not park here, it's illegal."
    raw_15 = "We must not park here, it's illegal."
    clean_15 = MoodleAutomator.sanitize_gap_text(raw_15, sq_prompt=sq_prompt_15, gap_num=15)
    assert clean_15 in ("must not", "must"), f"Se esperaba 'must not' o 'must', se obtuvo '{clean_15}'"

    print("✓ test_sanitize_echoed_sentences_modal_extraction pasó correctamente.")


def test_sanitize_prefix_suffix_stripping():
    """Si no es un verbo modal estándar pero repite prefijo y sufijo, recortarlos."""
    sq_prompt = "[HUECO 4] Computers [HUECO 4] a central part of our lives."
    raw = "Computers have become a central part of our lives."
    clean = MoodleAutomator.sanitize_gap_text(raw, sq_prompt=sq_prompt, gap_num=4)
    assert clean == "have become", f"Se esperaba 'have become', se obtuvo '{clean}'"
    print("✓ test_sanitize_prefix_suffix_stripping pasó correctamente.")


def test_sanitize_dictation_sentences_preserved():
    """En secciones de dictado, las oraciones completas deben conservarse íntegras."""
    full_prompt = "Section 5: Dictation. Listen to each sentence and write exactly what you hear."
    sq_prompt_37 = "[HUECO 37] 1."
    raw_37 = "1. He ran much faster when he was younger."
    clean_37 = MoodleAutomator.sanitize_gap_text(
        raw_37,
        sq_prompt=sq_prompt_37,
        full_prompt=full_prompt,
        gap_num=37
    )
    assert clean_37 == "He ran much faster when he was younger.", f"Se esperaba oración limpia sin prefijo '1.', se obtuvo '{clean_37}'"

    raw_38 = "We should leave now before it gets dark."
    clean_38 = MoodleAutomator.sanitize_gap_text(
        raw_38,
        sq_prompt="[HUECO 38] 2.",
        full_prompt=full_prompt,
        gap_num=38
    )
    assert clean_38 == "We should leave now before it gets dark.", f"Se esperaba oración intacta, se obtuvo '{clean_38}'"
    print("✓ test_sanitize_dictation_sentences_preserved pasó correctamente.")


if __name__ == "__main__":
    test_sanitize_atomic_values()
    test_sanitize_echoed_sentences_modal_extraction()
    test_sanitize_prefix_suffix_stripping()
    test_sanitize_dictation_sentences_preserved()
    print("✓ Todos los tests de test_gap_sanitizer.py pasaron exitosamente!")
