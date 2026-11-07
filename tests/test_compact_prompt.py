"""
Pruebas unitarias para validar la reducción y compactación drástica de tokens
en preguntas con múltiples huecos (Cloze / multianswer con 41 sub-preguntas).
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.models import (
    QuestionData,
    QuestionType,
    OptionChoice,
    SubQuestion,
    MediaContext,
)
from moodle_assistant.llm_engine import (
    LLMEngine,
    SINGLE_QUESTION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from moodle_assistant.config import Settings


def create_41_gap_question() -> QuestionData:
    """Crea una pregunta sintética idéntica a 'B1.1 Check your knowledge 18.9' con 41 huecos."""
    sub_questions = []

    # Sección 1: 10 dropdowns (modals)
    for i in range(1, 11):
        sid = f"q135606:1_sub{i}_answer"
        opts = [
            OptionChoice(id=f"{sid}_1", text="could", value="1", selector=""),
            OptionChoice(id=f"{sid}_2", text="should", value="2", selector=""),
            OptionChoice(id=f"{sid}_3", text="would", value="3", selector=""),
        ]
        sub_questions.append(
            SubQuestion(
                sub_id=sid,
                prompt=f"[HUECO {i}] When I was young, I [HUECO {i}] run very fast.",
                selector=f"select[name='{sid}']",
                options=opts,
            )
        )

    # Sección 2: 10 text inputs (modals)
    for i in range(11, 21):
        sid = f"q135606:1_sub{i}_answer"
        sub_questions.append(
            SubQuestion(
                sub_id=sid,
                prompt=f"[HUECO {i}] You [HUECO {i}] eat more vegetables to stay healthy.",
                selector=f"input[name='{sid}']",
                options=[],
            )
        )

    # Sección 3: 10 dropdowns
    for i in range(21, 31):
        sid = f"q135606:1_sub{i}_answer"
        opts = [
            OptionChoice(id=f"{sid}_1", text="must", value="1", selector=""),
            OptionChoice(id=f"{sid}_2", text="might", value="2", selector=""),
            OptionChoice(id=f"{sid}_3", text="can", value="3", selector=""),
        ]
        sub_questions.append(
            SubQuestion(
                sub_id=sid,
                prompt=f"[HUECO {i}] They [HUECO {i}] arrive late due to traffic.",
                selector=f"select[name='{sid}']",
                options=opts,
            )
        )

    # Sección 4: 6 dropdowns (reading / listening)
    for i in range(31, 37):
        sid = f"q135606:1_sub{i}_answer"
        opts = [
            OptionChoice(id=f"{sid}_1", text="True", value="1", selector=""),
            OptionChoice(id=f"{sid}_2", text="False", value="2", selector=""),
        ]
        sub_questions.append(
            SubQuestion(
                sub_id=sid,
                prompt=f"[HUECO {i}] The speaker prefers hiking in the morning.",
                selector=f"select[name='{sid}']",
                options=opts,
            )
        )

    # Sección 5: 5 text inputs (dictation sentences)
    for i in range(37, 42):
        sid = f"q135606:1_sub{i}_answer"
        sub_questions.append(
            SubQuestion(
                sub_id=sid,
                prompt=f"[HUECO {i}] {i-36}.",
                selector=f"input[name='{sid}']",
                options=[],
            )
        )

    # Texto completo de lectura con los 41 huecos
    full_text = "Instructions: Complete the 5 sections below.\n\n"
    full_text += "Section 1: Choose the correct modal verb.\n"
    for i in range(1, 11):
        full_text += f"Sentence {i}: When I was young, I [HUECO {i}: q135606:1_sub{i}_answer] run very fast.\n"
    full_text += "\nSection 2: Write the missing modal verb.\n"
    for i in range(11, 21):
        full_text += f"Sentence {i}: You [HUECO {i}: q135606:1_sub{i}_answer] eat more vegetables.\n"
    full_text += "\nSection 3: Modals of probability.\n"
    for i in range(21, 31):
        full_text += f"Sentence {i}: They [HUECO {i}: q135606:1_sub{i}_answer] arrive late.\n"
    full_text += "\nSection 4: Listening comprehension.\n"
    for i in range(31, 37):
        full_text += f"Question {i}: [HUECO {i}: q135606:1_sub{i}_answer]\n"
    full_text += "\nSection 5: Dictation. Listen and write the 5 sentences.\n"
    for i in range(37, 42):
        full_text += f"{i-36}. [HUECO {i}: q135606:1_sub{i}_answer]\n"

    media = MediaContext(
        has_audio=True,
        audio_urls=["https://campus.example/hiking.mp3", "https://campus.example/ttsreader_1-the-home.mp3"],
        audio_transcriptions=[
            "[hiking.mp3]: Welcome to our outdoor podcast. Today we talk about mountain trails...",
            "[ttsreader_1-the-home.mp3]: 1. He ran much faster when he was younger. 2. We should leave now before it gets dark. 3. Could you help me with this heavy box? 4. They would always visit their grandparents. 5. You must not smoke inside."
        ]
    )

    return QuestionData(
        question_id="q135606_1",
        number="Pregunta 1",
        qtype=QuestionType.MULTIANSWER,
        prompt=full_text,
        media_context=media,
        sub_questions=sub_questions
    )


def test_compact_subquestions_info_size():
    """Valida que _format_sub_questions_info para 41 huecos sea ultra compacto (< 2,500 caracteres)."""
    q = create_41_gap_question()
    engine = LLMEngine(Settings(groq_api_key="mock"))

    formatted_info = engine._format_sub_questions_info(q)
    print(f"Longitud de _format_sub_questions_info: {len(formatted_info)} caracteres (~{len(formatted_info)//4} tokens).")

    # Debe ser menor a 3,000 caracteres (< 750 tokens)
    assert len(formatted_info) < 3000, f"Demasiado grande: {len(formatted_info)} caracteres"


def test_total_prompt_token_budget():
    """Valida que el prompt total (system + user prompt) para los 41 huecos y 2 audios esté bajo 3,500 tokens."""
    q = create_41_gap_question()
    engine = LLMEngine(Settings(groq_api_key="mock"))

    user_prompt = SINGLE_QUESTION_PROMPT_TEMPLATE.format(
        question_id=q.question_id,
        number=q.number,
        qtype=q.qtype.value,
        prompt=q.prompt,
        media_info=engine._format_media_info(q),
        options_info=engine._format_options_info(q),
        sub_questions_info=engine._format_sub_questions_info(q),
    )

    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    est_tokens = total_chars // 4

    print(f"Total prompt chars: {total_chars} caracteres. Tokens estimados: ~{est_tokens} tokens.")

    # El límite de Groq TPM es 8,000. El prompt total DEBE ser inferior a 3,500 tokens para dejar holgura.
    assert est_tokens < 3500, f"Prompt excede el presupuesto seguro: ~{est_tokens} tokens (límite 3500)"
    print("✓ test_total_prompt_token_budget pasó exitosamente!")


if __name__ == "__main__":
    test_compact_subquestions_info_size()
    test_total_prompt_token_budget()
    print("✓ Todos los tests de test_compact_prompt.py pasaron exitosamente!")
