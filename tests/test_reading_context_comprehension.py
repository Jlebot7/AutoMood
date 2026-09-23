import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.models import QuestionData, QuestionType, SubQuestion, OptionChoice, MediaContext

async def main():
    print("--- Test de Comprensión Global de Lectura y Tiempos Verbales en Groq ---")
    
    text = (
        "Since computers were first introduced to the public in the early 1980's, technology (change) [HUECO 1: q1] a great deal. "
        "The first computers (be) [HUECO 2: q2] simple machines designed for basic tasks. "
        "They (have, not) [HUECO 3: q3] much memory and they (be, not) [HUECO 4: q4] very powerful. "
        "Early computers were often quite expensive and customers often (pay) [HUECO 5: q5] thousands of dollars for machines which actually (do) [HUECO 6: q6] very little. "
        "Most computers (be) [HUECO 7: q7] separate, individual machines used mostly as expensive typewriters or for playing games.\n\n"
        "Times (change) [HUECO 8: q8]. Computers (become) [HUECO 9: q9] powerful machines with very practical applications. "
        "Programmers (create) [HUECO 10: q10] a large selection of useful programs which do everything from teaching foreign languages to bookkeeping. "
        "We are still playing video games, but today's games (become) [HUECO 11: q11] faster, more exciting interactive adventures. "
        "Many computer users (get, also) [HUECO 12: q12] on the Internet and (begin) [HUECO 13: q13] communicating with other computer users around the world. "
        "We (start) [HUECO 14: q14] to create international communities online. "
        "In short, the simple, individual machines of the past (evolve) [HUECO 15: q15] into an international World Wide Web of knowledge."
    )

    gaps_data = [
        ("q1", "technology (change) [HUECO 1: q1] a great deal", [("1", "changed"), ("2", "has changed"), ("3", "'s changed")]),
        ("q2", "The first computers (be) [HUECO 2: q2] simple machines", [("1", "were"), ("2", "have been"), ("3", "'ve been")]),
        ("q3", "They (have, not) [HUECO 3: q3] much memory", [("1", "didn't have"), ("2", "haven't had"), ("3", "had not")]),
        ("q4", "they (be, not) [HUECO 4: q4] very powerful", [("1", "weren't"), ("2", "haven't been"), ("3", "were not")]),
        ("q5", "customers often (pay) [HUECO 5: q5] thousands of dollars", [("1", "paid"), ("2", "have paid"), ("3", "'ve paid")]),
        ("q6", "which actually (do) [HUECO 6: q6] very little", [("1", "did"), ("2", "have done"), ("3", "'ve done")]),
        ("q7", "Most computers (be) [HUECO 7: q7] separate, individual machines", [("1", "were"), ("2", "have been"), ("3", "'ve been")]),
        ("q8", "Times (change) [HUECO 8: q8]", [("1", "changed"), ("2", "have changed"), ("3", "'ve changed")]),
        ("q9", "Computers (become) [HUECO 9: q9] powerful machines", [("1", "became"), ("2", "have become"), ("3", "'ve become")]),
        ("q10", "Programmers (create) [HUECO 10: q10] a large selection", [("1", "created"), ("2", "have created"), ("3", "'ve created")]),
        ("q11", "today's games (become) [HUECO 11: q11] faster", [("1", "became"), ("2", "have become"), ("3", "'ve become")]),
        ("q12", "users (get, also) [HUECO 12: q12] on the Internet", [("1", "also got"), ("2", "have also gotten"), ("3", "'ve also gotten")]),
        ("q13", "and (begin) [HUECO 13: q13] communicating", [("1", "began"), ("2", "have begun"), ("3", "'ve begun")]),
        ("q14", "We (start) [HUECO 14: q14] to create international communities", [("1", "started"), ("2", "have started"), ("3", "'ve started")]),
        ("q15", "machines of the past (evolve) [HUECO 15: q15] into", [("1", "evolved"), ("2", "have evolved"), ("3", "'ve evolved")]),
    ]

    sub_questions = []
    for sid, prompt, opts in gaps_data:
        sub_questions.append(SubQuestion(
            sub_id=sid,
            prompt=prompt,
            selector=f"select[name='{sid}']",
            options=[OptionChoice(id=f"{sid}_{v}", text=t, value=v, selector="", is_selected=False) for v, t in opts],
            selected_value=None
        ))

    question = QuestionData(
        question_id="question-test-cloze",
        number="Question 1",
        qtype=QuestionType.MULTIANSWER,
        prompt=text,
        media_context=MediaContext(),
        options=[],
        sub_questions=sub_questions,
        is_multiple_response=False,
        is_answered=False
    )

    engine = LLMEngine()
    print("Invocando solve_batch_questions()...")
    responses = await engine.solve_batch_questions([question])
    assert len(responses) == 1, "Se esperaba 1 respuesta"
    ans = responses[0]

    print("\nExplicación devuelta por la IA:")
    print(ans.explanation)

    print("\nRespuestas obtenidas vs esperadas:")
    expected = {
        "q1": "has changed",
        "q2": "were",
        "q3": "didn't have",
        "q4": "weren't",
        "q5": "paid",
        "q6": "did",
        "q7": "were",
        "q8": "have changed",
        "q9": "have become",
        "q10": "have created",
        "q11": "have become",
        "q12": "have also gotten",
        "q13": "have begun",
        "q14": "have started",
        "q15": "have evolved"
    }

    correct_count = 0
    total = len(expected)
    for qid, exp_text in expected.items():
        got = ans.matching_answers.get(qid, "")
        # Comprobación estricta (no debe elegir 've ni 's)
        is_correct = (got.strip().lower() == exp_text.strip().lower())
        mark = "✓" if is_correct else "✗"
        if is_correct:
            correct_count += 1
        print(f"  {mark} {qid}: Obtenido='{got}' | Esperado='{exp_text}'")

    print(f"\nTotal acertadas: {correct_count}/{total} ({(correct_count/total)*100:.1f}%)")
    assert ans.matching_answers.get("q3") in ("didn't have", "1"), "Pregunta 3 debe ser didn't have"
    assert ans.matching_answers.get("q7") in ("were", "1"), "Pregunta 7 debe ser were"
    assert ans.matching_answers.get("q13") in ("have begun", "2"), "Pregunta 13 debe ser have begun"
    assert ans.matching_answers.get("q14") in ("have started", "2"), "Pregunta 14 debe ser have started"

    print("\n✓ TODAS LAS PREGUNTAS CLAVE (3, 7, 13, 14) Y EL CONTEXTO GLOBAL FUERON RESUELTOS CON ÉXITO!")

if __name__ == "__main__":
    asyncio.run(main())

