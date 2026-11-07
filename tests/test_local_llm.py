"""
Pruebas unitarias para LocalLLMEngine y el fallback automático desde Groq.
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from groq import RateLimitError
from moodle_assistant.config import Settings
from moodle_assistant.local_llm import LocalLLMEngine
from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.models import QuestionData, QuestionType, MediaContext, OptionChoice


def test_ollama_availability_offline():
    """Verifica que check_ollama_available retorne False de forma segura si no está corriendo el daemon."""
    cfg = Settings(ollama_base_url="http://127.0.0.1:9999")  # Puerto no utilizado
    engine = LocalLLMEngine(cfg)

    loop = asyncio.new_event_loop()
    try:
        available = loop.run_until_complete(engine.check_ollama_available())
    finally:
        loop.close()

    assert available is False
    print("✓ Comprobación de disponibilidad de Ollama offline pasó correctamente!")


def test_local_engine_with_ollama_mock():
    """Prueba que LocalLLMEngine procese correctamente peticiones hacia Ollama con respuesta simulada."""
    cfg = Settings(local_model_provider="ollama", ollama_model_id="qwen2.5:1.5b")
    engine = LocalLLMEngine(cfg)

    # Mock de _ensure_ollama_model y llamada HTTP a Ollama
    engine._ensure_ollama_model = AsyncMock()

    mock_json_content = '{"answers": [{"question_id": "q1", "explanation": "Prueba", "selected_options": ["opt1"], "confidence": 0.95}]}'
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"message": {"content": mock_json_content}})
        mock_post.return_value = mock_resp

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                engine.generate_response([{"role": "user", "content": "test"}])
            )
        finally:
            loop.close()

    assert "selected_options" in result
    assert "opt1" in result
    print("✓ LocalLLMEngine con Ollama mock generó respuesta exitosa!")


def test_llm_engine_automatic_fallback_on_413():
    """Prueba que LLMEngine active automáticamente LocalLLMEngine ante un error 413 de Groq."""
    cfg = Settings(
        groq_api_key="gsk_test_key_valid_mock_1234567890",
        enable_local_fallback=True,
    )
    llm = LLMEngine(cfg)

    # Simular que Groq lanza RateLimitError 413
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError(
            message="Error code: 413 - {'error': {'message': 'Request too large for model'}}",
            response=MagicMock(status_code=413),
            body={"code": "rate_limit_exceeded"}
        )
    )
    llm._client = mock_groq_client

    # Simular respuesta del motor local
    mock_local = MagicMock()
    local_json_response = '''{
        "question_id": "q1",
        "explanation": "Resuelto con éxito mediante modelo local de respaldo",
        "selected_options": ["opt1"],
        "text_answer": null,
        "matching_answers": null,
        "confidence": 0.92
    }'''
    mock_local.generate_response = AsyncMock(return_value=local_json_response)
    llm.local_engine = mock_local

    q = QuestionData(
        question_id="q1",
        number="Pregunta 1",
        qtype=QuestionType.MULTICHOICE,
        prompt="¿Cuál es el valor de x?",
        media_context=MediaContext(),
        options=[OptionChoice(id="opt1", text="x = 5", value="1", selector="#opt1", is_selected=False)],
        sub_questions=[]
    )

    loop = asyncio.new_event_loop()
    try:
        response = loop.run_until_complete(llm.solve_single_question(q))
    finally:
        loop.close()

    assert response.question_id == "q1"
    assert response.selected_options == ["opt1"]
    assert "modelo local de respaldo" in response.explanation
    assert mock_local.generate_response.call_count == 1
    print("✓ Fallback automático ante error 413 de Groq ejecutado exitosamente!")


def test_compute_memory_distribution():
    """Prueba que compute_memory_distribution calcule los límites de VRAM (95%) y desborde a RAM."""
    cfg = Settings()
    engine = LocalLLMEngine(cfg)

    # Simular 4096MB total, 3000MB libre en GPU 0 y 16000MB RAM disponible
    with patch("torch.cuda.mem_get_info", return_value=(3000 * 1024 * 1024, 4096 * 1024 * 1024)):
        with patch("psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=16000 * 1024 * 1024)
            mem_map = engine.compute_memory_distribution()

    assert 0 in mem_map
    assert "cpu" in mem_map
    # 95% de 3000MB libre = 2850MB
    assert mem_map[0] == "2850MB"
    # 85% de 16000MB = 13600MB
    assert mem_map["cpu"] == "13600MB"
    print(f"✓ compute_memory_distribution asignó correctamente: {mem_map}")


if __name__ == "__main__":
    test_ollama_availability_offline()
    test_local_engine_with_ollama_mock()
    test_llm_engine_automatic_fallback_on_413()
    test_compute_memory_distribution()
    print("✓ Todos los tests de test_local_llm.py pasaron exitosamente!")
