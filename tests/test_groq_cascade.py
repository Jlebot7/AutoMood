"""
Pruebas unitarias para la cascada inteligente de modelos en Groq
y fallback al modelo local cuando ocurre desbordamiento de cuota TPM (413).
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

from groq import RateLimitError
from moodle_assistant.llm_engine import LLMEngine
from moodle_assistant.config import Settings


def test_cascade_to_higher_tpm_groq_model():
    """Verifica que un error 413 en openai/gpt-oss-120b active la cascada hacia qwen/qwen3.8-27b."""
    engine = LLMEngine(Settings(groq_api_key="mock", groq_model="openai/gpt-oss-120b"))

    mock_client = MagicMock()
    models_called = []

    async def mock_create(model, messages, **kwargs):
        models_called.append(model)
        if model == "openai/gpt-oss-120b":
            # Simular Groq 413 RateLimitError
            mock_resp = MagicMock()
            mock_resp.status_code = 413
            mock_resp.headers = {}
            raise RateLimitError(
                message="Error code: 413 - {'error': {'message': 'Request too large on tokens per minute (TPM): Limit 8000, Requested 9340'}}",
                response=mock_resp,
                body={"error": {"code": "rate_limit_exceeded"}}
            )
        elif model == "qwen/qwen3.8-27b":
            res_obj = MagicMock()
            choice = MagicMock()
            choice.message.content = '{"answers": [{"question_id": "q1", "explanation": "OK", "matching_answers": {"sub1": "should"}}]}'
            res_obj.choices = [choice]
            return res_obj
        raise ValueError(f"Modelo no esperado: {model}")

    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
    engine._client = mock_client

    loop = asyncio.new_event_loop()
    try:
        content = loop.run_until_complete(
            engine._execute_with_retry([{"role": "user", "content": "test"}])
        )
    finally:
        loop.close()

    assert "openai/gpt-oss-120b" in models_called
    assert "qwen/qwen3.8-27b" in models_called
    assert "should" in content
    print(f"✓ Cascada de Groq exitosa: Modelos llamados: {models_called}")


def test_cascade_falls_back_to_local_llm_if_all_groq_fail():
    """Verifica que si todos los modelos de Groq arrojan 413, se recurra al motor local."""
    engine = LLMEngine(Settings(groq_api_key="mock", enable_local_fallback=True))

    mock_client = MagicMock()

    async def mock_create_all_fail(model, messages, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 413
        mock_resp.headers = {}
        raise RateLimitError(
            message=f"413 Request too large for {model}",
            response=mock_resp,
            body={"error": {"code": "rate_limit_exceeded"}}
        )

    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create_all_fail)
    engine._client = mock_client

    mock_local = MagicMock()
    mock_local.generate_response = AsyncMock(return_value='{"answers": [{"question_id": "q1", "explanation": "Local OK", "matching_answers": {"sub1": "could"}}]}')
    engine.local_engine = mock_local

    loop = asyncio.new_event_loop()
    try:
        content = loop.run_until_complete(
            engine._execute_with_retry([{"role": "user", "content": "test"}])
        )
    finally:
        loop.close()

    assert mock_local.generate_response.called
    assert "Local OK" in content
    print("✓ Fallback a Local LLM tras fallo total de cascada de Groq verificado exitosamente!")


if __name__ == "__main__":
    test_cascade_to_higher_tpm_groq_model()
    test_cascade_falls_back_to_local_llm_if_all_groq_fail()
    print("✓ Todos los tests de test_groq_cascade.py pasaron exitosamente!")
