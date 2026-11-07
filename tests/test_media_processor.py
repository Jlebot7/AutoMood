"""
Tests para AudioTranscriber e ImageOCREngine en AutoMood.
"""

import io
import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from PIL import Image, ImageDraw

from moodle_assistant.config import Settings
from moodle_assistant.media_processor import AudioTranscriber, ImageOCREngine


def create_test_image(text: str = "AutoMood Test 123") -> bytes:
    """Genera una imagen PNG sintética con texto legible para pruebas de OCR."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Dibujar texto simple
    draw.text((20, 35), text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_ocr_easyocr():
    """Prueba la extracción real de texto con EasyOCR sobre imagen sintética."""
    cfg = Settings(ocr_provider="easyocr", ocr_languages=["en"])
    engine = ImageOCREngine(cfg)

    img_bytes = create_test_image("AutoMood Test 123")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        extracted = loop.run_until_complete(
            engine.extract_text_from_bytes(img_bytes, filename="test.png", url="http://moodle.edu/img1.png")
        )
    finally:
        loop.close()

    assert "AutoMood" in extracted or "Test" in extracted or "123" in extracted, f"OCR falló en extraer texto esperado: '{extracted}'"

    # Verificar funcionamiento de la caché
    assert "url:http://moodle.edu/img1.png" in engine._cache
    assert engine._cache["url:http://moodle.edu/img1.png"] == extracted
    print(f"✓ EasyOCR extrajo correctamente: '{extracted}'")


def test_audio_transcriber_groq_mock():
    """Prueba AudioTranscriber invocando Groq Whisper con respuesta mockeada."""
    cfg = Settings(
        groq_api_key="gsk_test_key_valid_mock_1234567890",
        whisper_provider="groq",
        whisper_model="whisper-large-v3",
    )
    transcriber = AudioTranscriber(cfg)

    mock_client = MagicMock()
    mock_transcriptions = AsyncMock()
    mock_transcriptions.create = AsyncMock(return_value="Hello this is a listening test conversation.")
    mock_client.audio.transcriptions = mock_transcriptions
    transcriber._groq_client = mock_client

    sample_audio_bytes = b"ID3\x03\x00\x00\x00FAKE_AUDIO_DATA_FOR_TESTING"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            transcriber.transcribe_bytes(
                sample_audio_bytes,
                filename="listening.mp3",
                url="http://moodle.edu/audio/listening.mp3",
            )
        )
    finally:
        loop.close()

    assert result == "Hello this is a listening test conversation."
    assert "url:http://moodle.edu/audio/listening.mp3" in transcriber._cache

    # Segunda llamada debe ser instantánea desde caché sin llamar a create()
    loop2 = asyncio.new_event_loop()
    try:
        cached_res = loop2.run_until_complete(
            transcriber.transcribe_bytes(
                sample_audio_bytes,
                filename="listening.mp3",
                url="http://moodle.edu/audio/listening.mp3",
            )
        )
    finally:
        loop2.close()

    assert cached_res == result
    assert mock_transcriptions.create.call_count == 1
    print("✓ AudioTranscriber mock pasó exitosamente!")


def test_audio_transcriber_fallback():
    """Prueba que ante error en Groq, AudioTranscriber intenta fallback en OpenAI si está configurado."""
    cfg = Settings(
        groq_api_key="gsk_test_key_valid_mock_1234567890",
        openai_api_key="sk-openai-valid-test-key-12345",
        whisper_provider="groq",
    )
    transcriber = AudioTranscriber(cfg)

    # Forzar error en Groq
    mock_groq = MagicMock()
    mock_groq.audio.transcriptions.create = AsyncMock(side_effect=Exception("Groq API Timeout"))
    transcriber._groq_client = mock_groq

    # Mock OpenAI
    mock_openai = MagicMock()
    mock_openai.audio.transcriptions.create = AsyncMock(return_value="Fallback OpenAI transcription text.")
    transcriber._openai_client = mock_openai

    sample_audio_bytes = b"FAKE_AUDIO_DATA_FALLBACK"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            transcriber.transcribe_bytes(
                sample_audio_bytes,
                filename="test.mp3",
            )
        )
    finally:
        loop.close()

    assert result == "Fallback OpenAI transcription text."
    print("✓ AudioTranscriber fallback a OpenAI pasó exitosamente!")


def test_audio_transcriber_fallback_to_local_faster_whisper():
    """Prueba que si Groq falla y OpenAI no está configurado, hace fallback a faster-whisper local."""
    cfg = Settings(
        groq_api_key="gsk_test_key_valid_mock_1234567890",
        whisper_provider="groq",
    )
    transcriber = AudioTranscriber(cfg)

    # Forzar error en Groq
    mock_groq = MagicMock()
    mock_groq.audio.transcriptions.create = AsyncMock(side_effect=Exception("Groq Whisper 504 Gateway Timeout"))
    transcriber._groq_client = mock_groq

    # Mock faster-whisper local
    transcriber._transcribe_local = AsyncMock(return_value="Local faster-whisper transcription result.")

    sample_audio_bytes = b"FAKE_AUDIO_DATA_FOR_LOCAL_FALLBACK"

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            transcriber.transcribe_bytes(
                sample_audio_bytes,
                filename="dictation.mp3",
                url="https://campus.example/dictation.mp3",
            )
        )
    finally:
        loop.close()

    assert transcriber._transcribe_local.called
    assert result == "Local faster-whisper transcription result."
    print("✓ AudioTranscriber fallback a local faster-whisper pasó exitosamente!")


if __name__ == "__main__":
    test_image_ocr_easyocr()
    test_audio_transcriber_groq_mock()
    test_audio_transcriber_fallback()
    test_audio_transcriber_fallback_to_local_faster_whisper()
    print("✓ Todos los tests de test_media_processor.py pasaron exitosamente!")
