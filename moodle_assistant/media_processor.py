"""
Media processor module for AutoMood.
Provides Audio Transcription (Groq Whisper, OpenAI Whisper, local Faster-Whisper)
and Image OCR (EasyOCR, OpenAI Vision) for multimedia-rich Moodle quizzes.
"""

import asyncio
import base64
import hashlib
import io
import logging
from typing import Dict, List, Optional

from .config import AppConfig, Settings

logger = logging.getLogger("AutoMood.MediaProcessor")


class AudioTranscriber:
    """
    Transcribes audio files using Groq Whisper, OpenAI Whisper, or local Faster-Whisper.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._cache: Dict[str, str] = {}
        self._groq_client = None
        self._openai_client = None
        self._faster_whisper_model = None

    def _get_cache_key(self, audio_bytes: bytes, url: Optional[str] = None) -> str:
        if url:
            return f"url:{url}"
        return f"hash:{hashlib.sha256(audio_bytes).hexdigest()}"

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        url: Optional[str] = None,
    ) -> str:
        """
        Transcribes audio bytes to text using the configured provider.
        Results are cached to avoid redundant network calls or computations.
        """
        if not audio_bytes:
            return ""

        cache_key = self._get_cache_key(audio_bytes, url)
        if cache_key in self._cache:
            logger.debug("Audio transcription cache hit for %s", cache_key)
            return self._cache[cache_key]

        raw_provider = (self.config.whisper_provider or "auto").lower().strip()
        provider = raw_provider
        if provider == "auto":
            if self.config.is_groq_configured():
                provider = "groq"
            elif self.config.is_openai_configured():
                provider = "openai"
            else:
                provider = "local"

        result_text = ""

        try:
            if provider == "groq":
                result_text = await self._transcribe_groq(audio_bytes, filename)
            elif provider == "openai":
                result_text = await self._transcribe_openai(audio_bytes, filename)
            elif provider == "local":
                result_text = await self._transcribe_local(audio_bytes)
            else:
                logger.warning("Proveedor de Whisper no estándar: '%s'. Intentando Groq.", provider)
                result_text = await self._transcribe_groq(audio_bytes, filename)
        except Exception as exc:
            logger.warning("Error durante la transcripción de audio con %s: %s", provider, exc)
            # Si Groq falla y OpenAI está configurado, intentar fallback con OpenAI
            if provider == "groq" and self.config.is_openai_configured():
                logger.info("Intentando fallback de audio con OpenAI Whisper...")
                try:
                    result_text = await self._transcribe_openai(audio_bytes, filename)
                except Exception as fallback_exc:
                    logger.warning("Fallback con OpenAI Whisper también falló: %s", fallback_exc)

        # Fallback de última línea: si el audio sigue sin transcribirse, usar faster-whisper local
        if not (result_text and result_text.strip()):
            logger.info("Activando fallback de audio local con faster-whisper...")
            try:
                result_text = await self._transcribe_local(audio_bytes)
            except Exception as local_exc:
                logger.warning("Fallback local faster-whisper también falló: %s", local_exc)

        result_text = (result_text or "").strip()
        if result_text:
            self._cache[cache_key] = result_text
            logger.info("Audio transcrito exitosamente (%d caracteres)", len(result_text))

        return result_text

    async def _transcribe_groq(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using Groq Whisper API (whisper-large-v3 / whisper-large-v3-turbo)."""
        if not self.config.is_groq_configured():
            raise ValueError("GROQ_API_KEY no está configurada para transcripción de audio.")

        if self._groq_client is None:
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(api_key=self.config.groq_api_key)

        model = self.config.whisper_model or "whisper-large-v3"
        logger.info("Transcribiendo audio con Groq Whisper (%s)...", model)

        response = await self._groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=model,
            response_format="text",
        )

        if isinstance(response, str):
            return response
        return getattr(response, "text", str(response))

    async def _transcribe_openai(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using OpenAI Whisper API (whisper-1)."""
        if not self.config.is_openai_configured():
            raise ValueError("OPENAI_API_KEY no está configurada para transcripción de audio.")

        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)

        logger.info("Transcribiendo audio con OpenAI Whisper (whisper-1)...")

        response = await self._openai_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-1",
            response_format="text",
        )

        if isinstance(response, str):
            return response
        return getattr(response, "text", str(response))

    async def _transcribe_local(self, audio_bytes: bytes) -> str:
        """Transcribe using local faster-whisper on CPU/GPU."""
        return await asyncio.to_thread(self._run_local_faster_whisper, audio_bytes)

    def _run_local_faster_whisper(self, audio_bytes: bytes) -> str:
        from faster_whisper import WhisperModel
        import torch

        if self._faster_whisper_model is None:
            model_size = "base"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info("Cargando modelo local faster-whisper ('%s') en %s (%s)...", model_size, device, compute_type)
            try:
                self._faster_whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as e:
                logger.warning("No se pudo iniciar faster-whisper en GPU (%s), usando CPU...", e)
                self._faster_whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, _ = self._faster_whisper_model.transcribe(io.BytesIO(audio_bytes))
        texts = [seg.text for seg in segments]
        return " ".join(texts).strip()


class ImageOCREngine:
    """
    Performs OCR on image bytes using EasyOCR (local) or OpenAI Vision.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._cache: Dict[str, str] = {}
        self._easyocr_reader = None
        self._openai_client = None

    def _get_cache_key(self, image_bytes: bytes, url: Optional[str] = None) -> str:
        if url:
            return f"url:{url}"
        return f"hash:{hashlib.sha256(image_bytes).hexdigest()}"

    async def extract_text_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
        url: Optional[str] = None,
    ) -> str:
        """
        Extracts textual content from image bytes.
        """
        if not image_bytes:
            return ""

        cache_key = self._get_cache_key(image_bytes, url)
        if cache_key in self._cache:
            logger.debug("Image OCR cache hit for %s", cache_key)
            return self._cache[cache_key]

        provider = (self.config.ocr_provider or "easyocr").lower().strip()
        result_text = ""

        try:
            if provider == "openai_vision" and self.config.is_openai_configured():
                result_text = await self._ocr_openai_vision(image_bytes)
            else:
                result_text = await self._ocr_easyocr(image_bytes)
        except Exception as exc:
            logger.warning("Error durante OCR con %s: %s", provider, exc)
            if provider == "openai_vision":
                logger.info("Reintentando OCR con EasyOCR local...")
                try:
                    result_text = await self._ocr_easyocr(image_bytes)
                except Exception as easyocr_exc:
                    logger.warning("EasyOCR fallback también falló: %s", easyocr_exc)

        result_text = result_text.strip()
        if result_text:
            self._cache[cache_key] = result_text
            logger.info("OCR extrajo %d caracteres del recurso de imagen", len(result_text))

        return result_text

    async def _ocr_easyocr(self, image_bytes: bytes) -> str:
        """Runs EasyOCR in an async worker thread."""
        return await asyncio.to_thread(self._run_easyocr_sync, image_bytes)

    def _run_easyocr_sync(self, image_bytes: bytes) -> str:
        if self._easyocr_reader is None:
            import easyocr
            langs = self.config.ocr_languages or ["en", "es"]
            logger.info("Inicializando EasyOCR Reader con idiomas: %s...", langs)
            self._easyocr_reader = easyocr.Reader(langs, gpu=False)

        results = self._easyocr_reader.readtext(image_bytes, detail=0)
        if not results:
            return ""
        return "\n".join(results)

    async def _ocr_openai_vision(self, image_bytes: bytes) -> str:
        """Extracts text or diagram contents via OpenAI Vision (gpt-4o-mini)."""
        if not self.config.is_openai_configured():
            raise ValueError("OPENAI_API_KEY no está configurada para OCR Vision.")

        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        logger.info("Extrayendo texto de imagen mediante OpenAI Vision (gpt-4o-mini)...")

        prompt = (
            "Read and extract all visible text, equations, tables, questions, labels, "
            "or diagrams from this image. Transcribe every word and detail accurately. "
            "Output ONLY the extracted text and transcription without conversational commentary."
        )

        response = await self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            temperature=0.0,
        )

        content = response.choices[0].message.content or ""
        return content.strip()
