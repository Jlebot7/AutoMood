"""
Motor de inferencia LLM local de respaldo (Fallback) para AutoMood.
Soporta:
1. Ollama (con aceleración nativa en GPU NVIDIA GeForce RTX 3050 si está disponible).
2. Hugging Face Transformers (ejecución directa en proceso con PyTorch en GPU o CPU).
Descarga automáticamente el modelo si no está presente en la máquina.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

# Deshabilitar advertencias de symlinks en Windows Developer Mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import httpx
from rich.console import Console

from .config import Settings

logger = logging.getLogger("AutoMood.LocalLLM")
console = Console()


class LocalLLMEngine:
    """
    Motor local de inferencia de respaldo para cuando Groq no esté disponible,
    exceda límites de tokens (413/429) o no haya conexión externa.
    """

    def __init__(self, config: Settings):
        self.config = config
        self._provider = config.local_model_provider or "auto"
        self._transformers_model = None
        self._transformers_tokenizer = None
        self._device = None
        self._is_ollama_available: Optional[bool] = None

    async def check_ollama_available(self) -> bool:
        """Comprueba si el servidor local de Ollama está activo y respondiendo."""
        if self._is_ollama_available is True:
            return True

        url = f"{self.config.ollama_base_url.rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    self._is_ollama_available = True
                    logger.info("Servidor Ollama detectado y listo en %s", url)
                    return True
        except Exception:
            pass

        # Si no responde pero ollama está instalado en el sistema (y la URL es localhost estándar), intentar levantarlo en segundo plano
        is_default_localhost = ("127.0.0.1:11434" in url or "localhost:11434" in url)
        if is_default_localhost and self._try_spawn_ollama_serve():
            for _ in range(8):
                await asyncio.sleep(0.5)
                try:
                    async with httpx.AsyncClient(timeout=1.0) as client:
                        res = await client.get(url)
                        if res.status_code == 200:
                            self._is_ollama_available = True
                            logger.info("Servidor Ollama iniciado y respondiendo en %s", url)
                            return True
                except Exception:
                    continue

        self._is_ollama_available = False
        return False

    def _try_spawn_ollama_serve(self) -> bool:
        """Intenta iniciar el proceso 'ollama serve' en segundo plano si no estaba corriendo."""
        import shutil
        import subprocess

        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            default_win_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
            if os.path.isfile(default_win_path):
                ollama_bin = default_win_path

        if not ollama_bin:
            return False

        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                **kwargs
            )
            logger.info("Proceso 'ollama serve' iniciado en segundo plano desde %s", ollama_bin)
            return True
        except Exception as e:
            logger.debug("No se pudo iniciar 'ollama serve' automáticamente: %s", e)
            return False

    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Genera una respuesta en formato JSON utilizando el motor local disponible
        (Ollama con GPU o Transformers en proceso).
        """
        provider = self._provider

        if provider == "auto":
            # Preferir Ollama si está activo (mayor velocidad con aceleración GPU)
            has_ollama = await self.check_ollama_available()
            if has_ollama:
                provider = "ollama"
            else:
                provider = "transformers"

        if provider == "ollama":
            try:
                return await self._generate_with_ollama(messages)
            except Exception as e:
                logger.warning("Fallo al generar con Ollama (%s). Intentando Transformers...", e)
                return await self._generate_with_transformers(messages)
        else:
            return await self._generate_with_transformers(messages)

    # -------------------------------------------------------------------------
    # Implementación con Ollama (GPU nativa)
    # -------------------------------------------------------------------------
    async def _generate_with_ollama(self, messages: List[Dict[str, str]]) -> str:
        model_name = self.config.ollama_model_id or "qwen2.5:1.5b"
        base_url = self.config.ollama_base_url.rstrip('/')

        # 1. Asegurar que el modelo esté descargado en Ollama
        await self._ensure_ollama_model(base_url, model_name)

        # 2. Enviar petición de chat estructurado JSON
        chat_url = f"{base_url}/api/chat"
        payload = {
            "model": model_name,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": self.config.local_model_max_new_tokens,
            }
        }

        logger.info("Generando respuesta con Ollama (%s)...", model_name)
        async with httpx.AsyncClient(timeout=180.0) as client:
            res = await client.post(chat_url, json=payload)
            res.raise_for_status()
            data = res.json()
            content = data.get("message", {}).get("content", "")
            return content.strip()

    async def _ensure_ollama_model(self, base_url: str, model_name: str) -> None:
        """Verifica si el modelo existe en Ollama; si no, lo descarga automáticamente."""
        tags_url = f"{base_url}/api/tags"
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(tags_url)
            if res.status_code == 200:
                models = [m.get("name", "").split(":")[0] for m in res.json().get("models", [])]
                full_names = [m.get("name", "") for m in res.json().get("models", [])]
                target_base = model_name.split(":")[0]
                if model_name in full_names or target_base in models:
                    return

        console.print(f"[bold yellow]Descargando modelo local '{model_name}' en Ollama... Por favor espera.[/bold yellow]")
        pull_url = f"{base_url}/api/pull"
        async with httpx.AsyncClient(timeout=600.0) as client:
            res = await client.post(pull_url, json={"name": model_name, "stream": False})
            res.raise_for_status()
        console.print(f"[bold green]✓ Modelo '{model_name}' listo en Ollama.[/bold green]")

    def compute_memory_distribution(self) -> Dict[Any, str]:
        """
        Calcula la distribución óptima de memoria entre VRAM (GPU 0) y RAM (CPU).
        Garantiza que el modelo aproveche la VRAM hasta un máximo del 95% y envíe
        cualquier excedente a la memoria RAM del sistema.
        """
        import torch
        import psutil

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        # Llenar la VRAM hasta el 95% de capacidad libre/total disponible
        vram_target_95 = int(min(total_bytes * 0.95, free_bytes * 0.95))
        vram_target_mb = max(vram_target_95 // (1024 * 1024), 512)

        # RAM del sistema (CPU): permitir hasta el 85% de la RAM disponible
        ram_avail_bytes = psutil.virtual_memory().available
        ram_target_mb = max(int(ram_avail_bytes * 0.85) // (1024 * 1024), 2048)

        return {
            0: f"{vram_target_mb}MB",
            "cpu": f"{ram_target_mb}MB",
        }

    # -------------------------------------------------------------------------
    # Implementación con Transformers (PyTorch en proceso)
    # -------------------------------------------------------------------------
    async def _generate_with_transformers(self, messages: List[Dict[str, str]]) -> str:
        """Carga y ejecuta el modelo usando Hugging Face Transformers en un hilo de fondo."""
        return await asyncio.to_thread(self._run_transformers_sync, messages)

    def _run_transformers_sync(self, messages: List[Dict[str, str]]) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config.local_model_id or "Qwen/Qwen2.5-1.5B-Instruct"

        if self._transformers_model is None or self._transformers_tokenizer is None:
            console.print(
                f"[bold cyan]Inicializando modelo de IA local '{model_id}'...[/bold cyan]\n"
                "[dim](Si es la primera vez, se descargarán los pesos automáticamente)[/dim]"
            )

            # Selección de dispositivo (CUDA si está disponible, o CPU)
            pref_device = self.config.local_model_device.lower()
            use_cuda = torch.cuda.is_available() and pref_device in ("auto", "cuda")

            if use_cuda:
                self._device = "cuda"
                dtype = torch.float16
            else:
                self._device = "cpu"
                dtype = torch.float32

            logger.info("Cargando tokenizer para %s...", model_id)
            self._transformers_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            logger.info("Cargando modelo %s (dispositivo: %s, dtype: %s)...", model_id, self._device, dtype)
            has_accelerate = False
            try:
                import accelerate
                has_accelerate = True
            except ImportError:
                pass

            load_kwargs = {
                "dtype": dtype,
                "trust_remote_code": True,
            }

            if use_cuda and has_accelerate:
                try:
                    max_memory = self.compute_memory_distribution()
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = max_memory
                    
                    console.print(
                        f"[dim]Distribución de memoria VRAM/RAM: VRAM(GPU 0) máx = {max_memory[0]} (95%), "
                        f"RAM(CPU) excedente = {max_memory['cpu']}[/dim]"
                    )
                except Exception as ex:
                    logger.warning("No se pudo calcular límite de VRAM al 95%% (%s), usando device_map='auto'", ex)
                    load_kwargs["device_map"] = "auto"
            elif has_accelerate:
                load_kwargs["device_map"] = self._device

            self._transformers_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
            )
            if not has_accelerate and hasattr(self._transformers_model, "to"):
                self._transformers_model = self._transformers_model.to(self._device)

            hf_device_summary = getattr(self._transformers_model, "hf_device_map", None)
            if hf_device_summary:
                devices_used = set(hf_device_summary.values())
                console.print(f"[bold green]✓ Modelo local '{model_id}' cargado con éxito en: {devices_used}.[/bold green]")
            else:
                console.print(f"[bold green]✓ Modelo local '{model_id}' cargado en {self._device.upper()}.[/bold green]")

        # Aplicar plantilla de chat del modelo
        tokenizer = self._transformers_tokenizer
        model = self._transformers_model

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        primary_device = getattr(model, "device", None) or self._device
        inputs = tokenizer(prompt_text, return_tensors="pt").to(primary_device)
        max_tokens = self.config.local_model_max_new_tokens or 1024

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        temp = getattr(self.config, "local_model_temperature", 0.0)
        if temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Extraer solo los tokens generados por el asistente
        new_tokens = output_tokens[0][len(inputs.input_ids[0]):]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response_text.strip()
