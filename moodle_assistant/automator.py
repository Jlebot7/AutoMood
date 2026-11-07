"""
MoodleAssist-AI - Automatizador Web con nodriver (CDP)
Controlador asíncrono para interactuar con la página de Moodle: clics, tipeo,
selección de desplegables, scrolling humanizado y navegación entre páginas.
Gestión resiliente de targets CDP ante redirecciones y pestañas activas.
"""

from __future__ import annotations

import re
import json
import asyncio
import random
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import nodriver as uc

from .config import settings, selectors, Settings
from .models import (
    QuestionData,
    QuestionType,
    LLMResponse,
    ActionExecutionResult,
)
from .dom_parser import DOMParser, JS_EXTRACTOR_SCRIPT
from .media_processor import AudioTranscriber, ImageOCREngine

logger = logging.getLogger("moodle_assistant.automator")

FIND_ELEMENT_JS = """
function findElement(sel) {
    if (!sel) return null;
    if (typeof sel !== 'string') return sel;
    sel = sel.trim();
    if (sel.startsWith('#')) {
        const rawId = sel.slice(1);
        const elById = document.getElementById(rawId);
        if (elById) return elById;
        try {
            const escaped = (window.CSS && CSS.escape) ? CSS.escape(rawId) : rawId.replace(/["\\\\]/g, '\\\\$&');
            return document.querySelector('[id="' + escaped + '"]');
        } catch(e) {}
    }
    try {
        const el = document.querySelector(sel);
        if (el) return el;
    } catch(e) {}
    try {
        const elById = document.getElementById(sel);
        if (elById) return elById;
    } catch(e) {}
    try {
        const escaped = (window.CSS && CSS.escape) ? CSS.escape(sel) : sel.replace(/["\\\\]/g, '\\\\$&');
        const elByName = document.querySelector('[name="' + escaped + '"]');
        if (elByName) return elByName;
    } catch(e) {}
    return null;
}
"""


class MoodleAutomator:
    """Controlador CDP basado en nodriver para Moodle."""

    def __init__(self, cfg: Settings = settings):
        self.config = cfg
        self.browser: Optional[uc.Browser] = None
        self._tab: Optional[uc.Tab] = None
        self.audio_transcriber: Optional[AudioTranscriber] = None
        self.image_ocr: Optional[ImageOCREngine] = None
        if getattr(self.config, "enable_audio_transcription", False):
            self.audio_transcriber = AudioTranscriber(self.config)
        if getattr(self.config, "enable_image_ocr", False):
            self.image_ocr = ImageOCREngine(self.config)

    async def start(self) -> None:
        """Inicia la instancia de Chrome con el perfil de usuario especificado."""
        if self.browser is not None:
            return

        profile_dir = self.config.resolve_user_data_dir()
        logger.info(f"Iniciando navegador nodriver con perfil en {profile_dir}...")

        browser_args = [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
        ]

        self.browser = await uc.start(
            headless=self.config.headless,
            user_data_dir=str(profile_dir),
            browser_executable_path=self.config.browser_executable_path,
            browser_args=browser_args,
        )
        self._tab = self.browser.main_tab
        logger.info("Navegador inicializado exitosamente.")

    async def stop(self) -> None:
        """Detiene y cierra el navegador."""
        if self.browser is not None:
            try:
                self.browser.stop()
            except Exception as e:
                logger.debug(f"Error menor al detener browser: {e}")
            finally:
                self.browser = None
                self._tab = None
                logger.info("Navegador detenido.")

    async def get_active_tab(self, force_refresh: bool = False) -> uc.Tab:
        """
        Obtiene la pestaña activa y conectada.
        Sincroniza dinámicamente con los targets de Chrome para evitar errores
        'No target with given id found [code: -32602]'.
        """
        if self.browser is None:
            raise RuntimeError("El navegador no ha sido iniciado. Llama a 'start()' primero.")

        await self.browser.update_targets()
        page_tabs = [t for t in self.browser.targets if t.target.type_ == "page"]
        if not page_tabs:
            raise RuntimeError("No se detectaron pestañas abiertas en el navegador.")

        # 1. Si no forzamos refresco y _tab sigue vivo en browser.targets, reutilizarlo
        if not force_refresh and self._tab is not None:
            if any(t.target.target_id == self._tab.target.target_id for t in page_tabs):
                try:
                    if not self._tab.attached:
                        await self._tab.attach()
                    return self._tab
                except Exception:
                    pass

        # 2. Buscar pestaña que contenga un cuestionario de Moodle (div.que)
        for tab in reversed(page_tabs):
            try:
                if not tab.attached:
                    await tab.attach()
                is_que = await tab.evaluate("document.querySelectorAll('div.que').length > 0")
                if is_que:
                    self._tab = tab
                    return tab
            except Exception:
                continue

        # 3. Buscar pestaña que contenga 'quiz' o 'moodle' en la URL
        for tab in reversed(page_tabs):
            try:
                if not tab.attached:
                    await tab.attach()
                url = await tab.evaluate("window.location.href")
                if "mod/quiz" in url or "moodle" in url:
                    self._tab = tab
                    return tab
            except Exception:
                continue

        # 4. Fallback: usar la última pestaña (la más recientemente abierta o enfocada)
        last_tab = page_tabs[-1]
        try:
            if not last_tab.attached:
                await last_tab.attach()
            self._tab = last_tab
            return last_tab
        except Exception:
            self._tab = page_tabs[0]
            if not self._tab.attached:
                await self._tab.attach()
            return self._tab

    @property
    def tab(self) -> uc.Tab:
        """Retorna la pestaña cacheada o main_tab de respaldo."""
        if self.browser is None:
            raise RuntimeError("El navegador no ha sido iniciado.")
        if self._tab is None:
            self._tab = self.browser.main_tab
        return self._tab

    async def evaluate_in_tab(self, script: str, await_promise: bool = False) -> Any:
        """
        Evalúa un script en la pestaña activa con reintento y re-sincronización
        automática si el target de CDP cambió o se desconectó.
        """
        tab = await self.get_active_tab()
        try:
            return await tab.evaluate(script, await_promise=await_promise)
        except Exception as e:
            err_msg = str(e)
            if "No target with given id" in err_msg or "32602" in err_msg or "closed" in err_msg:
                logger.warning("Target CDP desincronizado. Re-sincronizando pestaña...")
                tab = await self.get_active_tab(force_refresh=True)
                return await tab.evaluate(script, await_promise=await_promise)
            raise

    async def navigate(self, url: str) -> bool:
        """Navega a una URL específica de forma segura validando el protocolo y manejando errores de CDP."""
        if not url:
            return False

        clean_url = url.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://") or clean_url.startswith("file://") or clean_url.startswith("chrome://")):
            if "." in clean_url or clean_url.startswith("localhost"):
                clean_url = "https://" + clean_url
            else:
                logger.warning(f"URL inválida para navegación: '{clean_url}'")
                return False

        logger.info(f"Navegando a: {clean_url}")
        try:
            tab = await self.get_active_tab()
            self._tab = await tab.get(clean_url)
            await self.wait_for_dom_ready()
            return True
        except Exception as e:
            logger.warning(f"Error al navegar a '{clean_url}': {e}")
            return False

    async def wait_for_dom_ready(self, timeout: float = 15.0) -> None:
        """Espera a que el DOM alcance el estado 'complete' o 'interactive'."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                state = await self.evaluate_in_tab("document.readyState")
                if state in ("complete", "interactive"):
                    await asyncio.sleep(0.5)
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)

    async def human_delay(self, min_s: Optional[float] = None, max_s: Optional[float] = None) -> None:
        """Pausa aleatoria entre acciones para simular comportamiento humano."""
        low = min_s if min_s is not None else self.config.human_delay_min
        high = max_s if max_s is not None else self.config.human_delay_max
        if high < low:
            high = low
        delay = random.uniform(low, high)
        await asyncio.sleep(delay)

    async def option_delay(self, min_s: Optional[float] = None, max_s: Optional[float] = None) -> None:
        """Pausa aleatoria natural entre selección de opciones o desplegables contiguos."""
        low = min_s if min_s is not None else self.config.option_delay_min
        high = max_s if max_s is not None else self.config.option_delay_max
        if high < low:
            high = low
        delay = random.uniform(low, high)
        await asyncio.sleep(delay)

    async def is_moodle_quiz_page(self) -> bool:
        """Comprueba si la página actual contiene preguntas de un cuestionario de Moodle."""
        check_script = """
        (() => {
            const hasQue = document.querySelectorAll("div.que").length > 0;
            const hasQuizUrl = window.location.href.includes("mod/quiz/attempt.php") || 
                               window.location.href.includes("mod/quiz/processattempt.php") ||
                               window.location.href.includes("mod/quiz/view.php");
            return hasQue || hasQuizUrl;
        })()
        """
        try:
            result = await self.evaluate_in_tab(check_script)
            return bool(result)
        except Exception as e:
            logger.debug(f"Error comprobando página de cuestionario: {e}")
            return False

    async def extract_questions(self) -> List[QuestionData]:
        """Extrae todas las preguntas de la página actual utilizando el script de evaluación CDP."""
        logger.info("Extrayendo preguntas del DOM de Moodle...")
        try:
            # Obtener y asegurar la pestaña activa donde está el cuestionario
            tab = await self.get_active_tab(force_refresh=True)

            # Esperar activamente si la página sigue cargando elementos div.que
            for _ in range(12):
                try:
                    count = await tab.evaluate("document.querySelectorAll('div.que').length")
                    if count > 0:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.3)

            raw_str = await self.evaluate_in_tab(JS_EXTRACTOR_SCRIPT)
            if not raw_str:
                return []
            questions = DOMParser.parse_raw_extracted_list(raw_str)
            if getattr(self.config, "enable_audio_transcription", False) or getattr(self.config, "enable_image_ocr", False):
                await self.process_questions_media(questions)
            return questions
        except Exception as e:
            logger.error(f"Error al evaluar script extractor en CDP: {e}")
            try:
                # Reintento forzoso
                await self.get_active_tab(force_refresh=True)
                raw_str = await self.evaluate_in_tab(JS_EXTRACTOR_SCRIPT)
                questions = DOMParser.parse_raw_extracted_list(raw_str)
                if getattr(self.config, "enable_audio_transcription", False) or getattr(self.config, "enable_image_ocr", False):
                    await self.process_questions_media(questions)
                return questions
            except Exception as e2:
                logger.error(f"Reintento fallido en extracción CDP: {e2}")
                return []

    async def fetch_media_as_bytes(self, url: str) -> Optional[bytes]:
        """
        Descarga un recurso binario (audio o imagen) desde la página de Moodle
        utilizando fetch() con credentials: 'include' para preservar cookies de sesión,
        retornando los bytes decodificados desde base64.
        """
        if not url:
            return None

        # Si es un Data URI (base64 embebido en DOM), decodificar directamente en memoria
        if url.startswith("data:"):
            try:
                import base64
                _, b64_part = url.split(",", 1)
                return base64.b64decode(b64_part)
            except Exception as e:
                logger.warning(f"Error decodificando data URI en memoria: {e}")
                return None

        fetch_js = f"""
        (async () => {{
            try {{
                const res = await fetch({repr(url)}, {{ credentials: 'include' }});
                if (!res.ok) return null;
                const blob = await res.blob();
                return new Promise((resolve) => {{
                    const reader = new FileReader();
                    reader.onloadend = () => {{
                        const base64 = reader.result ? reader.result.split(',')[1] : null;
                        resolve(base64);
                    }};
                    reader.onerror = () => resolve(null);
                    reader.readAsDataURL(blob);
                }});
            }} catch(e) {{
                return null;
            }}
        }})()
        """
        try:
            b64_str = await self.evaluate_in_tab(fetch_js, await_promise=True)
            if b64_str and isinstance(b64_str, str):
                import base64
                return base64.b64decode(b64_str)
        except Exception as e:
            logger.warning(f"Error descargando recurso multimedia ({url}): {e}")
        return None

    async def process_questions_media(self, questions: List[QuestionData]) -> None:
        """
        Descarga y procesa automáticamente recursos multimedia detectados en las preguntas:
        transcribe audios con Whisper y extrae texto de imágenes con OCR.
        """
        for q in questions:
            # 1. Transcripción de audios (Listening tests)
            if getattr(self.config, "enable_audio_transcription", False) and q.media_context.audio_urls:
                if not self.audio_transcriber:
                    self.audio_transcriber = AudioTranscriber(self.config)

                for aud_url in q.media_context.audio_urls:
                    try:
                        logger.info(f"Descargando audio para {q.number}: {aud_url}")
                        aud_bytes = await self.fetch_media_as_bytes(aud_url)
                        if aud_bytes:
                            clean_fname = aud_url.split("?")[0].split("/")[-1] or "audio.mp3"
                            if not clean_fname.endswith((".mp3", ".wav", ".ogg", ".m4a")):
                                clean_fname = "audio.mp3"
                            transcript = await self.audio_transcriber.transcribe_bytes(
                                aud_bytes, filename=clean_fname, url=aud_url
                            )
                            if transcript and transcript not in q.media_context.audio_transcriptions:
                                q.media_context.audio_transcriptions.append(transcript)
                    except Exception as exc:
                        logger.warning(f"No se pudo procesar audio en {q.number}: {exc}")

            # 2. Extracción OCR de imágenes
            if getattr(self.config, "enable_image_ocr", False) and q.media_context.image_urls:
                if not self.image_ocr:
                    self.image_ocr = ImageOCREngine(self.config)

                for img_url in q.media_context.image_urls:
                    try:
                        logger.info(f"Descargando imagen para OCR en {q.number}: {img_url}")
                        img_bytes = await self.fetch_media_as_bytes(img_url)
                        if img_bytes:
                            clean_fname = img_url.split("?")[0].split("/")[-1] or "image.png"
                            ocr_text = await self.image_ocr.extract_text_from_bytes(
                                img_bytes, filename=clean_fname, url=img_url
                            )
                            if ocr_text and ocr_text not in q.media_context.ocr_texts:
                                q.media_context.ocr_texts.append(ocr_text)
                    except Exception as exc:
                        logger.warning(f"No se pudo procesar OCR en {q.number}: {exc}")

    async def scroll_into_view(self, selector: str) -> bool:
        """Desplaza suavemente la ventana hasta el elemento indicado."""
        script = f"""
        (() => {{
            {FIND_ELEMENT_JS}
            const el = findElement({repr(selector)});
            if (el) {{
                el.scrollIntoView({{ behavior: "smooth", block: "center", inline: "nearest" }});
                return true;
            }}
            return false;
        }})()
        """
        try:
            return bool(await self.evaluate_in_tab(script))
        except Exception:
            return False

    async def click_option(self, selector: str) -> bool:
        """
        Realiza un clic seguro en una opción (radio button o checkbox),
        despachando eventos de clic y cambio para asegurar que Moodle capture el estado.
        """
        await self.scroll_into_view(selector)
        await self.human_delay(0.25, 0.6)

        script = f"""
        (() => {{
            {FIND_ELEMENT_JS}
            const el = findElement({repr(selector)});
            if (!el) return false;

            el.focus();
            if (el.type === "radio" || el.type === "checkbox") {{
                el.checked = true;
            }}
            el.click();

            el.dispatchEvent(new Event("input", {{ bubbles: true }}));
            el.dispatchEvent(new Event("change", {{ bubbles: true }}));
            return true;
        }})()
        """
        try:
            res = await self.evaluate_in_tab(script)
            await self.human_delay(0.15, 0.35)
            return bool(res)
        except Exception as e:
            logger.error(f"Error al hacer clic en selector {selector}: {e}")
            return False

    async def type_into_input(self, selector: str, text: str, human_typing: Optional[bool] = None) -> bool:
        """
        Escribe texto en un campo de texto o textarea simulando tecleo humano
        con pausas variables entre caracteres, o de forma inmediata si human_typing=False.
        """
        await self.scroll_into_view(selector)
        await self.human_delay(0.25, 0.6)

        use_human = human_typing if human_typing is not None else self.config.human_typing
        min_ms = int(self.config.typing_speed_min * 1000)
        max_ms = int(self.config.typing_speed_max * 1000)
        if max_ms < min_ms:
            max_ms = min_ms

        # Para textos muy largos (ej: ensayo > 180 caracteres), acelerar proporcionalmente
        # para que la escritura fluya sin demorarse excesivamente
        if len(text) > 180:
            scale = max(0.25, 180 / len(text))
            min_ms = max(5, int(min_ms * scale))
            max_ms = max(15, int(max_ms * scale))

        script = f"""
        (async () => {{
            {FIND_ELEMENT_JS}
            const el = findElement({repr(selector)});
            if (!el) return false;

            el.focus();
            if (el.isContentEditable) {{
                el.innerText = "";
            }} else {{
                el.value = "";
            }}

            const targetText = {repr(text)};
            const useHuman = {str(use_human).lower()};
            const minDelay = {min_ms};
            const maxDelay = {max_ms};

            if (!useHuman) {{
                if (el.isContentEditable) {{
                    el.innerText = targetText;
                }} else {{
                    el.value = targetText;
                }}
                el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                el.dispatchEvent(new Event("change", {{ bubbles: true }}));
                return true;
            }}

            for (let i = 0; i < targetText.length; i++) {{
                const ch = targetText[i];
                if (el.isContentEditable) {{
                    el.innerText += ch;
                }} else {{
                    el.value += ch;
                }}

                try {{
                    const ev = new InputEvent("input", {{
                        bubbles: true,
                        cancelable: true,
                        inputType: "insertText",
                        data: ch
                    }});
                    el.dispatchEvent(ev);
                }} catch (e) {{
                    el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                }}

                let delay = Math.floor(Math.random() * (maxDelay - minDelay + 1)) + minDelay;
                if (ch === " " || ch === "," || ch === "." || ch === "?" || ch === "!" || ch === ";") {{
                    delay += Math.floor(Math.random() * 110) + 50;
                }} else if (Math.random() < 0.05) {{
                    delay += Math.floor(Math.random() * 140) + 80;
                }}

                await new Promise(r => setTimeout(r, delay));
            }}

            el.dispatchEvent(new Event("change", {{ bubbles: true }}));
            el.dispatchEvent(new Event("blur", {{ bubbles: true }}));
            if (window.jQuery) {{
                try {{ window.jQuery(el).trigger("change"); }} catch(e) {{}}
            }}
            return true;
        }})()
        """
        try:
            res = await self.evaluate_in_tab(script, await_promise=True)
            await self.human_delay(0.2, 0.45)
            return bool(res)
        except Exception as e:
            logger.error(f"Error al escribir en selector {selector}: {e}")
            return False

    async def select_dropdown_option(self, select_selector: str, target_val_or_text: str) -> bool:
        """
        Selecciona una opción en un menú desplegable (<select>) simulando
        la apertura, lectura y elección humana con pausas realistas:
        1. Valor exacto (value)
        2. Texto exacto (text)
        3. Prefijo / Contenido
        """
        await self.scroll_into_view(select_selector)
        # Pausa natural simulando inspeccionar el desplegable antes de abrirlo
        await self.human_delay(0.3, 0.7)

        script = f"""
        (() => {{
            {FIND_ELEMENT_JS}
            const sel = findElement({repr(select_selector)});
            if (!sel || !sel.options) return false;

            sel.focus();

            const target = {repr(target_val_or_text)}.trim().toLowerCase();
            let matchedIdx = -1;

            let targetDigit = null;
            const digitMatch = target.match(/(?:opt|option|opcion)?[_\\s-]*(\\d+)/i);
            if (digitMatch) {{
                targetDigit = digitMatch[1];
            }}

            // 1. Coincidencia exacta por value o por dígito (ej: "opt_2" -> value "2")
            for (let i = 0; i < sel.options.length; i++) {{
                const optVal = (sel.options[i].value || "").trim().toLowerCase();
                if (optVal === target || (targetDigit !== null && optVal === targetDigit)) {{
                    matchedIdx = i;
                    break;
                }}
            }}

            // 2. Coincidencia exacta por texto visible
            if (matchedIdx === -1) {{
                for (let i = 0; i < sel.options.length; i++) {{
                    const optText = (sel.options[i].text || sel.options[i].innerText || "").trim().toLowerCase();
                    if (optText === target) {{
                        matchedIdx = i;
                        break;
                    }}
                }}
            }}

            // 3. Coincidencia por prefijo
            if (matchedIdx === -1) {{
                for (let i = 0; i < sel.options.length; i++) {{
                    const optText = (sel.options[i].text || sel.options[i].innerText || "").trim().toLowerCase();
                    if (optText.length > 2 && (optText.startsWith(target) || target.startsWith(optText))) {{
                        matchedIdx = i;
                        break;
                    }}
                }}
            }}

            // 4. Inclusión si no hubo coincidencia previa
            if (matchedIdx === -1) {{
                for (let i = 0; i < sel.options.length; i++) {{
                    const optText = (sel.options[i].text || sel.options[i].innerText || "").trim().toLowerCase();
                    if (optText.length > 3 && (optText.includes(target) || target.includes(optText))) {{
                        matchedIdx = i;
                        break;
                    }}
                }}
            }}

            if (matchedIdx !== -1) {{
                sel.selectedIndex = matchedIdx;
                sel.value = sel.options[matchedIdx].value;
                for (let j = 0; j < sel.options.length; j++) {{
                    if (j === matchedIdx) {{
                        sel.options[j].selected = true;
                        sel.options[j].setAttribute("selected", "selected");
                    }} else {{
                        sel.options[j].selected = false;
                        sel.options[j].removeAttribute("selected");
                    }}
                }}
                sel.dispatchEvent(new Event("focus", {{ bubbles: true }}));
                sel.dispatchEvent(new Event("input", {{ bubbles: true }}));
                sel.dispatchEvent(new Event("change", {{ bubbles: true }}));
                sel.dispatchEvent(new Event("blur", {{ bubbles: true }}));
                if (window.jQuery) {{
                    try {{ window.jQuery(sel).trigger("change"); }} catch(e) {{}}
                }}
                return true;
            }}
            return false;
        }})()
        """
        try:
            res = await self.evaluate_in_tab(script)
            # Pequeña pausa natural tras confirmar la selección
            await self.human_delay(0.2, 0.45)
            return bool(res)
        except Exception as e:
            logger.error(f"Error al seleccionar en dropdown {select_selector}: {e}")
            return False

    async def trigger_moodle_autosave(self) -> bool:
        """
        Fuerza la ejecución del autosave nativo de Moodle (M.mod_quiz.autosave.save_changes())
        para que los cambios se transmitan inmediatamente al servidor.
        """
        script = """
        (() => {
            let triggered = false;
            if (typeof M !== "undefined" && M.mod_quiz && M.mod_quiz.autosave) {
                try {
                    M.mod_quiz.autosave.save_changes();
                    triggered = true;
                } catch (e) {}
            }
            const form = document.getElementById("responseform");
            if (form) {
                try {
                    form.dispatchEvent(new Event("change", { bubbles: true }));
                } catch (e) {}
            }
            return triggered;
        })()
        """
        try:
            return bool(await self.evaluate_in_tab(script))
        except Exception:
            return False

    async def apply_solution(self, question: QuestionData, response: LLMResponse) -> ActionExecutionResult:
        """
        Aplica la solución generada por el LLM a los elementos correspondientes en el DOM.
        """
        qid = question.question_id
        qtype = question.qtype

        try:
            # 1. PREGUNTAS DE SELECCIÓN (MULTICHOICE / TRUEFALSE)
            if qtype in (QuestionType.MULTICHOICE, QuestionType.TRUEFALSE):
                applied_options = []
                for selected_item in response.selected_options:
                    target_str = selected_item.strip().lower()
                    
                    matched_opt = None
                    for opt in question.options:
                        opt_id_clean = opt.id.strip().lower()
                        opt_text_clean = opt.text.strip().lower()
                        opt_val_clean = opt.value.strip().lower()

                        if (target_str == opt_id_clean or 
                            target_str == opt_val_clean or 
                            target_str in opt_text_clean or 
                            opt_text_clean in target_str):
                            matched_opt = opt
                            break

                    if matched_opt:
                        ok = await self.click_option(matched_opt.selector)
                        if ok:
                            applied_options.append(matched_opt.text)
                            await self.option_delay()
                    else:
                        letter_match = None
                        for opt in question.options:
                            if opt.text.lower().startswith(f"{target_str}.") or opt.text.lower().startswith(f"{target_str})"):
                                letter_match = opt
                                break
                        if letter_match:
                            ok = await self.click_option(letter_match.selector)
                            if ok:
                                applied_options.append(letter_match.text)
                                await self.option_delay()

                if applied_options:
                    return ActionExecutionResult(
                        question_id=qid,
                        success=True,
                        action_taken=f"Seleccionada(s): {', '.join(applied_options)}"
                    )
                else:
                    return ActionExecutionResult(
                        question_id=qid,
                        success=False,
                        action_taken="Ninguna opción coincidió",
                        error_message=f"No se hallaron coincidencias para: {response.selected_options}"
                    )

            # 2. PREGUNTAS DE RESPUESTA CORTA / NUMÉRICA
            elif qtype in (QuestionType.SHORTANSWER, QuestionType.NUMERICAL):
                if not response.text_answer:
                    return ActionExecutionResult(
                        question_id=qid,
                        success=False,
                        action_taken="No se proporcionó texto de respuesta",
                        error_message="El LLM no generó un valor para text_answer."
                    )
                
                selector = question.options[0].selector if question.options else f"[id='{qid}'] input[type='text']"
                ok = await self.type_into_input(selector, response.text_answer)
                return ActionExecutionResult(
                    question_id=qid,
                    success=ok,
                    action_taken=f"Escrito texto: '{response.text_answer}'"
                )

            # 3. PREGUNTAS DE EMPAREJAMIENTO, HUECOS Y CLOZE (MATCH / GAPSELECT / MULTIANSWER)
            elif qtype in (QuestionType.MATCH, QuestionType.GAPSELECT, QuestionType.SELECTMULTI, QuestionType.MULTIANSWER):
                if not response.matching_answers and not response.text_answer:
                    return ActionExecutionResult(
                        question_id=qid,
                        success=False,
                        action_taken="No se proporcionaron respuestas para los desplegables/huecos",
                        error_message="El LLM no devolvió matching_answers ni text_answer."
                    )

                applied_count = 0
                matching_dict = response.matching_answers or {}

                # Si solo hay 1 subpregunta de texto y vino text_answer
                if len(question.sub_questions) == 1 and not question.sub_questions[0].options and response.text_answer:
                    ok = await self.type_into_input(question.sub_questions[0].selector, response.text_answer)
                    return ActionExecutionResult(
                        question_id=qid,
                        success=ok,
                        action_taken=f"Escrito en hueco: '{response.text_answer}'"
                    )

                for sq_idx, sq in enumerate(question.sub_questions):
                    target_val = None
                    # 1. Búsqueda exacta por sub_id
                    if sq.sub_id in matching_dict:
                        target_val = matching_dict[sq.sub_id]
                    else:
                        # 2. Búsqueda por candidatos de índice (ej: "1", "sub1", "HUECO 1", "gap_1")
                        idx_candidates = [
                            str(sq_idx + 1),
                            f"sub{sq_idx + 1}",
                            f"sub_{sq_idx + 1}",
                            f"gap_{sq_idx + 1}",
                            f"gap{sq_idx + 1}",
                            f"HUECO {sq_idx + 1}",
                            f"HUECO_{sq_idx + 1}",
                            f"hueco {sq_idx + 1}",
                            f"hueco_{sq_idx + 1}",
                            f"hueco{sq_idx + 1}",
                            str(sq_idx),
                        ]
                        for cand in idx_candidates:
                            if cand in matching_dict:
                                target_val = matching_dict[cand]
                                break

                    # 3. Búsqueda por subcadena / prompt si aún no se halla
                    if target_val is None:
                        sq_id_clean = sq.sub_id.lower()
                        for key, val in matching_dict.items():
                            k_clean = key.lower()
                            if (k_clean == sq_id_clean or
                                k_clean in sq_id_clean or
                                sq_id_clean in k_clean or
                                (sq.prompt and k_clean in sq.prompt.lower())):
                                target_val = val
                                break

                    if target_val is not None:
                        # Si es un desplegable con opciones (<select>)
                        if sq.options:
                            target_str = str(target_val).strip().lower()
                            chosen_opt = None

                            # 1. Coincidencia exacta por ID, value o text
                            for opt in sq.options:
                                if (opt.id.lower() == target_str or
                                    opt.value.lower() == target_str or
                                    opt.text.lower() == target_str):
                                    chosen_opt = opt
                                    break

                            # 2. Coincidencia si target_str es del tipo "opt_2", "option 2", "2"
                            if not chosen_opt:
                                opt_digits = re.findall(r'\d+', target_str)
                                if opt_digits:
                                    last_d = opt_digits[-1]
                                    for opt in sq.options:
                                        if (opt.value == last_d or
                                            opt.id.lower().endswith(f"_opt_{last_d}") or
                                            opt.id.lower().endswith(f"_{last_d}")):
                                            chosen_opt = opt
                                            break

                            # 3. Coincidencia por texto contenido
                            if not chosen_opt:
                                for opt in sq.options:
                                    opt_txt_clean = opt.text.lower()
                                    if opt_txt_clean and (opt_txt_clean in target_str or target_str in opt_txt_clean):
                                        chosen_opt = opt
                                        break

                            val_to_apply = chosen_opt.value if chosen_opt else str(target_val)
                            ok = await self.select_dropdown_option(sq.selector, val_to_apply)
                            if ok:
                                applied_count += 1
                            elif chosen_opt and chosen_opt.text:
                                # Reintento con texto completo visible si value falló
                                ok2 = await self.select_dropdown_option(sq.selector, chosen_opt.text)
                                if ok2:
                                    applied_count += 1
                        else:
                            # Campo de texto libre dentro de cloze
                            inp_selector = sq.selector or (f"input[name='{sq.sub_id}']" if sq.sub_id else "input[type='text']")
                            ok = await self.type_into_input(inp_selector, str(target_val))
                            if ok:
                                applied_count += 1

                        # Pausa aleatoria natural antes de pasar al siguiente hueco o desplegable
                        if sq_idx < len(question.sub_questions) - 1:
                            await self.option_delay()

                return ActionExecutionResult(
                    question_id=qid,
                    success=applied_count > 0,
                    action_taken=f"Completados {applied_count}/{len(question.sub_questions)} huecos/desplegables"
                )

            # 4. PREGUNTAS DE ENSAYO / DESARROLLO
            elif qtype == QuestionType.ESSAY:
                if not response.text_answer:
                    return ActionExecutionResult(
                        question_id=qid,
                        success=False,
                        action_taken="Sin texto para ensayo",
                        error_message="El LLM no proveyó texto para la respuesta de ensayo."
                    )
                selector = question.options[0].selector if question.options else f"[id='{qid}'] textarea"
                ok = await self.type_into_input(selector, response.text_answer)
                return ActionExecutionResult(
                    question_id=qid,
                    success=ok,
                    action_taken="Redacción de ensayo ingresada"
                )

            else:
                return ActionExecutionResult(
                    question_id=qid,
                    success=False,
                    action_taken="Tipo no soportado",
                    error_message=f"El tipo de pregunta '{qtype.value}' no tiene handler configurado."
                )

        except Exception as e:
            logger.error(f"Error aplicando solución a pregunta {qid}: {e}")
            return ActionExecutionResult(
                question_id=qid,
                success=False,
                action_taken="Excepción durante interacción",
                error_message=str(e)
            )

    async def click_next_page(self) -> bool:
        """Busca y presiona el botón 'Siguiente página' o equivalente de Moodle."""
        logger.info("Buscando botón para avanzar de página...")
        script = f"""
        (() => {{
            {FIND_ELEMENT_JS}
            const candidates = [
                "input[name='next']",
                "button[name='next']",
                "#mod_quiz-next-nav",
                ".mod_quiz-next-nav",
                "input[value*='Siguiente']",
                "input[value*='Next']",
                "input[name='finishattempt']",
                "button[name='finishattempt']",
                "a.endtestlink"
            ];

            for (const sel of candidates) {{
                try {{
                    const btn = findElement(sel);
                    if (btn && btn.offsetParent !== null) {{
                        btn.scrollIntoView({{ behavior: "smooth", block: "center" }});
                        btn.click();
                        return true;
                    }}
                }} catch (e) {{}}
            }}

            const allButtons = document.querySelectorAll("input[type='submit'], button");
            for (const b of allButtons) {{
                const text = (b.value || b.innerText || "").toLowerCase();
                if (text.includes("siguiente") || text.includes("next") || text.includes("terminar intento") || text.includes("finish attempt")) {{
                    b.scrollIntoView({{ behavior: "smooth", block: "center" }});
                    b.click();
                    return true;
                }}
            }}
            return false;
        }})()
        """
        try:
            clicked = await self.evaluate_in_tab(script)
            if clicked:
                logger.info("Botón presionado. Esperando carga de la siguiente página...")
                await self.wait_for_dom_ready()
                return True
            return False
        except Exception as e:
            logger.error(f"Error al intentar avanzar de página: {e}")
            return False

    async def take_screenshot(self, filename: str) -> Optional[str]:
        """Captura de pantalla de la página actual para evidencia o depuración."""
        try:
            path = Path(filename).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            tab = await self.get_active_tab()
            await tab.save_screenshot(str(path))
            return str(path)
        except Exception as e:
            logger.error(f"Error guardando screenshot: {e}")
            return None
