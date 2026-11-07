"""
MoodleAssist-AI - Parser del DOM de Moodle
Extrae, clasifica y limpia preguntas de actividades y cuestionarios en Moodle.
Soporta ejecución directa vía CDP (JavaScript inyectado) y parsing estático de HTML.
"""

from __future__ import annotations

import re
import html
import json
from typing import List, Dict, Any, Optional

from .config import selectors
from .models import (
    QuestionData,
    QuestionType,
    OptionChoice,
    SubQuestion,
    MediaContext,
)


# Script JavaScript optimizado para ser evaluado dentro del navegador vía CDP
JS_EXTRACTOR_SCRIPT = r"""
(() => {
    const questions = [];
    const questionNodes = document.querySelectorAll("div.que");

    questionNodes.forEach((node, index) => {
        const classList = Array.from(node.classList);
        const qId = node.id || `question-idx-${index}`;

        // Clasificar tipo de pregunta
        let qtype = "unknown";
        if (classList.includes("multichoice")) qtype = "multichoice";
        else if (classList.includes("truefalse")) qtype = "truefalse";
        else if (classList.includes("shortanswer")) qtype = "shortanswer";
        else if (classList.includes("numerical")) qtype = "numerical";
        else if (classList.includes("gapselect")) qtype = "gapselect";
        else if (classList.includes("selectmulti")) qtype = "selectmulti";
        else if (classList.includes("match")) qtype = "match";
        else if (classList.includes("multianswer")) qtype = "multianswer";
        else if (classList.includes("essay")) qtype = "essay";
        else {
            if (node.querySelector("input[type='radio']")) qtype = "multichoice";
            else if (node.querySelector("input[type='checkbox']")) qtype = "multichoice";
            else if (node.querySelector("select")) qtype = "multianswer";
            else if (node.querySelector("input[type='text']")) qtype = "shortanswer";
            else if (node.querySelector("textarea")) qtype = "essay";
        }

        // Número visible de pregunta
        const noNode = node.querySelector(".info .no, .info h3.no, .info .qno");
        const numberText = noNode ? noNode.innerText.trim() : `Pregunta ${index + 1}`;

        // Puntos o estado
        const gradeNode = node.querySelector(".info .grade, .info .gradeinfo");
        const pointsText = gradeNode ? gradeNode.innerText.trim() : null;

        // Extraer texto del enunciado y contexto multimedia
        const formulation = node.querySelector(".formulation, .content") || node;
        let promptText = "";
        const imageDescriptions = [];
        const imageUrls = [];
        const audioUrls = [];
        const equations = [];
        const tables = [];

        // Detectar elementos de audio y enlaces a archivos de audio
        node.querySelectorAll("audio, audio source").forEach(el => {
            const src = el.src || el.getAttribute("src");
            if (src) {
                try {
                    const abs = new URL(src, window.location.href).href;
                    if (!audioUrls.includes(abs)) audioUrls.push(abs);
                } catch(e) {
                    if (!audioUrls.includes(src)) audioUrls.push(src);
                }
            }
        });
        node.querySelectorAll("a[href]").forEach(a => {
            const href = a.getAttribute("href") || "";
            if (href.match(/\.(mp3|wav|ogg|m4a)(\?.*)?$/i) || (href.includes("audio") && href.includes("pluginfile.php"))) {
                try {
                    const abs = new URL(href, window.location.href).href;
                    if (!audioUrls.includes(abs)) audioUrls.push(abs);
                } catch(e) {
                    if (!audioUrls.includes(href)) audioUrls.push(href);
                }
            }
        });

        if (formulation) {
            const clone = formulation.cloneNode(true);

            // Eliminar ruido no deseado (marcas, botones de ayuda, subquestions, plugins de audio y scripts)
            clone.querySelectorAll(".accesshide, .sr-only, .validationerror, .questionflag, .info, script, style, svg, audio, video, .mediaplugin, .video-js").forEach(el => el.remove());

            // Detectar imágenes y URLs de recursos gráficos (filtrando iconos de UI)
            clone.querySelectorAll("img").forEach(img => {
                const alt = img.getAttribute("alt") || "";
                const title = img.getAttribute("title") || "";
                const desc = alt || title;
                if (desc) imageDescriptions.push(desc);

                const src = img.getAttribute("src") || img.src || "";
                if (src) {
                    const isIcon = src.includes("/pix/") || src.includes("/theme/") || src.includes("icon") || src.includes("emoticon") ||
                                   img.classList.contains("icon") || img.classList.contains("emoticon");
                    const w = parseInt(img.getAttribute("width") || img.width || "100", 10);
                    const h = parseInt(img.getAttribute("height") || img.height || "100", 10);
                    if (!isIcon && !(w < 35 && h < 35)) {
                        try {
                            const abs = new URL(src, window.location.href).href;
                            if (!imageUrls.includes(abs)) imageUrls.push(abs);
                        } catch(e) {
                            if (!imageUrls.includes(src)) imageUrls.push(src);
                        }
                    }
                }
            });

            // Detectar ecuaciones matemáticas (LaTeX, KaTeX, MathML)
            clone.querySelectorAll(".math, .katex, math, .latex").forEach(eq => {
                const eqText = eq.getAttribute("data-latex") || eq.innerText || eq.getAttribute("title") || "";
                if (eqText) equations.push(eqText.trim());
            });

            // Detectar tablas y resumirlas
            clone.querySelectorAll("table").forEach(tbl => {
                const rows = [];
                tbl.querySelectorAll("tr").forEach(tr => {
                    const cells = Array.from(tr.querySelectorAll("th, td")).map(c => c.innerText.trim());
                    if (cells.length) rows.push(cells.join(" | "));
                });
                if (rows.length) tables.push(rows.join("\n"));
            });

            // Si es multianswer o gapselect, etiquetar cada hueco/select con su índice e ID para dar contexto completo
            if (qtype === "multianswer" || qtype === "gapselect" || qtype === "selectmulti") {
                let gapCounter = 1;
                clone.querySelectorAll(".subquestion, select, input[type='text']:not([type='hidden'])").forEach(s => {
                    const sid = s.name || s.id || `gap_${gapCounter}`;
                    const span = document.createElement("span");
                    span.innerText = ` [HUECO ${gapCounter}: ${sid}] `;
                    if (s.parentNode) s.parentNode.replaceChild(span, s);
                    gapCounter++;
                });
            }

            promptText = clone.innerText.trim();
        }

        const options = [];
        const subQuestions = [];
        let isMultipleResponse = false;
        let isAnswered = false;

        // 1. Multichoice / True-False
        if (qtype === "multichoice" || qtype === "truefalse") {
            const radioOrCheckInputs = node.querySelectorAll("input[type='radio'], input[type='checkbox']");
            const checkboxes = node.querySelectorAll("input[type='checkbox']");
            if (checkboxes.length > 1) {
                isMultipleResponse = true;
            }

            radioOrCheckInputs.forEach((input, optIdx) => {
                const inputId = input.id;
                let optText = "";
                
                let labelNode = inputId ? node.querySelector(`label[for="${inputId}"]`) : null;
                if (!labelNode) {
                    labelNode = input.closest("label") || input.parentElement;
                }

                if (labelNode) {
                    const labelClone = labelNode.cloneNode(true);
                    labelClone.querySelectorAll(".accesshide, .sr-only").forEach(e => e.remove());
                    optText = labelClone.innerText.trim();
                }

                if (!optText) {
                    optText = input.value || `Opción ${optIdx + 1}`;
                }

                if (input.checked) {
                    isAnswered = true;
                }

                const selector = inputId ? (inputId.includes(':') ? `[id="${inputId}"]` : `#${inputId}`) : `input[name="${input.name}"][value="${input.value}"]`;

                options.push({
                    id: inputId || `opt-${qId}-${optIdx}`,
                    text: optText,
                    value: input.value || "",
                    selector: selector,
                    is_selected: input.checked
                });
            });
        }

        // 2. Matching (Emparejamiento tradicional con filas/tablas)
        else if (qtype === "match") {
            const rows = node.querySelectorAll(".answer tr, .formulation tr");
            rows.forEach((row, rIdx) => {
                const textCell = row.querySelector("td.text, .control-label, td:first-child");
                const selectElem = row.querySelector("select");

                if (selectElem) {
                    const subPrompt = textCell ? textCell.innerText.trim() : `Elemento ${rIdx + 1}`;
                    const selectId = selectElem.id || `select-${qId}-${rIdx}`;
                    const selectSelector = selectElem.name ? `select[name="${selectElem.name}"]` : (selectElem.id ? `[id="${selectElem.id}"]` : `select:nth-of-type(${rIdx + 1})`);
                    
                    const subOpts = [];
                    let currSelected = null;

                    Array.from(selectElem.options).forEach(opt => {
                        const optVal = opt.value;
                        const optTxt = (opt.innerText || opt.text || "").trim();
                        const isBlank = (optVal === "" || optTxt === "" || optTxt === "\u00a0" || optTxt === "Elegir..." || optTxt === "Choose...");
                        if (!isBlank) {
                            subOpts.push({
                                id: `${selectId}-opt-${optVal}`,
                                text: optTxt,
                                value: optVal,
                                selector: `${selectSelector} option[value="${optVal}"]`,
                                is_selected: opt.selected
                            });
                        }
                        if (opt.selected && !isBlank) {
                            currSelected = optVal;
                            isAnswered = true;
                        }
                    });

                    subQuestions.push({
                        sub_id: selectElem.name || `sub_${rIdx}`,
                        prompt: subPrompt,
                        selector: selectSelector,
                        options: subOpts,
                        selected_value: currSelected
                    });
                }
            });
        }

        // 3. GapSelect / Multianswer / Cloze (Desplegables y campos de texto incrustados en oraciones)
        else if (qtype === "gapselect" || qtype === "selectmulti" || qtype === "multianswer") {
            const isMediaControl = (el) => {
                return Boolean(
                    el.closest(".video-js, .mediaplugin, audio, video") ||
                    (el.id && el.id.startsWith("vjs_")) ||
                    (el.className && typeof el.className === "string" && el.className.includes("vjs-"))
                );
            };

            const formulationNode = node.querySelector(".formulation, .content") || node;
            const allGapNodes = Array.from(
                formulationNode.querySelectorAll("select, input[type='text']:not([type='hidden'])")
            ).filter(el => !isMediaControl(el));

            allGapNodes.forEach((gapElem, gIdx) => {
                const gapNum = gIdx + 1;
                const isSelect = (gapElem.tagName.toLowerCase() === "select");
                const sid = gapElem.name || gapElem.id || `gap_${gapNum}`;
                
                let elemSelector = "";
                if (isSelect) {
                    elemSelector = gapElem.name ? `select[name="${gapElem.name}"]` : (gapElem.id ? `[id="${gapElem.id}"]` : `select:nth-of-type(${gapNum})`);
                } else {
                    elemSelector = gapElem.name ? `input[name="${gapElem.name}"]` : (gapElem.id ? `[id="${gapElem.id}"]` : `input[type="text"]:nth-of-type(${gapNum})`);
                }

                // Contexto de la oración circundante para precisión máxima
                let sentenceContext = "";
                const parentBlock = gapElem.closest("p, li, tr, td, dd, dt, blockquote, .r0, .r1");
                if (parentBlock && !parentBlock.classList.contains("formulation") && !parentBlock.classList.contains("que")) {
                    const bClone = parentBlock.cloneNode(true);
                    bClone.querySelectorAll(".accesshide, .sr-only, .validationerror, script, style, svg, audio, video, .mediaplugin, .video-js").forEach(el => el.remove());
                    
                    let localCount = 1;
                    bClone.querySelectorAll("select, input[type='text']:not([type='hidden'])").forEach(s => {
                        const sSubId = s.name || s.id || `gap_${localCount}`;
                        const isThis = (s.name === gapElem.name || s.id === gapElem.id);
                        const marker = isThis ? ` [HUECO ${gapNum}: ${sid}] ` : ` [otro_hueco: ${sSubId}] `;
                        const span = document.createElement("span");
                        span.innerText = marker;
                        if (s.parentNode) s.parentNode.replaceChild(span, s);
                        localCount++;
                    });
                    sentenceContext = bClone.innerText.trim();
                }

                if (!sentenceContext || sentenceContext.length > 250) {
                    const prevText = (gapElem.previousSibling && gapElem.previousSibling.textContent) ? gapElem.previousSibling.textContent.slice(-60).trim() : "";
                    const nextText = (gapElem.nextSibling && gapElem.nextSibling.textContent) ? gapElem.nextSibling.textContent.slice(0, 60).trim() : "";
                    if (prevText || nextText) {
                        sentenceContext = `... ${prevText} [HUECO ${gapNum}: ${sid}] ${nextText} ...`.trim();
                    } else if (sentenceContext && sentenceContext.length > 250) {
                        sentenceContext = sentenceContext.slice(0, 240) + "...";
                    }
                }

                const subOpts = [];
                let currSelected = null;

                if (isSelect) {
                    Array.from(gapElem.options).forEach(opt => {
                        const optVal = opt.value;
                        const optTxt = (opt.innerText || opt.text || "").trim();
                        const isBlank = (optVal === "" || optTxt === "" || optTxt === "\u00a0" || optTxt === "Elegir..." || optTxt === "Choose...");
                        if (!isBlank) {
                            subOpts.push({
                                id: `${sid}_opt_${optVal}`,
                                text: optTxt,
                                value: optVal,
                                selector: `${elemSelector} option[value="${optVal}"]`,
                                is_selected: opt.selected
                            });
                        }
                        if (opt.selected && !isBlank) {
                            currSelected = optVal;
                            isAnswered = true;
                        }
                    });
                } else {
                    if (gapElem.value && gapElem.value.trim().length > 0) {
                        currSelected = gapElem.value.trim();
                        isAnswered = true;
                    }
                }

                subQuestions.push({
                    sub_id: sid,
                    prompt: sentenceContext ? `[HUECO ${gapNum}] ${sentenceContext}` : `Espacio en blanco [HUECO ${gapNum}]`,
                    selector: elemSelector,
                    options: subOpts,
                    selected_value: currSelected
                });
            });
        }

        // 4. Short Answer & Numerical
        else if (qtype === "shortanswer" || qtype === "numerical") {
            const textInput = node.querySelector("input[type='text']");
            if (textInput) {
                const selector = textInput.name ? `input[name="${textInput.name}"]` : (textInput.id ? `[id="${textInput.id}"]` : `input[type="text"]`);
                if (textInput.value && textInput.value.trim().length > 0) {
                    isAnswered = true;
                }
                options.push({
                    id: textInput.id || `input-${qId}`,
                    text: textInput.value || "",
                    value: textInput.value || "",
                    selector: selector,
                    is_selected: Boolean(textInput.value)
                });
            }
        }

        // 5. Essay
        else if (qtype === "essay") {
            const textarea = node.querySelector("textarea");
            if (textarea) {
                const selector = textarea.id ? `#${textarea.id}` : `textarea[name="${textarea.name}"]`;
                if (textarea.value && textarea.value.trim().length > 0) {
                    isAnswered = true;
                }
                options.push({
                    id: textarea.id || `textarea-${qId}`,
                    text: textarea.value || "",
                    value: textarea.value || "",
                    selector: selector,
                    is_selected: Boolean(textarea.value)
                });
            }
        }

        questions.push({
            question_id: qId,
            number: numberText,
            qtype: qtype,
            prompt: promptText,
            points_text: pointsText,
            media_context: {
                has_images: (imageUrls.length > 0 || imageDescriptions.length > 0),
                image_descriptions: imageDescriptions,
                image_urls: imageUrls,
                ocr_texts: [],
                has_audio: (audioUrls.length > 0),
                audio_urls: audioUrls,
                audio_transcriptions: [],
                equations: equations,
                tables: tables
            },
            options: options,
            sub_questions: subQuestions,
            is_multiple_response: isMultipleResponse,
            is_answered: isAnswered
        });
    });

    return JSON.stringify(questions);
})()
"""


class DOMParser:
    """Manejador y transformador de elementos del DOM de Moodle."""

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Limpia cadenas de texto eliminando espacios múltiples y saltos redundantes."""
        if not raw_text:
            return ""
        text = html.unescape(raw_text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    def clean_moodle_boilerplate(prompt: str) -> str:
        """Remueve frases estándar de Moodle que no aportan al contenido de la pregunta."""
        patterns = [
            r"Marcar con bandera la pregunta",
            r"Flag question",
            r"Puntúa como \d+([.,]\d+)?",
            r"Puntuación de \d+([.,]\d+)?",
            r"Marked out of \d+([.,]\d+)?",
            r"Sin responder aún",
            r"Not yet answered",
            r"Respuesta guardada",
            r"Answer saved",
            r"Puntaje de \d+([.,]\d+)?",
        ]
        cleaned = prompt
        for p in patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
        return DOMParser.clean_text(cleaned)

    @classmethod
    def parse_raw_extracted_list(cls, extracted_items: Any) -> List[QuestionData]:
        """
        Convierte el JSON o lista cruda devuelta por la ejecución de JavaScript en objetos QuestionData.
        """
        if isinstance(extracted_items, str):
            try:
                extracted_items = json.loads(extracted_items)
            except Exception:
                extracted_items = []

        if not isinstance(extracted_items, list):
            return []

        questions: List[QuestionData] = []
        for item in extracted_items:
            prompt = cls.clean_moodle_boilerplate(item.get("prompt", ""))
            
            raw_media = item.get("media_context", {})
            img_urls = raw_media.get("image_urls", [])
            aud_urls = raw_media.get("audio_urls", [])
            media = MediaContext(
                has_images=raw_media.get("has_images", False) or bool(img_urls),
                image_descriptions=raw_media.get("image_descriptions", []),
                image_urls=img_urls,
                ocr_texts=raw_media.get("ocr_texts", []),
                has_audio=raw_media.get("has_audio", False) or bool(aud_urls),
                audio_urls=aud_urls,
                audio_transcriptions=raw_media.get("audio_transcriptions", []),
                equations=raw_media.get("equations", []),
                tables=raw_media.get("tables", [])
            )

            raw_opts = item.get("options", [])
            options = [
                OptionChoice(
                    id=opt["id"],
                    text=cls.clean_text(opt.get("text", "")),
                    value=str(opt.get("value", "")),
                    selector=opt.get("selector", ""),
                    is_selected=opt.get("is_selected", False)
                )
                for opt in raw_opts
            ]

            raw_subs = item.get("sub_questions", [])
            sub_questions = []
            for sub in raw_subs:
                sub_opts = [
                    OptionChoice(
                        id=sopt["id"],
                        text=cls.clean_text(sopt.get("text", "")),
                        value=str(sopt.get("value", "")),
                        selector=sopt.get("selector", ""),
                        is_selected=sopt.get("is_selected", False)
                    )
                    for sopt in sub.get("options", [])
                ]
                sub_questions.append(
                    SubQuestion(
                        sub_id=sub["sub_id"],
                        prompt=cls.clean_text(sub.get("prompt", "")),
                        selector=sub.get("selector", ""),
                        options=sub_opts,
                        selected_value=str(sub.get("selected_value")) if sub.get("selected_value") is not None else None
                    )
                )

            raw_qtype = item.get("qtype", "unknown")
            try:
                qtype_enum = QuestionType(raw_qtype)
            except ValueError:
                qtype_enum = QuestionType.UNKNOWN

            q_data = QuestionData(
                question_id=item.get("question_id", "q-unknown"),
                number=item.get("number", "Pregunta"),
                qtype=qtype_enum,
                prompt=prompt,
                points_text=item.get("points_text"),
                media_context=media,
                options=options,
                sub_questions=sub_questions,
                is_multiple_response=item.get("is_multiple_response", False),
                is_answered=item.get("is_answered", False)
            )
            questions.append(q_data)

        return questions

    @classmethod
    def parse_html_string(cls, html_content: str) -> List[QuestionData]:
        """
        Parser estático de HTML para pruebas offline y testing sin navegador.
        """
        questions: List[QuestionData] = []
        pattern = r'<div[^>]*\bclass="[^"]*\bque\b[^"]*"[^>]*>'
        matches = list(re.finditer(pattern, html_content, flags=re.IGNORECASE))
        if not matches:
            return questions

        blocks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(html_content)
            blocks.append(html_content[start:end])

        q_idx = 0
        for block in blocks:
            q_idx += 1
            id_match = re.search(r'id="([^"]+)"', block, re.IGNORECASE)
            qid = id_match.group(1) if id_match else f"q_{q_idx}"

            class_match = re.search(r'class="([^"]+)"', block, re.IGNORECASE)
            classes = class_match.group(1).split() if class_match else []

            qtype = QuestionType.UNKNOWN
            if "multichoice" in classes:
                qtype = QuestionType.MULTICHOICE
            elif "truefalse" in classes:
                qtype = QuestionType.TRUEFALSE
            elif "shortanswer" in classes:
                qtype = QuestionType.SHORTANSWER
            elif "numerical" in classes:
                qtype = QuestionType.NUMERICAL
            elif "gapselect" in classes:
                qtype = QuestionType.GAPSELECT
            elif "match" in classes:
                qtype = QuestionType.MATCH
            elif "multianswer" in classes:
                qtype = QuestionType.MULTIANSWER
            elif "essay" in classes:
                qtype = QuestionType.ESSAY

            no_match = re.search(r'<(?:h3|span|div)[^>]*class="[^"]*no[^"]*"[^>]*>(.*?)</(?:h3|span|div)>', block, re.DOTALL | re.IGNORECASE)
            q_num = re.sub(r'<[^>]+>', '', no_match.group(1)).strip() if no_match else f"Pregunta {q_idx}"

            # Limpiar reproductores multimedia (Video.js, mediaplugin) del bloque completo
            clean_block = re.sub(r'<div[^>]*\bclass="[^"]*(?:video-js|mediaplugin)[^"]*"[^>]*>.*?</div>\s*</div>', '', block, flags=re.DOTALL | re.IGNORECASE)
            clean_block = re.sub(r'<(?:audio|video)\b[^>]*>.*?</(?:audio|video)>', '', clean_block, flags=re.DOTALL | re.IGNORECASE)

            # Formulation / prompt
            form_match = re.search(r'<div[^>]*class="[^"]*formulation[^"]*"[^>]*>(.*?)</div>\s*</div>', clean_block, re.DOTALL | re.IGNORECASE)
            prompt_html = form_match.group(1) if form_match else clean_block

            # Reemplazar selects e inputs de texto con etiquetas numeradas [HUECO N: sid] en el prompt general
            gap_c = 1
            def _replace_gap(m):
                nonlocal gap_c
                tag = m.group(0)
                if 'id="vjs_' in tag or 'class="vjs-' in tag:
                    return ""
                if any(k in tag.lower() for k in ['type="hidden"', 'type="submit"', 'type="radio"', 'type="checkbox"', 'placeholder="search"']):
                    return ""
                name_m = re.search(r'name="([^"]+)"', tag)
                id_m = re.search(r'id="([^"]+)"', tag)
                sid = name_m.group(1) if name_m else (id_m.group(1) if id_m else f"gap_{gap_c}")
                res = f" [HUECO {gap_c}: {sid}] "
                gap_c += 1
                return res

            if qtype in (QuestionType.MULTIANSWER, QuestionType.GAPSELECT, QuestionType.SELECTMULTI):
                prompt_annotated = re.sub(r'(<select\b[^>]*>.*?</select>|<input\b[^>]*type=["\']text["\'][^>]*>)', _replace_gap, prompt_html, flags=re.DOTALL | re.IGNORECASE)
            else:
                prompt_annotated = re.sub(r'<select\b[^>]*>.*?</select>', _replace_gap, prompt_html, flags=re.DOTALL | re.IGNORECASE)

            clean_prompt = cls.clean_moodle_boilerplate(re.sub(r'<[^>]+>', ' ', prompt_annotated))

            # Audio URLs
            audio_urls = []
            for aud_src in re.findall(r'<(?:audio|source)[^>]+src=["\']([^"\']+)["\']', block, re.IGNORECASE):
                if aud_src not in audio_urls:
                    audio_urls.append(aud_src)
            for link_aud in re.findall(r'<a[^>]+href=["\']([^"\']+\.(?:mp3|wav|ogg|m4a)(?:\?[^"\']*)?)["\']', block, re.IGNORECASE):
                if link_aud not in audio_urls:
                    audio_urls.append(link_aud)

            # Image URLs (filtering UI icons)
            image_urls = []
            for img_src in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', prompt_html, re.IGNORECASE):
                if any(k in img_src.lower() for k in ["/pix/", "/theme/", "icon", "emoticon"]):
                    continue
                if img_src not in image_urls:
                    image_urls.append(img_src)

            has_imgs = bool(image_urls or re.search(r'<img\b', prompt_html, re.IGNORECASE))
            img_alts = re.findall(r'<img[^>]*alt="([^"]+)"', prompt_html, re.IGNORECASE)
            eqs = re.findall(r'<span[^>]*class="[^"]*math[^"]*"[^>]*>(.*?)</span>', prompt_html, re.IGNORECASE)

            options: List[OptionChoice] = []
            is_multiple = False
            if qtype in (QuestionType.MULTICHOICE, QuestionType.TRUEFALSE):
                inputs = re.findall(r'<input\s+([^>]+)>', block, re.IGNORECASE)
                checkbox_count = sum(1 for inp in inputs if 'type="checkbox"' in inp)
                if checkbox_count > 1:
                    is_multiple = True
                    qtype = QuestionType.MULTICHOICE

                for inp in inputs:
                    if 'type="radio"' in inp or 'type="checkbox"' in inp:
                        inp_id_m = re.search(r'id="([^"]+)"', inp)
                        inp_val_m = re.search(r'value="([^"]+)"', inp)
                        checked = 'checked' in inp

                        iid = inp_id_m.group(1) if inp_id_m else f"opt_{len(options)}"
                        val = inp_val_m.group(1) if inp_val_m else ""
                        selector = f"[id='{iid}']" if (inp_id_m and ':' in iid) else (f"#{iid}" if inp_id_m else f"input[value='{val}']")

                        opt_text = ""
                        if inp_id_m:
                            lbl_m = re.search(rf'<label[^>]*for="{iid}"[^>]*>(.*?)</label>', block, re.DOTALL | re.IGNORECASE)
                            if lbl_m:
                                opt_text = re.sub(r'<[^>]+>', ' ', lbl_m.group(1)).strip()

                        options.append(OptionChoice(
                            id=iid,
                            text=cls.clean_text(opt_text or f"Opción {len(options)+1}"),
                            value=val,
                            selector=selector,
                            is_selected=checked
                        ))

            sub_questions: List[SubQuestion] = []
            if qtype in (QuestionType.MULTIANSWER, QuestionType.GAPSELECT, QuestionType.SELECTMULTI, QuestionType.MATCH):
                control_matches = list(re.finditer(r'(<select\b([^>]*)>(.*?)</select>|<input\b([^>]*)>)', clean_block, re.DOTALL | re.IGNORECASE))
                gap_idx = 0
                for cm in control_matches:
                    full_match = cm.group(0)
                    if "<select" in full_match:
                        sel_attrs = cm.group(2)
                        sel_content = cm.group(3)
                        if 'id="vjs_' in sel_attrs or 'class="vjs-' in sel_attrs:
                            continue
                        name_m = re.search(r'name="([^"]+)"', sel_attrs)
                        id_m = re.search(r'id="([^"]+)"', sel_attrs)
                        sid = name_m.group(1) if name_m else (id_m.group(1) if id_m else '')
                        if not sid or ('sub' not in sid and '_answer' not in sid and 'gap' not in sid):
                            continue
                        gap_idx += 1
                        sel_selector = f"select[name='{sid}']" if name_m else f"[id='{sid}']"

                        opt_tags = re.findall(r'<option\s+([^>]+)>(.*?)</option>', sel_content, re.DOTALL | re.IGNORECASE)
                        sub_opts = []
                        curr_sel = None
                        for o_attr, o_txt in opt_tags:
                            v_m = re.search(r'value="([^"]*)"', o_attr)
                            val = v_m.group(1) if v_m else ""
                            txt = re.sub(r'<[^>]+>', '', o_txt).strip()
                            is_sel = 'selected' in o_attr
                            is_blank = (val == "" or txt == "" or txt == "&nbsp;" or txt in ("Elegir...", "Choose..."))
                            if not is_blank:
                                sub_opts.append(OptionChoice(
                                    id=f"{sid}_opt_{val}",
                                    text=txt,
                                    value=val,
                                    selector=f"{sel_selector} option[value='{val}']",
                                    is_selected=is_sel
                                ))
                            if is_sel and not is_blank:
                                curr_sel = val

                        sub_questions.append(SubQuestion(
                            sub_id=sid,
                            prompt=f"Espacio desplegable [HUECO {gap_idx}]",
                            selector=sel_selector,
                            options=sub_opts,
                            selected_value=curr_sel
                        ))
                    elif "<input" in full_match:
                        inp_attrs = cm.group(4)
                        if any(k in inp_attrs.lower() for k in ['type="hidden"', 'type="submit"', 'type="radio"', 'type="checkbox"']):
                            continue
                        if 'id="vjs_' in inp_attrs or 'class="vjs-' in inp_attrs:
                            continue
                        name_m = re.search(r'name="([^"]+)"', inp_attrs)
                        id_m = re.search(r'id="([^"]+)"', inp_attrs)
                        sid = name_m.group(1) if name_m else (id_m.group(1) if id_m else '')
                        if not sid or ('sub' not in sid and '_answer' not in sid and 'gap' not in sid):
                            continue
                        gap_idx += 1
                        inp_selector = f"input[name='{sid}']" if name_m else f"[id='{sid}']"
                        val_m = re.search(r'value="([^"]*)"', inp_attrs)
                        curr_val = val_m.group(1) if val_m else None

                        sub_questions.append(SubQuestion(
                            sub_id=sid,
                            prompt=f"Campo de texto libre [HUECO {gap_idx}]",
                            selector=inp_selector,
                            options=[],
                            selected_value=curr_val
                        ))

            questions.append(QuestionData(
                question_id=qid,
                number=q_num,
                qtype=qtype,
                prompt=clean_prompt or "Actividad / Cuestionario Moodle",
                media_context=MediaContext(
                    has_images=has_imgs,
                    image_descriptions=img_alts,
                    image_urls=image_urls,
                    has_audio=bool(audio_urls),
                    audio_urls=audio_urls,
                    equations=[re.sub(r'<[^>]+>', '', eq).strip() for eq in eqs]
                ),
                options=options,
                sub_questions=sub_questions,
                is_multiple_response=is_multiple,
                is_answered=any(opt.is_selected for opt in options) or any(sq.selected_value is not None for sq in sub_questions)
            ))

        return questions
