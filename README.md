# 🎓 MoodleAssist-AI (AutoMood)

Herramienta de automatización asistida por IA para la **resolución de cuestionarios, talleres y actividades académicas en plataformas Moodle**. 

Combina la velocidad de inferencia de **Groq** (`llama-3.3-70b-versatile`), la precisión acústica de **Whisper** para listening tests, visión **OCR** local para enunciados gráficos, y la indetectabilidad de **nodriver** (CDP nativo sin WebDriver tradicional) con simulación de comportamiento y tecleo humano.

---

## 🌟 Características Principales

### 🧠 1. Comprensión Lectora Global, Detección de Ejemplos y Tiempos Verbales
- **Lectura Integral Previa**: En preguntas de lectura o textos con múltiples huecos (Cloze / `multianswer` / `gapselect`), el modelo analiza la narración completa de principio a fin antes de resolver espacios individuales.
- **Detección Automática de Ejemplos Guía (In-Context Examples)**: Si el ejercicio incluye un ejemplo modelo (*"Example:"*, *"e.g."*, *"Modelo:"*, *"Sample answer:"* o el hueco `[0]` resuelto), el motor identifica su morfología, tiempo verbal, si incluye o no sujeto y su convención de mayúsculas/minúsculas para replicar exactamente el mismo formato.
- **Análisis de Línea Temporal (Timeline)**:
  - Distinción rigurosa entre oraciones de hechos pasados concluidos (**Simple Past**) y la era actual o cambios continuos (**Present Perfect**).
  - Reconocimiento de marcadores temporales clave (*"Since..."*, *"ago"*, *"in the 1980s"*, *"today"*, *"now"*).
- **Reglas Estrictas de Contracciones**:
  - Aplica contracciones negativas donde corresponde (*"didn't"*, *"weren't"*, *"haven't"*).
  - Evita contracciones afirmativas inválidas tras sustantivos (*"technology has changed"*, no *"technology 's changed"*; *"Times have changed"*, no *"Times 've changed"*).

---

### 🎧 2. Transcripción de Audio con Whisper (Listening Tests)
- **Soporte Nativo para Cuestionarios de Audio**: Detecta automáticamente etiquetas `<audio>`, `<source>` y enlaces a audios (`.mp3`, `.wav`, `.ogg`, `.m4a`).
- **Descarga Autenticada en Sesión**: Descarga archivos protegidos (`/pluginfile.php/...`) ejecutando `fetch(..., { credentials: 'include' })` dentro del contexto del navegador a Base64, preservando las cookies de sesión del estudiante sin requerir credenciales adicionales.
- **Múltiples Proveedores**:
  - **Groq Whisper** (`whisper-large-v3`, `whisper-large-v3-turbo`): Transcripción ultra rápida en milisegundos usando la misma API Key de Groq.
  - **OpenAI Whisper** (`whisper-1`): Alternativa o fallback automático.
  - **Faster-Whisper**: Opción 100% local ejecutada en CPU o GPU.
- **Caché en Memoria**: Evita descargas o transcripciones redundantes mediante claves URL o hash SHA-256.

---

### 🖼️ 3. OCR de Imágenes para Contenido no Accesible
- **Extracción de Texto y Diagramas**: Cuando los exámenes incluyen preguntas embebidas en imágenes en lugar de texto DOM accesible, el sistema extrae el enunciado para suministrarlo al LLM.
- **EasyOCR Local**: Motor primario ligero, multilingüe (español e inglés por defecto), sin costos de API y con ejecución 100% local.
- **OpenAI Vision Fallback (`gpt-4o-mini`)**: Respaldo en la nube para interpretar diagramas o gráficos complejos cuando se proporciona `OPENAI_API_KEY`.
- **Filtro Inteligente de Iconos**: Descarta automáticamente imágenes decorativas, temas e iconos de interfaz (`/pix/`, `/theme/`, `.icon`, gráficos < 35x35px).

---

### 🎭 4. Simulación de Comportamiento Humano (Antidetección)
- **Delays Aleatorios entre Dropdowns y Opciones**: Pausas naturales variables entre 0.8s y 2.0s (`OPTION_DELAY_MIN` / `OPTION_DELAY_MAX`) entre huecos contiguos, simulando el tiempo de lectura del estudiante.
- **Autocompletado de Texto Natural (Pulsación Carácter por Carácter)**:
  - Tipeo a través de eventos reales `InputEvent("input", { inputType: "insertText" })`.
  - Velocidad variable de pulsación entre 40ms y 100ms por carácter (~50-80 PPM).
  - Pausas cognitivas al escribir espacios y signos de puntuación (`,`, `.`, `?`, `!`, `;`).
  - Micro-vacilaciones humanas aleatorias y aceleración progresiva en ensayos largos.
- **Resolución Universal de Selectores**: Maneja selectores complejos de Moodle con dos puntos (`:`) o nombres especiales mediante `findElement()` tolerante a fallos.

---

### 🤖 6. Modelo de IA Local de Respaldo (Fallback Automático)
- **Cero Dependencia de Conectividad**: Si la API de Groq no está disponible, se agotan los reintentos o se superan los límites de tokens (errores 413 o 429), AutoMood activa automáticamente un modelo de IA local sin interrumpir la resolución del cuestionario.
- **Doble Compatibilidad de Motores**:
  - **Ollama (Aceleración nativa en GPU)**: Si Ollama está instalado en el sistema, corre con GPU NVIDIA (ej. GeForce RTX 3050) entregando respuestas en milisegundos con modelos como `qwen2.5:1.5b`.
  - **Hugging Face Transformers (In-Process PyTorch)**: Ejecución directa en proceso con `Qwen/Qwen2.5-1.5B-Instruct` (o `0.5B`), descargando automáticamente los pesos desde Hugging Face la primera vez si es necesario.
- **Sanitización Antidesbordamiento de Tokens**: Depuración de datos en Base64 en el payload de las preguntas, reduciendo el consumo de tokens en un 98% y eliminando la causa de errores de tamaño de petición.

---

### 📋 7. Soporte Exhaustivo de Tipos de Preguntas Moodle
- **Selección Múltiple y Simple** (`multichoice` con radio buttons o checkboxes).
- **Verdadero / Falso** (`truefalse`).
- **Emparejamiento / Relación de Columnas** (`match`).
- **Cloze / Texto con Desplegables Incrustados** (`multianswer`, `gapselect`, `selectmulti`).
- **Respuesta Corta y Numérica** (`shortanswer`, `numerical`).
- **Pregunta Abierta / Ensayo** (`essay`).

---

## 📁 Estructura del Proyecto

```text
AutoMood/
├── moodle_assistant/
│   ├── automator.py        # Controlador CDP con nodriver, typing humano, delays y descarga autenticada
│   ├── config.py           # Configuración Pydantic, variables de entorno y selectores CSS
│   ├── dom_parser.py       # Extractor del DOM, clasificación de tipos, audio e imágenes
│   ├── llm_engine.py       # Motor AsyncGroq con prompts optimizados, lectura global y ejemplos
│   ├── local_llm.py        # Motor de IA local (Ollama con GPU o Transformers en proceso)
│   ├── media_processor.py  # AudioTranscriber (Groq/OpenAI/Faster-Whisper) e ImageOCREngine (EasyOCR)
│   ├── models.py           # Modelos de datos Pydantic v2 (QuestionData, MediaContext, LLMResponse)
│   ├── main.py             # CLI interactivo con Rich (tablas, insignias, confirmación paso a paso)
│   └── requirements.txt    # Dependencias fijadas del proyecto
├── tests/
│   ├── test_dom_parser.py                     # Parsing de preguntas estándar
│   ├── test_human_delays_and_typing.py        # Simulación de delays y escritura carácter por carácter
│   ├── test_local_llm.py                      # Fallback automático a modelo de IA local (413/offline)
│   ├── test_media_extraction.py               # Extracción DOM de audio/imágenes y prompts LLM
│   ├── test_media_processor.py                # Transcripción Whisper y extracción EasyOCR
│   ├── test_models.py                         # Validación de modelos de datos
│   ├── test_payload_token_size.py             # Sanitización y prevención de desbordamiento de tokens
│   └── test_reading_context_comprehension.py  # Comprensión de lectura, contracciones y tiempos (15/15)
├── run.py                  # Script lanzador conveniente desde la raíz
└── .env.example            # Plantilla completa de variables de entorno
```

---

## 🚀 Requisitos e Instalación

### 1. Requisitos Previos
- **Python 3.11+** (compatible con Python 3.11, 3.12, 3.13 y 3.14).
- **Google Chrome** instalado en el sistema.
- **API Key de Groq** (gratuita en [Groq Console](https://console.groq.com/keys)).
- *(Opcional)* **API Key de OpenAI** si deseas usar OpenAI Whisper u OpenAI Vision como respaldo.

### 2. Instalación de Dependencias

```bash
pip install -r moodle_assistant/requirements.txt
```

O instalando los paquetes clave:
```bash
pip install nodriver groq openai faster-whisper easyocr pillow torch pydantic python-dotenv rich
```

### 3. Configuración del Archivo `.env`

Copia `.env.example` como `.env` y ajusta tus preferencias:

```env
# Clave API de Groq (Obligatoria para inferencia de texto y Groq Whisper)
GROQ_API_KEY=gsk_tu_clave_aqui

# Modelo LLM de Inferencia
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.1
GROQ_TIMEOUT=45.0
GROQ_MAX_RETRIES=3

# Modelo Local de Respaldo (Fallback automático ante fallos de Groq, errores 413 o falta de internet)
ENABLE_LOCAL_FALLBACK=true
LOCAL_MODEL_PROVIDER=auto           # auto (GPU Ollama o Transformers), ollama, transformers
LOCAL_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct # HuggingFace ID (Qwen2.5-1.5B o Qwen2.5-0.5B)
OLLAMA_MODEL_ID=qwen2.5:1.5b       # Modelo Ollama para aceleración en GPU RTX 3050
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Transcripción de Audio (Whisper)
ENABLE_AUDIO_TRANSCRIPTION=True
WHISPER_PROVIDER=groq               # groq, openai, local
WHISPER_MODEL=whisper-large-v3     # whisper-large-v3, whisper-large-v3-turbo

# OCR de Imágenes
ENABLE_IMAGE_OCR=True
OCR_PROVIDER=easyocr               # easyocr, openai_vision
OCR_LANGUAGES=es,en                # Idiomas para EasyOCR

# Opcional: Clave API de OpenAI (solo para whisper-1 o gpt-4o-mini vision)
# OPENAI_API_KEY=sk-...

# Configuración del Navegador
HEADLESS=False
CHROME_USER_DATA_DIR=./chrome_profile
HUMAN_DELAY_MIN=0.4
HUMAN_DELAY_MAX=1.0
OPTION_DELAY_MIN=0.8
OPTION_DELAY_MAX=2.0
TYPING_SPEED_MIN=0.04
TYPING_SPEED_MAX=0.10

# Opcional: URL directa de inicio
# MOODLE_URL=https://campus.tuuniversidad.edu/login/index.php
```

---

## 💻 Uso del Asistente

Ejecuta el asistente interactivo:

```bash
python run.py
```

### Flujo de Trabajo Recomendado

1. **[Opción 1] Iniciar Navegador y Login en Moodle**:
   - Se abrirá una ventana de Chrome con un perfil de usuario local y persistente (`chrome_profile`).
   - Inicia sesión normalmente en tu campus virtual (completando 2FA o SSO institucional si aplica).
   - Accede a la actividad o cuestionario que deseas resolver.
2. **[Opción 2] Resolver Página Actual (Modo Asistido con Confirmación)**:
   - Extrae las preguntas visibles y descarga automáticamente audios o imágenes asociados.
   - Groq analiza el texto, evalúa ejemplos e instrucciones de respuesta, y genera la solución.
   - Se muestra la tabla en consola con las respuestas propuestas, justificaciones, nivel de confianza e insignias (`🎧 Audio (OK)`, `🖼️ OCR (OK)`).
   - Puedes confirmar la aplicación automática en el navegador pulsando `y`.
3. **[Opción 3] Resolver Cuestionario Completo (Modo Autónomo)**:
   - Resuelve secuencialmente página por página avanzando de forma continua hasta finalizar.
4. **[Opción 4] Modo Dry-Run (Solo Consulta)**:
   - Consulta al LLM y muestra las soluciones en consola sin tocar ni interactuar con la página web.

---

## 🧪 Ejecución de Pruebas Automatizadas

El proyecto cuenta con una suite completa de pruebas unitarias y de integración:

```bash
# Modelos de datos y validaciones
python tests/test_models.py

# Extracción DOM de preguntas y filtrado
python tests/test_dom_parser.py

# Delays humanizados y autocompletado carácter por carácter
python tests/test_human_delays_and_typing.py

# Comprensión lectora, detección de tiempos verbales y contracciones (100% de aciertos)
python tests/test_reading_context_comprehension.py

# Transcripción de audio con Whisper y OCR con EasyOCR
python tests/test_media_processor.py

# Extracción de recursos multimedia en DOM y prompts enriquecidos
python tests/test_media_extraction.py
```

---

## 🛡️ Seguridad y Buenas Prácticas

- **Privacidad Local**: Las cookies, contraseñas de sesión y perfiles se almacenan de manera estrictamente local en la carpeta `chrome_profile/`.
- **Datos Seguros**: El archivo `.env` está protegido en `.gitignore` para prevenir cualquier exposición accidental de credenciales.
- **Uso Responsable**: Desarrollado con propósitos de investigación en automatización web, accesibilidad y asistencia para el aprendizaje.
