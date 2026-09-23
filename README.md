# 🎓 MoodleAssist-AI

Herramienta modular de escritorio y CLI para la **resolución inteligente y automatizada de cuestionarios, talleres y actividades en plataformas Moodle**. 

Combina la potencia de inferencia del modelo **`llama-3.3-70b-versatile`** a través de **AsyncGroq** con la velocidad y evasión de detección de **nodriver** (automatización asíncrona directa mediante Chrome DevTools Protocol - CDP).

---

## 🌟 Características Principales

- **⚡ Comunicación Directa CDP con `nodriver`**: Control total del navegador sin WebDriver tradicional ni binarios pesados de Selenium, indetectable frente a mecanismos anti-bot estándar.
- **🧠 Motor de Inferencia Groq (`llama-3.3-70b-versatile`)**:
  - Salida forzada en formato estructurado JSON (`response_format={"type": "json_object"}`).
  - Manejo de reintentos exponenciales automáticos (*exponential backoff*) ante límites de tasa (*Rate Limits*).
  - Prompts especializados con razonamiento académico y contextualización de fórmulas LaTeX y descripciones de imágenes.
- **🔍 Extracción Inteligente del DOM de Moodle**:
  - Inyección JavaScript en cliente vía CDP de alta velocidad.
  - Soporte completo para temas Moodle Boost, Classic y personalizados.
  - Saneamiento de textos ruidosos ("Marcar con bandera", "Puntúa como...").
- **📋 Cobertura Completa de Tipos de Pregunta**:
  - **Selección Múltiple y Simple** (`multichoice` con radio buttons o checkboxes).
  - **Verdadero / Falso** (`truefalse`).
  - **Emparejamiento / Relación de Columnas** (`match` con múltiples dropdowns).
  - **Menús Desplegables en Texto** (`gapselect`, `selectmulti`).
  - **Respuesta Corta y Numérica** (`shortanswer`, `numerical`).
  - **Ensayo / Pregunta Abierta** (`essay`).
- **🖥️ Consola Interactiva Moderna con `Rich`**:
  - Tablas estilizadas con justificación del razonamiento de la IA y porcentaje de confianza.
  - Modos de ejecución: **Asistido (Paso a paso con confirmación)**, **Autónomo (Full Auto)** y **Dry-Run (Solo consulta sin modificar el formulario)**.
  - Perfil de usuario persistente (`chrome_profile`) para no tener que iniciar sesión ni validar 2FA repetidamente.

---

## 📁 Arquitectura del Proyecto

```text
moodle_assistant/
├── config.py           # Variables de entorno, timeouts, configuraciones de Groq y selectores CSS
├── models.py           # Modelos de datos Pydantic v2 (QuestionData, LLMResponse, LogEntry, etc.)
├── dom_parser.py       # Extracción de elementos Moodle, clasificación de preguntas y filtro multimedia
├── llm_engine.py       # Cliente AsyncGroq con prompts optimizados y validación de JSON
├── automator.py        # Controlador de nodriver (simulación de clicks, llenado de texto, selecciones)
├── main.py             # CLI interactivo con rich (menu principal, pausas, reportes)
├── requirements.txt    # Dependencias fijadas del proyecto
└── .env.example        # Plantilla de variables de entorno
run.py                  # Script lanzador conveniente desde la raíz
```

---

## 🚀 Requisitos e Instalación

### 1. Requisitos Previos
- **Python 3.11+** instalado en el sistema.
- **Google Chrome** instalado.
- Una **API Key de Groq** gratuita (obtén una en [Groq Console](https://console.groq.com/keys)).

### 2. Instalación de Dependencias

```bash
pip install -r moodle_assistant/requirements.txt
```

O instalando individualmente:
```bash
pip install nodriver groq pydantic python-dotenv rich
```

### 3. Configuración del Archivo `.env`

Copia el archivo de ejemplo o edita el archivo `.env` en la raíz o en `moodle_assistant/`:

```env
# Clave API de Groq
GROQ_API_KEY=gsk_tu_clave_aqui

# Modelo (Recomendado: llama-3.3-70b-versatile)
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.1
GROQ_TIMEOUT=45.0
GROQ_MAX_RETRIES=3

# Configuración del Navegador
HEADLESS=False
CHROME_USER_DATA_DIR=./chrome_profile
HUMAN_DELAY_MIN=0.4
HUMAN_DELAY_MAX=1.0

# Opcional: URL inicial de tu campus Moodle
# MOODLE_URL=https://campus.tuuniversidad.edu/login/index.php
```

---

## 💻 Uso del Sistema

Para iniciar el asistente interactivo ejecuta:

```bash
python run.py
```

O directamente el módulo:

```bash
python -m moodle_assistant.main
```

### Flujo de Trabajo Recomendado

1. **[Opción 1] Iniciar Navegador y Login en Moodle**:
   - Se abrirá una ventana de Chrome con un perfil persistente.
   - Inicia sesión normalmente en tu campus (puedes completar autenticaciones de 2 pasos o SSO institucional).
   - Navega hasta el examen o taller y pulsa *Comenzar intento*.
2. **[Opción 2] Resolver Página Actual (Modo Asistido)**:
   - El sistema analiza las preguntas visibles.
   - Groq evalúa el enunciado, contexto y opciones en milisegundos.
   - Se presenta una tabla en consola con el número de pregunta, respuesta propuesta, justificación y confianza.
   - Se solicita confirmación (`[S/n]`) antes de realizar cualquier clic o escritura en el navegador.
3. **[Opción 3] Resolver Cuestionario Completo (Modo Autónomo)**:
   - Resuelve secuencialmente página por página y presiona el botón *Siguiente página* hasta finalizar.
4. **[Opción 4] Modo Dry-Run**:
   - Extrae y consulta al LLM mostrando las soluciones en la terminal sin tocar los elementos web. Ideal para revisión previa.

---

## 🧪 Pruebas Unitarias

El proyecto incluye tests unitarios para verificar cada capa:

```bash
python tests/test_models.py
python tests/test_dom_parser.py
python tests/test_llm_engine.py
```

---

## 🛡️ Seguridad y Consideraciones Éticas

- Las contraseñas y cookies se almacenan exclusivamente de manera local en el directorio del perfil de Chrome (`chrome_profile`).
- Las llamadas a la API de Groq envían únicamente el texto de los enunciados y opciones de las preguntas para su resolución.
- Esta herramienta está pensada para fines pedagógicos, investigación en automatización web asistida por IA y asistencia accesible.

