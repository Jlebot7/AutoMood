"""
MoodleAssist-AI - Configuración del Sistema
Carga y valida variables de entorno, configuraciones de Groq, navegador y selectores CSS de Moodle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Buscar .env en el directorio actual o en la carpeta del paquete
_current_dir = Path(__file__).resolve().parent
_env_candidates = [
    _current_dir / ".env",
    _current_dir.parent / ".env",
    Path.cwd() / ".env",
]
for env_path in _env_candidates:
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
        break


class MoodleSelectors:
    """Selectores CSS estándar y compatibles con temas Boost y Classic de Moodle."""

    # Contenedores de preguntas
    QUESTION_CONTAINER: str = "div.que"
    QUESTION_INFO: str = ".info"
    QUESTION_NUMBER: str = ".info .no, .info h3.no, .info .qno"
    QUESTION_POINTS: str = ".info .grade, .info .gradeinfo"
    QUESTION_STATUS: str = ".info .state"
    QUESTION_TEXT: str = ".qtext, .formulation .qtext, .content .qtext"
    FORMULATION_BLOCK: str = ".formulation, .content"
    ANSWER_BLOCK: str = ".answer, .ablock"

    # Clases identificadoras de tipos de pregunta en Moodle
    CLASS_MULTICHOICE: str = "multichoice"
    CLASS_TRUEFALSE: str = "truefalse"
    CLASS_SHORTANSWER: str = "shortanswer"
    CLASS_NUMERICAL: str = "numerical"
    CLASS_GAPSELECT: str = "gapselect"
    CLASS_SELECTMULTI: str = "selectmulti"
    CLASS_MATCH: str = "match"
    CLASS_MULTIANSWER: str = "multianswer"  # Cloze
    CLASS_ESSAY: str = "essay"

    # Inputs y opciones
    INPUT_RADIO: str = "input[type='radio']"
    INPUT_CHECKBOX: str = "input[type='checkbox']"
    INPUT_TEXT: str = "input[type='text']:not([type='hidden'])"
    SELECT_DROPDOWN: str = "select"
    TEXTAREA: str = "textarea"
    OPTION_LABEL: str = "label"

    # Botones de navegación del cuestionario
    BTN_NEXT_PAGE: str = "input[name='next'], button[name='next'], a.mod_quiz-next-nav, input.mod_quiz-next-nav"
    BTN_FINISH_ATTEMPT: str = "input[name='finishattempt'], button[name='finishattempt'], a.endtestlink, .endtestlink"
    BTN_SUBMIT_ALL: str = "button.btn-finishattempt, input[value*='Enviar todo'], button:has-text('Enviar todo y terminar')"
    QUIZ_NAV_BLOCK: str = "#mod_quiz_navblock, .qnbutton"


class Settings(BaseModel):
    """Configuraciones validadas del sistema."""

    # Configuración de Groq
    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    groq_temperature: float = Field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    )
    groq_timeout: float = Field(
        default_factory=lambda: float(os.getenv("GROQ_TIMEOUT", "45.0"))
    )
    groq_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_RETRIES", "3"))
    )

    # Configuración de Navegador (nodriver)
    headless: bool = Field(
        default_factory=lambda: os.getenv("HEADLESS", "false").strip().lower() in ("true", "1", "yes")
    )
    chrome_user_data_dir: str = Field(
        default_factory=lambda: os.getenv("CHROME_USER_DATA_DIR", "./chrome_profile")
    )
    browser_executable_path: Optional[str] = Field(
        default_factory=lambda: os.getenv("BROWSER_EXECUTABLE_PATH", None) or None
    )

    # Delays para simulación humanizada
    human_delay_min: float = Field(
        default_factory=lambda: float(os.getenv("HUMAN_DELAY_MIN", "0.4"))
    )
    human_delay_max: float = Field(
        default_factory=lambda: float(os.getenv("HUMAN_DELAY_MAX", "1.0"))
    )
    option_delay_min: float = Field(
        default_factory=lambda: float(os.getenv("OPTION_DELAY_MIN", "0.8"))
    )
    option_delay_max: float = Field(
        default_factory=lambda: float(os.getenv("OPTION_DELAY_MAX", "2.0"))
    )
    typing_speed_min: float = Field(
        default_factory=lambda: float(os.getenv("TYPING_SPEED_MIN", "0.04"))
    )
    typing_speed_max: float = Field(
        default_factory=lambda: float(os.getenv("TYPING_SPEED_MAX", "0.10"))
    )
    human_typing: bool = Field(
        default_factory=lambda: os.getenv("HUMAN_TYPING", "true").strip().lower() in ("true", "1", "yes")
    )

    # URL inicial opcional
    moodle_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("MOODLE_URL", None) or None
    )

    def is_groq_configured(self) -> bool:
        """Verifica si la clave API de Groq ha sido configurada adecuadamente."""
        return bool(self.groq_api_key and not self.groq_api_key.startswith("your_") and not self.groq_api_key.startswith("tu_"))

    def resolve_user_data_dir(self) -> Path:
        """Retorna la ruta absoluta del directorio de perfil de Chrome."""
        p = Path(self.chrome_user_data_dir)
        if not p.is_absolute():
            p = (_current_dir.parent / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


# Instancia singleton de configuración
settings = Settings()
selectors = MoodleSelectors()

