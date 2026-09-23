"""
MoodleAssist-AI - Modelos de Datos Pydantic
Definición de esquemas de datos para preguntas de Moodle, respuestas de la IA y registros de ejecución.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Tipos de preguntas soportadas en Moodle."""
    MULTICHOICE = "multichoice"      # Selección simple (radio) o múltiple (checkbox)
    TRUEFALSE = "truefalse"          # Verdadero / Falso
    SHORTANSWER = "shortanswer"      # Respuesta corta en texto
    NUMERICAL = "numerical"          # Respuesta numérica
    GAPSELECT = "gapselect"          # Menús desplegables integrados en texto
    SELECTMULTI = "selectmulti"      # Selección desplegable
    MATCH = "match"                  # Emparejamiento / Relación de columnas
    MULTIANSWER = "multianswer"      # Cloze (preguntas anidadas con múltiples tipos)
    ESSAY = "essay"                  # Ensayo / Desarrollo
    UNKNOWN = "unknown"              # Tipo no reconocido


class OptionChoice(BaseModel):
    """Opción individual de respuesta en preguntas de selección o desplegables."""
    id: str = Field(description="ID DOM o identificador sintético de la opción")
    text: str = Field(description="Texto visible de la opción")
    value: str = Field(default="", description="Atributo 'value' del input u option")
    selector: str = Field(description="Selector CSS exacto para interactuar")
    is_selected: bool = Field(default=False, description="Indica si la opción ya está seleccionada")


class SubQuestion(BaseModel):
    """Sub-pregunta dentro de una pregunta de emparejamiento (match) o cloze."""
    sub_id: str = Field(description="ID o índice de la sub-pregunta")
    prompt: str = Field(description="Texto o enunciado de la sub-pregunta")
    selector: str = Field(description="Selector CSS del elemento select o input asociado")
    options: List[OptionChoice] = Field(default_factory=list, description="Opciones disponibles en el desplegable")
    selected_value: Optional[str] = Field(default=None, description="Valor actualmente seleccionado si existe")


class MediaContext(BaseModel):
    """Información enriquecida sobre elementos multimedia, fórmulas y tablas."""
    has_images: bool = Field(default=False)
    image_descriptions: List[str] = Field(default_factory=list, description="Textos alternativos o títulos de imágenes")
    equations: List[str] = Field(default_factory=list, description="Fórmulas LaTeX/MathML detectadas")
    tables: List[str] = Field(default_factory=list, description="Representación en texto de tablas de datos")


class QuestionData(BaseModel):
    """Representación completa de una pregunta extraída del DOM de Moodle."""
    question_id: str = Field(description="Identificador único en el DOM (ej: q1, question-100-1)")
    number: str = Field(default="Pregunta", description="Número visible (ej: 'Pregunta 1')")
    qtype: QuestionType = Field(default=QuestionType.UNKNOWN, description="Tipo clasificado de la pregunta")
    prompt: str = Field(description="Texto limpio y formateado del enunciado")
    points_text: Optional[str] = Field(default=None, description="Puntuación o valor de la pregunta")
    media_context: MediaContext = Field(default_factory=MediaContext)
    options: List[OptionChoice] = Field(default_factory=list, description="Opciones de selección")
    sub_questions: List[SubQuestion] = Field(default_factory=list, description="Sub-preguntas para matching o dropdowns")
    is_multiple_response: bool = Field(default=False, description="True si admite más de una opción (checkboxes)")
    is_answered: bool = Field(default=False, description="True si ya cuenta con alguna respuesta marcada")
    raw_html: Optional[str] = Field(default=None, description="Fragmento HTML crudo para debugging si es necesario")


class LLMResponse(BaseModel):
    """Estructura de respuesta generada por la IA para una pregunta individual."""
    question_id: str = Field(description="Identificador de la pregunta respondida")
    explanation: str = Field(description="Breve justificación o razonamiento de la respuesta elegida")
    selected_options: List[str] = Field(
        default_factory=list,
        description="Lista de IDs o textos exactos de las opciones seleccionadas (para multichoice/truefalse)"
    )
    text_answer: Optional[str] = Field(
        default=None,
        description="Texto exacto a escribir para preguntas de respuesta corta, numérica o ensayo"
    )
    matching_answers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Diccionario con pares {sub_id: opcion_o_valor_elegido} para emparejamiento o dropdowns"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza en la respuesta de 0.0 a 1.0"
    )


class BatchLLMResponse(BaseModel):
    """Envoltorio para respuestas de múltiples preguntas procesadas en una sola llamada."""
    answers: List[LLMResponse] = Field(default_factory=list)


class ActionExecutionResult(BaseModel):
    """Resultado de la ejecución de una acción de automatización en el DOM."""
    question_id: str
    success: bool
    action_taken: str
    error_message: Optional[str] = None


class LogEntry(BaseModel):
    """Entrada de registro detallada para seguimiento y reporte."""
    timestamp: datetime = Field(default_factory=datetime.now)
    question_id: str
    question_number: str
    question_type: QuestionType
    chosen_action: str
    confidence: float
    status: str  # SUCCESS, WARNING, ERROR, DRY_RUN
    details: Optional[str] = None


class QuizSessionReport(BaseModel):
    """Reporte consolidado de una sesión de resolución de cuestionario."""
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    total_questions: int = 0
    solved_questions: int = 0
    errors_count: int = 0
    entries: List[LogEntry] = Field(default_factory=list)

