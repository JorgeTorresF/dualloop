"""Tipos de datos centrales de DualLoop.

El esquema de `Question` reutiliza deliberadamente el mismo contrato que usa
simple-jev (https://github.com/featherless-ai/simple-jev) para sus preguntas
choice/score/noul. Compartir un único esquema entre un LLM de razonamiento,
un clasificador tipado y un motor de reglas es lo que permite compararlos
como "peras con peras" en el árbitro — ningún framework de orquestación
mainstream (LangGraph, CrewAI, AutoGen, Semantic Kernel) define hoy un
contrato así entre motores heterogéneos; cada uno resuelve routing solo
entre variantes de LLM.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

QuestionType = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Question:
    """Una pregunta de decisión, en el mismo esquema que simple-jev.

    - choice: elegir entre 2-50 opciones (`criteria` = {opcion: descripcion|None})
    - score:  puntuar sobre una rúbrica de 2-50 niveles (`criteria` = {nivel: descripcion|None})
    - noul:   juicio binario con grado de certeza (no requiere `criteria`)
    """

    type: QuestionType
    instructions: str
    state: str = ""
    criteria: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.type in ("choice", "score"):
            if not self.criteria:
                raise ValueError(f"Question type={self.type!r} requiere 'criteria' no vacío")
            if self.type == "choice" and not (2 <= len(self.criteria) <= 50):
                raise ValueError("choice requiere entre 2 y 50 opciones en 'criteria'")


@dataclass
class EngineOutput:
    """Salida cruda de un motor, antes de calibrar."""

    engine_name: str
    value: Any
    raw_confidence: float
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Vote:
    """Un EngineOutput ya calibrado: lo que el árbitro compara entre motores."""

    engine_name: str
    value: Any
    raw_confidence: float
    calibrated_confidence: float
    latency_ms: float
    raw: dict[str, Any]


@dataclass
class Decision:
    """Registro auditable completo de una decisión de arbitraje."""

    id: str
    task_type: str
    question: Question
    votes: list[Vote]
    chosen_engine: str
    chosen_value: Any
    chosen_confidence: float
    escalation_threshold_used: float
    disagreement: bool
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


@dataclass
class Outcome:
    """Lo que pasó de verdad, reportado después de la decisión.

    `correct=None` representa "no se pudo determinar" (resultado ambiguo o
    parcial) y NO dispara recalibración; solo queda como nota de auditoría.
    """

    decision_id: str
    correct: Optional[bool]
    reward: Optional[float] = None
    note: str = ""
    ground_truth: Any = None
    source: str = "explicit"
    reported_at: float = field(default_factory=time.time)
