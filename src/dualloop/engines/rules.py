"""Motor determinista basado en una función Python registrada por quien
integra la librería. Útil como tercer brazo de la cascada (coste ~0,
confianza fija cuando aplica) y para representar lógica de negocio que no
se quiere delegar a un modelo probabilístico.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..types import Question
from .base import BaseEngine

RuleFn = Callable[[str, Question], Optional[tuple[Any, float]]]


class RuleEngine(BaseEngine):
    """`rule_fn(task_type, question)` debe devolver `(valor, confianza)` si
    la regla cubre el caso, o `None` si no aplica (el Arbiter lo trata como
    fallo de este motor y sigue escalando a los demás)."""

    relative_cost = 0.01

    def __init__(self, rule_fn: RuleFn, *, name: str = "rules") -> None:
        self.name = name
        self._rule_fn = rule_fn

    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        result = self._rule_fn(task_type, question)
        if result is None:
            raise ValueError("La regla no cubre este caso (sin match)")
        value, confidence = result
        return value, confidence, {"source": "rule"}
