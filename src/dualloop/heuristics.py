"""Inferencia opcional de outcomes a partir de señales indirectas.

`report_outcome()` explícito es la única vía verdaderamente generalizable
de cerrar el bucle (nadie puede inventar un outcome que no se observó). Los
heurísticos de este módulo son un mecanismo COMPLEMENTARIO, deshabilitado
por defecto, para quien quiera exprimir señal adicional cuando reportar el
outcome a mano no es práctico. Son inherentemente frágiles y específicos de
dominio — literalmente el mismo tipo de "heurística indirecta" que la
investigación descartó como generalizable para todo el mundo, así que se
documentan aquí como ejemplo de referencia, no como recomendación por
defecto.
"""

from __future__ import annotations

from typing import Protocol

from .types import Decision, Outcome


class OutcomeHeuristic(Protocol):
    name: str

    def try_infer(self, decision: Decision, context: dict) -> Outcome | None: ...


class HumanOverrideHeuristic:
    """Ejemplo de referencia: si un humano corrige la decisión dentro de
    `window_seconds`, se infiere que la decisión original fue incorrecta.

    ADVERTENCIA: un override puede deberse a que cambió el contexto, no a
    que la decisión estuviera mal. Úsalo solo si entiendes el patrón de
    correcciones de tu propio dominio, y revisa periódicamente si está
    sesgando la calibración con outcomes mal inferidos.
    """

    name = "human_override"

    def __init__(self, window_seconds: float = 300.0) -> None:
        self.window_seconds = window_seconds

    def try_infer(self, decision: Decision, context: dict) -> Outcome | None:
        override_at = context.get("override_at")
        if override_at is None:
            return None
        if override_at - decision.created_at > self.window_seconds:
            return None
        return Outcome(
            decision_id=decision.id,
            correct=False,
            note="Inferido: un humano corrigio la decision dentro de la ventana de override.",
            source=self.name,
        )
