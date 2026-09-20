"""Contrato base que cualquier motor de decisión debe cumplir.

Un "motor" puede ser un LLM de razonamiento, un clasificador tipado (Jev),
un conjunto de reglas deterministas, o cualquier otra cosa que sepa
responder una `Question`. Todos comparten la misma interfaz para que el
Arbiter pueda tratarlos de forma intercambiable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..types import EngineOutput, Question


class BaseEngine(ABC):
    """Clase base: mide latencia y atrapa errores de forma uniforme.

    Subclases solo implementan `_decide_raw`, que debe devolver
    `(value, raw_confidence, raw_response_dict)` o lanzar una excepción si
    el motor no puede responder (se traduce en un EngineOutput con `error`
    en vez de propagar la excepción, para que el Arbiter pueda seguir
    escalando a otros motores).
    """

    name: str = "engine"
    #  Coste relativo indicativo (no es dinero real): úsalo para declarar
    #  que un motor es "barato y rápido" (p.ej. 0.1) o "caro y lento"
    #  (p.ej. 5.0). Solo afecta el orden de exploración en frío del bandit.
    relative_cost: float = 1.0

    def decide(self, task_type: str, question: Question) -> EngineOutput:
        t0 = time.monotonic()
        try:
            value, confidence, raw = self._decide_raw(task_type, question)
            latency_ms = (time.monotonic() - t0) * 1000
            return EngineOutput(self.name, value, float(confidence), latency_ms, raw)
        except Exception as exc:  # noqa: BLE001 - un motor no debe tumbar el arbitraje
            latency_ms = (time.monotonic() - t0) * 1000
            return EngineOutput(self.name, None, 0.0, latency_ms, {}, error=str(exc))

    @abstractmethod
    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        """Implementación concreta del motor. Debe lanzar excepción si no puede responder."""
        raise NotImplementedError
