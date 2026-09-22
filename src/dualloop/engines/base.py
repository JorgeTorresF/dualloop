"""The base contract every decision engine must satisfy.

An "engine" can be a reasoning LLM, a typed classifier (Jev), a set of
deterministic rules, or anything else that knows how to answer a
`Question`. They all share one interface so the Arbiter can treat them
interchangeably.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..types import EngineOutput, Question


class HttpClientOwner:
    """Manages the lifecycle of an engine's HTTP client.

    An engine can be handed an injected `httpx.Client` (to reuse
    connections, configure retries, or for testing) or build its own. Only
    **the one the engine created** is ever closed: an injected client
    belongs to whoever injected it, and closing it here would break anyone
    else sharing it.

    Engines using this mixin work as context managers::

        with JevEngine(base_url=..., model=...) as jev:
            loop = DualLoop(engines=[jev])
            ...
    """

    def close(self) -> None:
        """Close the HTTP client if it is ours. Idempotent."""
        if getattr(self, "_owns_client", False):
            self._client.close()
            self._owns_client = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False


class BaseEngine(ABC):
    """Base class: measures latency and catches errors uniformly.

    Subclasses only implement `_decide_raw`, which must return
    `(value, raw_confidence, raw_response_dict)` or raise if the engine
    cannot answer. A raised exception becomes an EngineOutput carrying
    `error` rather than propagating, so the Arbiter can keep escalating to
    other engines.
    """

    name: str = "engine"
    #  Indicative relative cost (not real money): use it to declare that
    #  an engine is "cheap and fast" (say 0.1) or "expensive and slow"
    #  (say 5.0). It only affects the bandit's cold-start exploration order.
    relative_cost: float = 1.0

    def decide(self, task_type: str, question: Question) -> EngineOutput:
        t0 = time.monotonic()
        try:
            value, confidence, raw = self._decide_raw(task_type, question)
            latency_ms = (time.monotonic() - t0) * 1000
            return EngineOutput(self.name, value, float(confidence), latency_ms, raw)
        except Exception as exc:  # noqa: BLE001 - one engine must not bring arbitration down
            latency_ms = (time.monotonic() - t0) * 1000
            return EngineOutput(self.name, None, 0.0, latency_ms, {}, error=str(exc))

    @abstractmethod
    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        """The engine's concrete implementation. Must raise if it cannot answer."""
        raise NotImplementedError
