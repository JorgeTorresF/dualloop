"""Deterministic engine backed by a Python function supplied by whoever
integrates the library. Useful as a third arm of the cascade (cost ~0,
fixed confidence when it applies) and to express business logic you do not
want to delegate to a probabilistic model.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..types import Question
from .base import BaseEngine

RuleFn = Callable[[str, Question], Optional[tuple[Any, float]]]


class RuleEngine(BaseEngine):
    """`rule_fn(task_type, question)` must return `(value, confidence)` if
    the rule covers the case, or `None` if it does not apply (the Arbiter
    treats that as a failure of this engine and keeps escalating)."""

    relative_cost = 0.01

    def __init__(self, rule_fn: RuleFn, *, name: str = "rules") -> None:
        self.name = name
        self._rule_fn = rule_fn

    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        result = self._rule_fn(task_type, question)
        if result is None:
            raise ValueError("The rule does not cover this case (no match)")
        value, confidence = result
        return value, confidence, {"source": "rule"}
