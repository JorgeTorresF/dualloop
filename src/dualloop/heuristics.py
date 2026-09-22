"""Optional inference of outcomes from indirect signals.

An explicit `report_outcome()` is the only genuinely generalisable way to
close the loop: nobody can invent an outcome that was never observed. The
heuristics in this module are a COMPLEMENTARY mechanism, disabled by
default, for anyone who wants to squeeze extra signal out when reporting
the outcome by hand is impractical. They are inherently fragile and
domain-specific — literally the same kind of "indirect heuristic" the
research dismissed as generalisable for everyone — so they are documented
here as a worked example, not as a default recommendation.
"""

from __future__ import annotations

from typing import Protocol

from .types import Decision, Outcome


class OutcomeHeuristic(Protocol):
    name: str

    def try_infer(self, decision: Decision, context: dict) -> Outcome | None: ...


class HumanOverrideHeuristic:
    """Worked example: if a human overrides the decision within
    `window_seconds`, infer that the original decision was wrong.

    WARNING: an override may mean the context changed, not that the
    decision was wrong. Use this only if you understand the correction
    patterns of your own domain, and check periodically whether it is
    skewing calibration with badly inferred outcomes.
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
            note="Inferred: a human overrode the decision within the override window.",
            source=self.name,
        )
