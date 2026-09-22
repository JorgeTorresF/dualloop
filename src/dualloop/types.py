"""DualLoop's core data types.

`Question` deliberately reuses the same contract simple-jev
(https://github.com/featherless-ai/simple-jev) uses for its
choice/score/noul questions. Sharing one schema across a reasoning LLM, a
typed classifier and a rule engine is what makes them comparable
like-for-like in the arbiter -- no mainstream orchestration framework
(LangGraph, CrewAI, AutoGen, Semantic Kernel) defines such a contract
across heterogeneous engines today; each solves routing only between LLM
variants.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

QuestionType = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Question:
    """A decision question, in the same schema as simple-jev.

    - choice: pick one of 2-50 options (`criteria` = {option: description|None})
    - score:  rate on a rubric of 2-50 levels (`criteria` = {level: description|None})
    - noul:   binary judgment with a degree of certainty (no `criteria` needed)
    """

    type: QuestionType
    instructions: str
    state: str = ""
    criteria: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.type in ("choice", "score"):
            if not self.criteria:
                raise ValueError(f"Question type={self.type!r} requires a non-empty 'criteria'")
            if self.type == "choice" and not (2 <= len(self.criteria) <= 50):
                raise ValueError("choice requires between 2 and 50 options in 'criteria'")


@dataclass
class EngineOutput:
    """An engine's raw output, before calibration."""

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
    """A calibrated EngineOutput: what the arbiter compares across engines."""

    engine_name: str
    value: Any
    raw_confidence: float
    calibrated_confidence: float
    latency_ms: float
    raw: dict[str, Any]


@dataclass
class Decision:
    """Complete auditable record of one arbitration decision."""

    id: str
    task_type: str
    question: Question
    votes: list[Vote]
    chosen_engine: str
    chosen_value: Any
    chosen_confidence: float
    escalation_threshold_used: float
    disagreement: bool
    abstained: bool = False
    """True when no engine cleared the threshold and the loop declares
    itself incompetent for this case (requires `abstain_below_threshold=True`).

    `chosen_value` still carries the best vote available, so the caller can
    show it to whoever decides; but a decision with `abstained=True` must
    NOT be executed without human review."""
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


@dataclass
class Outcome:
    """What actually happened, reported after the decision.

    `correct=None` means "could not be determined" (an ambiguous or partial
    result) and does NOT trigger recalibration; it is kept as an audit note
    only.
    """

    decision_id: str
    correct: Optional[bool]
    reward: Optional[float] = None
    note: str = ""
    ground_truth: Any = None
    source: str = "explicit"
    reported_at: float = field(default_factory=time.time)
