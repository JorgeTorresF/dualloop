"""The arbiter: orchestrates the consultation cascade across heterogeneous
engines.

Flow for each decision:
1. Rank the engines available for this task_type by learned reliability
   (ReliabilityBandit.rank), biased towards cheap ones while there is no
   evidence yet.
2. Consult the first, calibrate its raw confidence (ConfidenceCalibrator).
3. If the calibrated confidence clears the task type's adaptive threshold,
   accept that answer and do NOT escalate further (cost saving).
4. Otherwise move to the next engine in the learned order, until the list
   or the `max_engines_per_decision` limit runs out.
5. Keep the vote with the highest calibrated confidence among all engines
   consulted, and set `disagreement=True` if they did not agree.

The whole process is recorded in the resulting `Decision` -- which engines
were consulted, in what order, their raw and calibrated outputs, and the
threshold used -- so that `DualLoop.explain()` can reconstruct why the
decision came out the way it did.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bandit import AdaptiveThreshold, ReliabilityBandit
from .calibration import ConfidenceCalibrator
from .engines.base import BaseEngine
from .types import Decision, EngineOutput, Question, Vote


@dataclass
class ArbitrationResult:
    decision: Decision
    consulted_engines: list[str]


class Arbiter:
    def __init__(
        self,
        engines: dict[str, BaseEngine],
        calibrators: dict[tuple[str, str], ConfidenceCalibrator],
        bandit: ReliabilityBandit,
        threshold: AdaptiveThreshold,
        *,
        max_engines_per_decision: int | None = None,
        abstain_below_threshold: bool = False,
    ) -> None:
        self._engines = engines
        self._calibrators = calibrators
        self._bandit = bandit
        self._threshold = threshold
        self._max_engines = max_engines_per_decision or len(engines)
        self._abstain_below_threshold = abstain_below_threshold

    def _calibrator_for(self, task_type: str, engine_name: str) -> ConfidenceCalibrator:
        key = (task_type, engine_name)
        if key not in self._calibrators:
            self._calibrators[key] = ConfidenceCalibrator()
        return self._calibrators[key]

    def decide(
        self,
        task_type: str,
        question: Question,
        engine_subset: list[str] | None = None,
    ) -> ArbitrationResult:
        names = engine_subset or list(self._engines.keys())
        if not names:
            raise ValueError("No engines available to arbitrate this decision")

        order = self._bandit.rank(task_type, names)
        threshold = self._threshold.get(task_type)

        votes: list[Vote] = []
        consulted: list[str] = []
        best: Vote | None = None

        for engine_name in order[: self._max_engines]:
            engine = self._engines[engine_name]
            output: EngineOutput = engine.decide(task_type, question)
            consulted.append(engine_name)
            if not output.ok:
                continue

            calibrator = self._calibrator_for(task_type, engine_name)
            calibrated = calibrator.calibrate(output.raw_confidence)
            vote = Vote(
                engine_name=engine_name,
                value=output.value,
                raw_confidence=output.raw_confidence,
                calibrated_confidence=calibrated,
                latency_ms=output.latency_ms,
                raw=output.raw,
            )
            votes.append(vote)
            if best is None or vote.calibrated_confidence > best.calibrated_confidence:
                best = vote
            if calibrated >= threshold:
                break  # confident enough: stop escalating

        if best is None:
            raise RuntimeError(
                f"No engine could answer for task_type={task_type!r} "
                f"(engines consulted: {consulted})"
            )

        disagreement = len({v.value for v in votes}) > 1
        # If the cascade ran out without anyone clearing the threshold,
        # the system has no answer it trusts. Saying so is the honest
        # move: deciding anyway makes the threshold decorative.
        abstained = (
            self._abstain_below_threshold and best.calibrated_confidence < threshold
        )

        decision = Decision(
            id=Decision.new_id(),
            task_type=task_type,
            question=question,
            votes=votes,
            chosen_engine=best.engine_name,
            chosen_value=best.value,
            chosen_confidence=best.calibrated_confidence,
            escalation_threshold_used=threshold,
            disagreement=disagreement,
            abstained=abstained,
        )
        return ArbitrationResult(decision=decision, consulted_engines=consulted)
