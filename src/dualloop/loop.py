"""The library's single entry point.

`DualLoop` combines heterogeneous engines (a reasoning LLM, a Jev-style
typed classifier, rules...) under a learned, auditable arbiter, and closes
the loop decision -> outcome -> recalibration through `report_outcome()`,
retraining no model at all: each call updates three online statistics
(per-engine calibration, per-engine reliability, per-task-type acceptance
threshold) with a closed-form Bayesian update in O(1).
"""

from __future__ import annotations

from typing import Optional

from .arbiter import Arbiter
from .bandit import AdaptiveThreshold, ReliabilityBandit
from .calibration import ConfidenceCalibrator
from .engines.base import BaseEngine
from .heuristics import OutcomeHeuristic
from .store import InMemoryStore, Store
from .types import Decision, Outcome, Question, Vote


class DualLoop:
    def __init__(
        self,
        engines: list[BaseEngine],
        *,
        store: Optional[Store] = None,
        seed: Optional[int] = None,
        default_threshold: float = 0.85,
        threshold_lo: Optional[float] = None,
        threshold_hi: Optional[float] = None,
        max_engines_per_decision: Optional[int] = None,
        heuristics: Optional[list[OutcomeHeuristic]] = None,
        abstain_below_threshold: bool = False,
    ) -> None:
        if not engines:
            raise ValueError("DualLoop needs at least one engine")

        self.store = store or InMemoryStore()
        by_name = {e.name: e for e in engines}
        if len(by_name) != len(engines):
            raise ValueError("Engine names must be unique")
        # Initial order by ascending cost: it informs the bandit's
        # optimistic prior (see ReliabilityBandit), it does not decide the
        # outcome.
        self._engines_by_name = dict(sorted(by_name.items(), key=lambda kv: kv[1].relative_cost))

        self._calibrators: dict[tuple[str, str], ConfidenceCalibrator] = {}
        cost_by_engine = {name: e.relative_cost for name, e in self._engines_by_name.items()}
        self._bandit = ReliabilityBandit(cost_by_engine=cost_by_engine, seed=seed)
        # lo/hi are only forwarded when the caller sets them, so that
        # AdaptiveThreshold's defaults are not duplicated here.
        bounds = {}
        if threshold_lo is not None:
            bounds["lo"] = threshold_lo
        if threshold_hi is not None:
            bounds["hi"] = threshold_hi
        self._threshold = AdaptiveThreshold(default=default_threshold, **bounds)
        self._arbiter = Arbiter(
            self._engines_by_name,
            self._calibrators,
            self._bandit,
            self._threshold,
            max_engines_per_decision=max_engines_per_decision,
            abstain_below_threshold=abstain_below_threshold,
        )
        self._heuristics = heuristics or []
        self._load_state()

    # ---------------------------------------------------------------- decidir
    def decide(
        self,
        task_type: str,
        question: Question,
        engine_subset: Optional[list[str]] = None,
    ) -> Decision:
        result = self._arbiter.decide(task_type, question, engine_subset=engine_subset)
        self.store.save_decision(result.decision)
        return result.decision

    # ------------------------------------------------------- cerrar el bucle
    def report_outcome(
        self,
        decision_id: str,
        correct: Optional[bool] = None,
        *,
        reward: Optional[float] = None,
        note: str = "",
        ground_truth: object = None,
        source: str = "explicit",
        per_engine_correct: Optional[dict[str, bool]] = None,
    ) -> Outcome:
        """Report what actually happened after a decision.

        `correct=None` records the outcome as an audit note without
        recalibrating (an ambiguous or still-unknown result).
        `per_engine_correct` lets you supply the truth per engine when you
        know it -- from an offline evaluation set, say -- for a sharper
        recalibration than the default inference, which is only certain
        about the chosen engine and those that agreed with it. See
        `_infer_vote_correct`, which leaves un-updated any vote whose truth
        cannot be deduced.
        """
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(f"No decision exists with id={decision_id!r}")

        outcome = Outcome(
            decision_id=decision_id,
            correct=correct,
            reward=reward,
            note=note,
            ground_truth=ground_truth,
            source=source,
        )
        self.store.save_outcome(outcome)

        if correct is not None or per_engine_correct:
            self._recalibrate(decision, correct, per_engine_correct)

        return outcome

    def try_infer_outcomes(self, decision_id: str, context: dict) -> list[Outcome]:
        """Run the optional registered heuristics against a decision. It
        is never called automatically: whoever integrates the library
        decides when to invoke it."""
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(f"No decision exists with id={decision_id!r}")

        inferred: list[Outcome] = []
        for heuristic in self._heuristics:
            outcome = heuristic.try_infer(decision, context)
            if outcome is not None:
                self.store.save_outcome(outcome)
                if outcome.correct is not None:
                    self._recalibrate(decision, outcome.correct, None)
                inferred.append(outcome)
        return inferred

    @staticmethod
    def _infer_vote_correct(
        decision: Decision, vote: Vote, correct: bool
    ) -> Optional[bool]:
        """Infer whether a vote was right, from the chosen answer's outcome.

        The chosen engine gets the exact signal. For the others:

        - if the chosen answer was right, whoever dissented was wrong;
        - if the chosen answer was wrong, whoever agreed was wrong;
        - if the chosen answer was wrong and the engine dissented, it
          depends on how many possible answers there were. With two (a
          binary ``choice`` or a ``noul``) the dissenter was necessarily
          right. With more than two, knowing the chosen answer was false
          does NOT identify which one was true.

        Returns ``None`` when the vote's truth is unknown; the caller then
        updates nothing. This is deliberately conservative: feeding the
        calibrator an invented label biases the system towards consensus,
        which is precisely what an arbiter across heterogeneous engines
        must not do. If you would rather have more signal at the price of
        assuming uniformity across the remaining options, this is the only
        line to change -- for instance by returning an update weighted by
        1/(n_options - 1).
        """
        if vote.engine_name == decision.chosen_engine:
            return correct

        agreed = vote.value == decision.chosen_value
        if correct:
            return agreed
        if agreed:
            return False

        n_options = (
            2
            if decision.question.type == "noul"
            else len(decision.question.criteria or ())
        )
        return True if n_options == 2 else None

    def _recalibrate(
        self,
        decision: Decision,
        correct: Optional[bool],
        per_engine_correct: Optional[dict[str, bool]],
    ) -> None:
        for vote in decision.votes:
            if per_engine_correct and vote.engine_name in per_engine_correct:
                vote_correct: Optional[bool] = per_engine_correct[vote.engine_name]
            elif correct is not None:
                vote_correct = self._infer_vote_correct(decision, vote, correct)
            else:
                continue
            if vote_correct is None:
                continue
            calibrator = self._calibrators.setdefault(
                (decision.task_type, vote.engine_name), ConfidenceCalibrator()
            )
            calibrator.update(vote.raw_confidence, vote_correct)

        if correct is not None:
            self._bandit.update(decision.task_type, decision.chosen_engine, correct)
            # The threshold targets the error rate among ACCEPTED
            # answers. A decision the loop abstained on was not accepted,
            # so feeding it here breaks the threshold's semantics: it
            # would raise the bar for an error the system had already
            # flagged as doubtful. The calibrators and the bandit do learn
            # from it.
            if not decision.abstained:
                self._threshold.update_on_accepted_outcome(decision.task_type, correct)

        self._save_state()

    # ------------------------------------------------------------- audit
    def explain(self, decision_id: str) -> dict:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(f"No decision exists with id={decision_id!r}")
        outcomes = self.store.get_outcomes(decision_id)
        return {
            "decision": decision,
            "outcomes": outcomes,
            "reliability_at_decision_time": {
                v.engine_name: self._bandit.mean(decision.task_type, v.engine_name)
                for v in decision.votes
            },
            "current_threshold": self._threshold.get(decision.task_type),
        }

    # ------------------------------------------------- state persistence
    def _save_state(self) -> None:
        # Calibrators are stored in a single blob, rewritten in full on
        # every outcome: the cost is O(task_type x engine pairs). This is
        # deliberate. One blob per calibrator would also require keeping an
        # index, because the `Store` protocol exposes no way to enumerate
        # keys, and the saving only shows up with hundreds of task types.
        # If it ever does, that is the change: a "calibrators::index" blob
        # holding the keys, plus one blob per calibrator.
        self.store.save_state_blob("bandit", self._bandit.state_dict())
        self.store.save_state_blob("threshold", self._threshold.state_dict())
        self.store.save_state_blob(
            "calibrators",
            {f"{t}||{e}": c.state_dict() for (t, e), c in self._calibrators.items()},
        )

    def _load_state(self) -> None:
        bandit_state = self.store.load_state_blob("bandit")
        if bandit_state:
            self._bandit.load_state_dict(bandit_state)

        threshold_state = self.store.load_state_blob("threshold")
        if threshold_state:
            self._threshold.load_state_dict(threshold_state)

        calibrators_state = self.store.load_state_blob("calibrators")
        if calibrators_state:
            for key, state in calibrators_state.items():
                t, e = key.split("||", 1)
                calibrator = ConfidenceCalibrator()
                calibrator.load_state_dict(state)
                self._calibrators[(t, e)] = calibrator
