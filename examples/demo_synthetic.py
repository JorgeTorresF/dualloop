"""End-to-end synthetic demo of DualLoop.

Simulates a toy world with three heterogeneous engines of differing
reliability and cost, and shows empirically that:

1. The cascade resolves most cases with the cheap engine and escalates to
   the expensive one only when it has to (cost saving).
2. Confidence calibration improves with experience (the cheap engine
   starts deliberately MIScalibrated -- overconfident on hard cases -- so
   that the effect of correcting it is visible).
3. The accuracy of accepted answers stays stable or improves over time,
   with no manual retraining: everything goes through `report_outcome()`.

IMPORTANT: this is a SYNTHETIC reference benchmark, not a production
validation. It verifies that the recalibration mechanism behaves as
designed; it is not evidence that it solves any real domain. It needs no
network and no credentials: all three engines are local simulations.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualloop import DualLoop, InMemoryStore, Question  # noqa: E402
from dualloop.engines.base import BaseEngine  # noqa: E402

LABELS = ["A", "B"]
N_DECISIONS = 800
WINDOW = 50
SEED = 7


class SimFastEngine(BaseEngine):
    """Simulates a Jev-style typed classifier: cheap, fast, but deliberately
    OVERCONFIDENT on hard cases, so that the effect of calibrating it is
    visible in the metrics."""

    name = "jev_sim"
    relative_cost = 0.1

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _decide_raw(self, task_type: str, question: Question):
        true_label = question.state.split("|")[1]
        difficulty = question.state.split("|")[2]
        acc = 0.92 if difficulty == "easy" else 0.55
        correct = self._rng.random() < acc
        value = true_label if correct else next(o for o in LABELS if o != true_label)
        if difficulty == "easy":
            confidence = self._rng.uniform(0.78, 0.98)
        else:
            # Overconfident even when wrong: this is precisely what
            # calibration has to correct as experience accumulates.
            confidence = self._rng.uniform(0.65, 0.92)
        return value, confidence, {"sim": "fast", "correct": correct}


class SimSlowEngine(BaseEngine):
    """Simula un LLM de razonamiento: caro, lento, mejor calibrado y más
    fiable en casos difíciles."""

    name = "llm_sim"
    relative_cost = 5.0

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _decide_raw(self, task_type: str, question: Question):
        true_label = question.state.split("|")[1]
        difficulty = question.state.split("|")[2]
        acc = 0.85 if difficulty == "easy" else 0.90
        correct = self._rng.random() < acc
        value = true_label if correct else next(o for o in LABELS if o != true_label)
        confidence = self._rng.uniform(0.7, 0.95) if correct else self._rng.uniform(0.4, 0.7)
        return value, confidence, {"sim": "slow", "correct": correct}


class SimRuleEngine(BaseEngine):
    """Simulates a business rule: free and instant, but only covers the
    subset of cases flagged as 'obvious'."""

    name = "rules_sim"
    relative_cost = 0.01

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _decide_raw(self, task_type: str, question: Question):
        true_label, difficulty, obvious = question.state.split("|")[1:4]
        if obvious != "yes":
            raise ValueError("The rule does not cover this case (not obvious)")
        correct = self._rng.random() < 0.98
        value = true_label if correct else next(o for o in LABELS if o != true_label)
        return value, 0.99, {"sim": "rule", "correct": correct}


def make_question(true_label: str, difficulty: str, obvious: str) -> Question:
    # The state encodes the simulated ground truth so the engines can
    # "cheat" in a controlled way. In a real case the state would carry the
    # actual business context, never the answer.
    state = f"synthetic_case|{true_label}|{difficulty}|{obvious}"
    return Question(
        type="choice",
        instructions="Choose the correct category for this synthetic case.",
        state=state,
        criteria={"A": None, "B": None},
    )


def windowed(values: list[float], window: int) -> list[float]:
    return [statistics.mean(values[i : i + window]) for i in range(0, len(values), window)]


def main() -> None:
    rng = random.Random(SEED)
    loop = DualLoop(
        engines=[SimFastEngine(rng), SimSlowEngine(rng), SimRuleEngine(rng)],
        store=InMemoryStore(),
        seed=SEED,
        default_threshold=0.75,
    )

    accuracy_series: list[float] = []
    n_engines_series: list[int] = []
    calib_error_series: list[float] = []

    for i in range(N_DECISIONS):
        true_label = rng.choice(LABELS)
        difficulty = "easy" if rng.random() < 0.7 else "hard"
        obvious = "yes" if rng.random() < 0.15 else "no"
        question = make_question(true_label, difficulty, obvious)

        decision = loop.decide("synthetic_case", question)
        correct = decision.chosen_value == true_label

        per_engine_correct = {v.engine_name: (v.value == true_label) for v in decision.votes}
        loop.report_outcome(decision.id, correct=correct, per_engine_correct=per_engine_correct)

        accuracy_series.append(1.0 if correct else 0.0)
        n_engines_series.append(len(decision.votes))
        calib_error_series.append(abs(decision.chosen_confidence - (1.0 if correct else 0.0)))

    acc_w = windowed(accuracy_series, WINDOW)
    n_w = windowed([float(n) for n in n_engines_series], WINDOW)
    calib_w = windowed(calib_error_series, WINDOW)

    print(f"DualLoop - synthetic demo ({N_DECISIONS} decisions, window={WINDOW})\n")
    print(f"{'window':>8} | {'accuracy':>9} | {'engines/decision':>16} | {'calibration err':>18}")
    print("-" * 62)
    for idx in (0, len(acc_w) // 2, len(acc_w) - 1):
        print(f"{idx * WINDOW:>8} | {acc_w[idx]:>9.3f} | {n_w[idx]:>16.2f} | {calib_w[idx]:>18.3f}")

    print("\nFirst window vs last window:")
    print(f"  accuracy:             {acc_w[0]:.3f} -> {acc_w[-1]:.3f}")
    print(f"  engines per decision: {n_w[0]:.2f} -> {n_w[-1]:.2f}  (lower = cheaper)")
    print(f"  calibration error:    {calib_w[0]:.3f} -> {calib_w[-1]:.3f}  (lower = better calibrated)")

    # ---- audit trail for one concrete decision ----
    last_decision_id = decision.id
    explanation = loop.explain(last_decision_id)
    print(f"\nAudit of the last decision ({last_decision_id[:8]}...):")
    print(f"  task_type: {explanation['decision'].task_type}")
    print(f"  chosen engine: {explanation['decision'].chosen_engine}")
    print(f"  calibrated confidence: {explanation['decision'].chosen_confidence:.3f}")
    print(f"  threshold in force: {explanation['decision'].escalation_threshold_used:.3f}")
    print(f"  learned reliability per engine: {explanation['reliability_at_decision_time']}")
    print(f"  outcomes recorded: {len(explanation['outcomes'])}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        x = [i * WINDOW for i in range(len(acc_w))]
        axes[0].plot(x, acc_w)
        axes[0].set_ylabel("accuracy")
        axes[1].plot(x, n_w)
        axes[1].set_ylabel("engines/decision")
        axes[2].plot(x, calib_w)
        axes[2].set_ylabel("calibration error")
        axes[2].set_xlabel("decision #")
        fig.suptitle("DualLoop - online recalibration on synthetic data")
        out_path = Path(__file__).with_name("demo_synthetic_result.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\n(install the 'demo' extra -> pip install '.[demo]' to generate the plot)")


if __name__ == "__main__":
    main()
