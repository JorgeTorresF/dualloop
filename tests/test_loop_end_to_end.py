import tempfile
from pathlib import Path

from dualloop import DualLoop, InMemoryStore, Question, SQLiteStore
from dualloop.engines.base import BaseEngine


class TogglableEngine(BaseEngine):
    """Test engine whose correctness can be set from outside, to simulate
    a controlled series of outcomes."""

    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.relative_cost = cost
        self.next_correct = True
        self.true_label = "A"

    def _decide_raw(self, task_type, question):
        value = self.true_label if self.next_correct else "B"
        return value, 0.8, {}


QUESTION = Question(type="choice", instructions="elige", state="ctx", criteria={"A": None, "B": None})


def test_decide_and_report_outcome_updates_state():
    engine = TogglableEngine("only")
    loop = DualLoop(
        engines=[engine], store=InMemoryStore(), default_threshold=0.5, threshold_lo=0.0
    )

    decision = loop.decide("t", QUESTION)
    assert decision.chosen_value == "A"

    outcome = loop.report_outcome(decision.id, correct=True)
    assert outcome.decision_id == decision.id

    explanation = loop.explain(decision.id)
    assert explanation["decision"].id == decision.id
    assert len(explanation["outcomes"]) == 1


def test_recalibration_shifts_confidence_after_many_wrong_outcomes():
    engine = TogglableEngine("only")
    loop = DualLoop(engines=[engine], store=InMemoryStore(), default_threshold=0.99)

    # The engine always reports 0.8 confidence but is systematically wrong.
    engine.next_correct = False
    for _ in range(60):
        decision = loop.decide("t", QUESTION)
        loop.report_outcome(decision.id, correct=False)

    calibrated_conf = loop.explain(decision.id)["decision"].chosen_confidence
    assert calibrated_conf < 0.5, f"calibrated confidence should have dropped, got {calibrated_conf}"


def test_missing_decision_raises_keyerror():
    engine = TogglableEngine("only")
    loop = DualLoop(engines=[engine])
    try:
        loop.report_outcome("does-not-exist", correct=True)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_sqlite_store_persists_state_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "dualloop_test.db")

        engine1 = TogglableEngine("only")
        loop1 = DualLoop(engines=[engine1], store=SQLiteStore(db_path), default_threshold=0.99)
        engine1.next_correct = False
        decision = None
        for _ in range(30):
            decision = loop1.decide("t", QUESTION)
            loop1.report_outcome(decision.id, correct=False)
        conf_before_reload = loop1.explain(decision.id)["decision"].chosen_confidence

        # New instance, same file: calibration and bandit state must come
        # back without observing anything again.
        engine2 = TogglableEngine("only")
        loop2 = DualLoop(engines=[engine2], store=SQLiteStore(db_path), default_threshold=0.99)
        new_decision = loop2.decide("t", QUESTION)
        # Since the engine keeps being wrong at 0.8 confidence, the
        # calibrated confidence in the new instance must stay low.
        assert new_decision.chosen_confidence < 0.5
        assert conf_before_reload < 0.5
