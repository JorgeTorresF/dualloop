from dualloop.arbiter import Arbiter
from dualloop.bandit import AdaptiveThreshold, ReliabilityBandit
from dualloop.calibration import ConfidenceCalibrator
from dualloop.engines.base import BaseEngine
from dualloop.types import Question


class FixedEngine(BaseEngine):
    def __init__(self, name: str, value, confidence: float, cost: float = 1.0, fail: bool = False):
        self.name = name
        self.relative_cost = cost
        self._value = value
        self._confidence = confidence
        self._fail = fail

    def _decide_raw(self, task_type, question):
        if self._fail:
            raise ValueError("motor no disponible")
        return self._value, self._confidence, {}


def make_arbiter(engines, threshold_default=0.7, max_engines=None):
    engines_by_name = {e.name: e for e in engines}
    calibrators = {}
    bandit = ReliabilityBandit(cost_by_engine={e.name: e.relative_cost for e in engines})
    threshold = AdaptiveThreshold(default=threshold_default)
    return Arbiter(engines_by_name, calibrators, bandit, threshold, max_engines_per_decision=max_engines)


QUESTION = Question(type="choice", instructions="elige", state="ctx", criteria={"A": None, "B": None})


def test_accepts_cheap_engine_when_confident_enough():
    cheap = FixedEngine("cheap", "A", 0.95, cost=0.1)
    expensive = FixedEngine("expensive", "B", 0.6, cost=5.0)
    arbiter = make_arbiter([cheap, expensive], threshold_default=0.7)

    result = arbiter.decide("t", QUESTION)
    assert result.decision.chosen_engine == "cheap"
    assert len(result.consulted_engines) == 1  # no escalo


def test_escalates_when_cheap_engine_not_confident():
    cheap = FixedEngine("cheap", "A", 0.3, cost=0.1)
    expensive = FixedEngine("expensive", "B", 0.9, cost=5.0)
    arbiter = make_arbiter([cheap, expensive], threshold_default=0.7)

    result = arbiter.decide("t", QUESTION)
    assert len(result.consulted_engines) == 2
    assert result.decision.chosen_engine == "expensive"


def test_disagreement_flag_set_when_engines_differ():
    cheap = FixedEngine("cheap", "A", 0.3, cost=0.1)
    expensive = FixedEngine("expensive", "B", 0.5, cost=5.0)
    arbiter = make_arbiter([cheap, expensive], threshold_default=0.99)

    result = arbiter.decide("t", QUESTION)
    assert result.decision.disagreement is True


def test_skips_failing_engine_and_uses_next():
    broken = FixedEngine("broken", "A", 0.99, cost=0.1, fail=True)
    working = FixedEngine("working", "B", 0.8, cost=5.0)
    arbiter = make_arbiter([broken, working], threshold_default=0.5)

    result = arbiter.decide("t", QUESTION)
    assert result.decision.chosen_engine == "working"


def test_raises_when_all_engines_fail():
    broken1 = FixedEngine("b1", "A", 0.9, fail=True)
    broken2 = FixedEngine("b2", "B", 0.9, fail=True)
    arbiter = make_arbiter([broken1, broken2])

    try:
        arbiter.decide("t", QUESTION)
        assert False, "se esperaba RuntimeError"
    except RuntimeError:
        pass
