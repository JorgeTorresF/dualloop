"""Abstention: what happens when no engine clears the threshold.

Without `abstain_below_threshold` the loop accepts the best vote anyway,
below the threshold: the threshold is recorded but never acts as a floor.
For a gate where "I don't know, let a person look at it" is a legitimate
answer, that is exactly what you do not want.
"""

from dualloop import DualLoop, InMemoryStore, Question
from dualloop.engines.base import BaseEngine

QUESTION = Question(
    type="choice", instructions="pick one", state="ctx",
    criteria={"A": None, "B": None},
)


class FixedEngine(BaseEngine):
    def __init__(self, name: str, value: str, confidence: float, cost: float = 1.0):
        self.name = name
        self.relative_cost = cost
        self._value = value
        self._confidence = confidence

    def _decide_raw(self, task_type, question):
        return self._value, self._confidence, {}


def _loop(abstain: bool, confidence: float = 0.3, threshold: float = 0.9):
    return DualLoop(
        engines=[
            FixedEngine("a", "A", confidence, cost=0.1),
            FixedEngine("b", "B", confidence, cost=5.0),
        ],
        store=InMemoryStore(),
        seed=0,
        default_threshold=threshold,
        abstain_below_threshold=abstain,
    )


def test_by_default_it_does_not_abstain_and_decides_anyway():
    decision = _loop(abstain=False).decide("t", QUESTION)
    assert decision.abstained is False
    assert decision.chosen_confidence < decision.escalation_threshold_used
    assert decision.chosen_value in ("A", "B"), "still decides: previous behaviour"


def test_it_abstains_when_nobody_clears_the_threshold():
    decision = _loop(abstain=True).decide("t", QUESTION)
    assert decision.abstained is True
    assert decision.chosen_confidence < decision.escalation_threshold_used
    assert decision.chosen_value in ("A", "B"), "the best vote stays available to the human"


def test_it_does_not_abstain_when_someone_clears_the_threshold():
    decision = _loop(abstain=True, confidence=0.95, threshold=0.7).decide("t", QUESTION)
    assert decision.abstained is False


def test_an_abstention_does_not_move_the_threshold_but_does_move_the_bandit():
    # The threshold targets the error rate among ACCEPTED answers. If the
    # system abstained, that case was not accepted and must not tighten it.
    loop = _loop(abstain=True)
    before = loop._threshold.get("t")

    decision = loop.decide("t", QUESTION)
    assert decision.abstained is True
    loop.report_outcome(decision.id, correct=False)

    assert loop._threshold.get("t") == before, "an abstention must not move the threshold"
    assert loop._bandit.mean("t", decision.chosen_engine) < 1.0, "the bandit does learn"


def test_without_abstention_the_same_failure_does_tighten_the_threshold():
    # Control: same scenario with abstention switched off.
    loop = _loop(abstain=False)
    before = loop._threshold.get("t")

    decision = loop.decide("t", QUESTION)
    loop.report_outcome(decision.id, correct=False)

    assert loop._threshold.get("t") > before, "an accepted failure does tighten the threshold"
