"""Recalibration with SEVERAL engines that disagree.

This file deliberately covers the seam between the two halves of the
suite: `test_arbiter.py` exercises multi-engine arbitration but never
recalibrates, and `test_loop_end_to_end.py` recalibrates but with a single
engine. Without a second engine there is no dissenter, so the most delicate
case -- what happens to the engine that dissented when the chosen answer
turned out wrong -- was observed by no test at all.
"""

from dualloop import DualLoop, InMemoryStore, Question
from dualloop.engines.base import BaseEngine
from dualloop.types import Decision, Vote

RAW = 0.8

BINARY = Question(
    type="choice", instructions="pick one", state="ctx",
    criteria={"A": None, "B": None},
)
TERNARY = Question(
    type="choice", instructions="pick one", state="ctx",
    criteria={"A": None, "B": None, "C": None},
)


class FixedEngine(BaseEngine):
    def __init__(self, name: str, value: str, cost: float = 1.0):
        self.name = name
        self.relative_cost = cost
        self._value = value

    def _decide_raw(self, task_type, question):
        return self._value, RAW, {}


def _loop_with_decision(question, chosen_value, other_value):
    """Build a DualLoop plus an already-made Decision carrying two votes."""
    loop = DualLoop(
        engines=[FixedEngine("chosen", chosen_value), FixedEngine("other", other_value)],
        store=InMemoryStore(),
    )
    decision = Decision(
        id=Decision.new_id(),
        task_type="t",
        question=question,
        votes=[
            Vote("chosen", chosen_value, RAW, RAW, 1.0, {}),
            Vote("other", other_value, RAW, RAW, 1.0, {}),
        ],
        chosen_engine="chosen",
        chosen_value=chosen_value,
        chosen_confidence=RAW,
        escalation_threshold_used=0.7,
        disagreement=chosen_value != other_value,
        )
    loop.store.save_decision(decision)
    return loop, decision


def _bin_of(loop, engine_name):
    """(alpha, beta) of the bin matching RAW for that engine, or None if the
    engine has no calibrator at all -- meaning nothing was updated."""
    calibrator = loop._calibrators.get(("t", engine_name))
    if calibrator is None:
        return None
    b = calibrator._bins[calibrator._bin_index(RAW)]
    return (b.alpha, b.beta)


def test_dissenter_is_right_when_the_chosen_answer_fails_on_a_binary():
    # With two options, if the chosen answer was wrong the other one was
    # right: the engine that dissented was correct, and punishing it would
    # bias the whole system towards consensus.
    loop, decision = _loop_with_decision(BINARY, "A", "B")
    loop.report_outcome(decision.id, correct=False)

    assert _bin_of(loop, "chosen") == (1.0, 2.0), "the chosen engine was wrong"
    assert _bin_of(loop, "other") == (2.0, 1.0), "the dissenter was right"


def test_dissenter_is_not_updated_with_more_than_two_options():
    # With three options, knowing that 'A' was false does not tell us 'B'
    # was true: the dissenter's truth is unknown and must not be invented.
    loop, decision = _loop_with_decision(TERNARY, "A", "B")
    loop.report_outcome(decision.id, correct=False)

    assert _bin_of(loop, "chosen") == (1.0, 2.0), "the chosen engine was wrong"
    assert _bin_of(loop, "other") is None, "the dissenter's truth is unknown"


def test_dissenter_is_wrong_when_the_chosen_answer_is_right():
    loop, decision = _loop_with_decision(BINARY, "A", "B")
    loop.report_outcome(decision.id, correct=True)

    assert _bin_of(loop, "chosen") == (2.0, 1.0)
    assert _bin_of(loop, "other") == (1.0, 2.0), "dissenting from a right answer is being wrong"


def test_an_agreeing_engine_shares_the_chosen_answers_fate():
    loop, decision = _loop_with_decision(BINARY, "A", "A")
    loop.report_outcome(decision.id, correct=False)

    assert _bin_of(loop, "chosen") == (1.0, 2.0)
    assert _bin_of(loop, "other") == (1.0, 2.0), "agreeing with a wrong answer is being wrong"


def test_per_engine_correct_takes_precedence_over_inference():
    # With three options the inference abstains, but if the caller supplies
    # the per-engine truth, that wins.
    loop, decision = _loop_with_decision(TERNARY, "A", "B")
    loop.report_outcome(decision.id, correct=False, per_engine_correct={"other": True})

    assert _bin_of(loop, "other") == (2.0, 1.0), "explicit truth wins"
