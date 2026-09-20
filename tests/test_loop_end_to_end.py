import tempfile
from pathlib import Path

from dualloop import DualLoop, InMemoryStore, Question, SQLiteStore
from dualloop.engines.base import BaseEngine


class TogglableEngine(BaseEngine):
    """Motor de prueba cuyo acierto se puede fijar desde fuera para
    simular una serie controlada de outcomes."""

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
    loop = DualLoop(engines=[engine], store=InMemoryStore(), default_threshold=0.5)

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

    # el motor siempre dice 0.8 de confianza pero se equivoca sistematicamente
    engine.next_correct = False
    for _ in range(60):
        decision = loop.decide("t", QUESTION)
        loop.report_outcome(decision.id, correct=False)

    calibrated_conf = loop.explain(decision.id)["decision"].chosen_confidence
    assert calibrated_conf < 0.5, f"la confianza calibrada deberia haber bajado, obtuve {calibrated_conf}"


def test_missing_decision_raises_keyerror():
    engine = TogglableEngine("only")
    loop = DualLoop(engines=[engine])
    try:
        loop.report_outcome("no-existe", correct=True)
        assert False, "se esperaba KeyError"
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

        # nueva instancia, mismo fichero: el estado de calibracion/bandit
        # debe recuperarse sin volver a observar nada
        engine2 = TogglableEngine("only")
        loop2 = DualLoop(engines=[engine2], store=SQLiteStore(db_path), default_threshold=0.99)
        new_decision = loop2.decide("t", QUESTION)
        # como el motor sigue "equivocandose" con confianza 0.8, la
        # confianza calibrada en la nueva instancia debe seguir siendo baja
        assert new_decision.chosen_confidence < 0.5
        assert conf_before_reload < 0.5
