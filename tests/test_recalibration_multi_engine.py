"""Recalibracion con VARIOS motores que discrepan.

Este fichero cubre deliberadamente la costura entre las dos mitades de la
suite: `test_arbiter.py` prueba el arbitraje multi-motor pero no recalibra,
y `test_loop_end_to_end.py` recalibra pero con un solo motor. Sin un
segundo motor no existe discrepante, asi que el caso mas delicado
-que pasa con quien discrepo cuando la decision elegida fallo- no lo
observaba ningun test.
"""

from dualloop import DualLoop, InMemoryStore, Question
from dualloop.engines.base import BaseEngine
from dualloop.types import Decision, Vote

RAW = 0.8

BINARIA = Question(
    type="choice", instructions="elige", state="ctx",
    criteria={"A": None, "B": None},
)
TERNARIA = Question(
    type="choice", instructions="elige", state="ctx",
    criteria={"A": None, "B": None, "C": None},
)


class FixedEngine(BaseEngine):
    def __init__(self, name: str, value: str, cost: float = 1.0):
        self.name = name
        self.relative_cost = cost
        self._value = value

    def _decide_raw(self, task_type, question):
        return self._value, RAW, {}


def _loop_con_decision(question, chosen_value, other_value):
    """Construye un DualLoop y una Decision ya tomada con dos votos."""
    loop = DualLoop(
        engines=[FixedEngine("elegido", chosen_value), FixedEngine("otro", other_value)],
        store=InMemoryStore(),
    )
    decision = Decision(
        id=Decision.new_id(),
        task_type="t",
        question=question,
        votes=[
            Vote("elegido", chosen_value, RAW, RAW, 1.0, {}),
            Vote("otro", other_value, RAW, RAW, 1.0, {}),
        ],
        chosen_engine="elegido",
        chosen_value=chosen_value,
        chosen_confidence=RAW,
        escalation_threshold_used=0.7,
        disagreement=chosen_value != other_value,
        )
    loop.store.save_decision(decision)
    return loop, decision


def _bin_de(loop, engine_name):
    """(alpha, beta) del bin que corresponde a RAW para ese motor, o None
    si el motor no tiene calibrador (no se actualizo nada)."""
    calibrator = loop._calibrators.get(("t", engine_name))
    if calibrator is None:
        return None
    b = calibrator._bins[calibrator._bin_index(RAW)]
    return (b.alpha, b.beta)


def test_discrepante_acierta_cuando_la_elegida_falla_en_binaria():
    # Con dos opciones, si la elegida fallo la otra era la buena: el motor
    # que discrepo acerto, y castigarlo sesgaria el sistema al consenso.
    loop, decision = _loop_con_decision(BINARIA, "A", "B")
    loop.report_outcome(decision.id, correct=False)

    assert _bin_de(loop, "elegido") == (1.0, 2.0), "el elegido fallo"
    assert _bin_de(loop, "otro") == (2.0, 1.0), "el discrepante acerto"


def test_discrepante_no_se_actualiza_con_mas_de_dos_opciones():
    # Con tres opciones, saber que 'A' era falsa no dice que 'B' fuera la
    # buena: la verdad del discrepante es desconocida y no se inventa.
    loop, decision = _loop_con_decision(TERNARIA, "A", "B")
    loop.report_outcome(decision.id, correct=False)

    assert _bin_de(loop, "elegido") == (1.0, 2.0), "el elegido fallo"
    assert _bin_de(loop, "otro") is None, "la verdad del discrepante es desconocida"


def test_discrepante_falla_cuando_la_elegida_acierta():
    loop, decision = _loop_con_decision(BINARIA, "A", "B")
    loop.report_outcome(decision.id, correct=True)

    assert _bin_de(loop, "elegido") == (2.0, 1.0)
    assert _bin_de(loop, "otro") == (1.0, 2.0), "discrepar de un acierto es fallar"


def test_quien_coincide_comparte_el_destino_de_la_elegida():
    loop, decision = _loop_con_decision(BINARIA, "A", "A")
    loop.report_outcome(decision.id, correct=False)

    assert _bin_de(loop, "elegido") == (1.0, 2.0)
    assert _bin_de(loop, "otro") == (1.0, 2.0), "coincidir con un fallo es fallar"


def test_per_engine_correct_tiene_prioridad_sobre_la_inferencia():
    # Con tres opciones la inferencia se abstiene, pero si el llamante
    # aporta la verdad por motor, esa manda.
    loop, decision = _loop_con_decision(TERNARIA, "A", "B")
    loop.report_outcome(decision.id, correct=False, per_engine_correct={"otro": True})

    assert _bin_de(loop, "otro") == (2.0, 1.0), "la verdad explicita manda"
