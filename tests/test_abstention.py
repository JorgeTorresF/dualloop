"""Abstencion: que pasa cuando ningun motor alcanza el umbral.

Sin `abstain_below_threshold` el loop acepta igualmente el mejor voto, por
debajo del umbral: el umbral queda registrado pero no actua como suelo.
Para un gate donde la respuesta correcta puede ser "no se, que lo mire una
persona", eso es justo lo que no se quiere.
"""

from dualloop import DualLoop, InMemoryStore, Question
from dualloop.engines.base import BaseEngine

QUESTION = Question(
    type="choice", instructions="elige", state="ctx",
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


def test_por_defecto_no_se_abstiene_y_decide_igualmente():
    decision = _loop(abstain=False).decide("t", QUESTION)
    assert decision.abstained is False
    assert decision.chosen_confidence < decision.escalation_threshold_used
    assert decision.chosen_value in ("A", "B"), "sigue decidiendo, comportamiento previo"


def test_se_abstiene_cuando_nadie_alcanza_el_umbral():
    decision = _loop(abstain=True).decide("t", QUESTION)
    assert decision.abstained is True
    assert decision.chosen_confidence < decision.escalation_threshold_used
    assert decision.chosen_value in ("A", "B"), "el mejor voto sigue disponible para el humano"


def test_no_se_abstiene_si_alguien_supera_el_umbral():
    decision = _loop(abstain=True, confidence=0.95, threshold=0.7).decide("t", QUESTION)
    assert decision.abstained is False


def test_una_abstencion_no_mueve_el_umbral_pero_si_el_bandit():
    # El umbral persigue la tasa de error de lo ACEPTADO. Si el sistema se
    # abstuvo, ese caso no fue aceptado y no debe endurecer el umbral.
    loop = _loop(abstain=True)
    antes = loop._threshold.get("t")

    decision = loop.decide("t", QUESTION)
    assert decision.abstained is True
    loop.report_outcome(decision.id, correct=False)

    assert loop._threshold.get("t") == antes, "una abstencion no mueve el umbral"
    assert loop._bandit.mean("t", decision.chosen_engine) < 1.0, "el bandit si aprende"


def test_sin_abstencion_el_mismo_fallo_si_endurece_el_umbral():
    # Control: mismo escenario con la abstencion apagada.
    loop = _loop(abstain=False)
    antes = loop._threshold.get("t")

    decision = loop.decide("t", QUESTION)
    loop.report_outcome(decision.id, correct=False)

    assert loop._threshold.get("t") > antes, "un fallo aceptado si endurece el umbral"
