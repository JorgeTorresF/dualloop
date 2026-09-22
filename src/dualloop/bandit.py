"""Arbitraje aprendido entre motores heterogéneos: a qué motor consultar
primero (ReliabilityBandit) y cuándo dejar de escalar (AdaptiveThreshold).

Ningún router comercial revisado (RouteLLM, OpenRouter, Martian, Not
Diamond) ni meta-controlador de investigación (Meta-Reasoner,
arXiv:2502.19918; AAMC) arbitra entre motores de naturaleza distinta —
todos enrutan entre variantes de LLM. Aquí el "motor" es una etiqueta
opaca: puede ser un LLM, un clasificador tipado o una regla; el bandit no
necesita saber la diferencia, solo si acertó o no.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class _Arm:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, correct: bool) -> None:
        if correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class ReliabilityBandit:
    """Thompson sampling Beta-Bernoulli por (task_type, motor).

    Se usa solo para ORDENAR la cascada (qué motor probar primero), no para
    decidir la respuesta final — eso lo hace la confianza calibrada de cada
    voto. En frío, sin ninguna observación, se aplica un sesgo optimista
    suave a favor de los motores más baratos (`cost_by_engine`) para que la
    cascada empiece explorando por el motor barato en vez de al azar; el
    sesgo se diluye rápido en cuanto llegan outcomes reales.
    """

    def __init__(
        self,
        cost_by_engine: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._arms: dict[tuple[str, str], _Arm] = {}
        self._rng = random.Random(seed)
        self._cost_by_engine = cost_by_engine or {}

    def _arm(self, task_type: str, engine_name: str) -> _Arm:
        key = (task_type, engine_name)
        if key not in self._arms:
            cost = self._cost_by_engine.get(engine_name, 1.0)
            bias = min(2.0, 1.0 / max(cost, 0.01))
            self._arms[key] = _Arm(alpha=1.0 + bias, beta=1.0)
        return self._arms[key]

    def rank(self, task_type: str, engine_names: list[str]) -> list[str]:
        sampled = [(name, self._arm(task_type, name).sample(self._rng)) for name in engine_names]
        sampled.sort(key=lambda t: t[1], reverse=True)
        return [name for name, _ in sampled]

    def update(self, task_type: str, engine_name: str, correct: bool) -> None:
        self._arm(task_type, engine_name).update(correct)

    def mean(self, task_type: str, engine_name: str) -> float:
        return self._arm(task_type, engine_name).mean()

    def state_dict(self) -> dict:
        return {f"{t}||{e}": (a.alpha, a.beta) for (t, e), a in self._arms.items()}

    def load_state_dict(self, state: dict) -> None:
        for key, (a, b) in state.items():
            t, e = key.split("||", 1)
            self._arms[(t, e)] = _Arm(alpha=a, beta=b)


class AdaptiveThreshold:
    """Umbral de aceptación por task_type, ajustado online vía aproximación
    estocástica de paso constante.

    Cada vez que se acepta una respuesta y se conoce su outcome real, el
    umbral se desplaza hacia el punto donde la tasa de error de las
    respuestas aceptadas iguala `target_error_rate`: si el error observado
    supera el objetivo, sube (más exigente, escala más); si es menor, baja
    un poco (menos escalamiento innecesario, más barato).

    **Sobre el paso.** `lr` es constante, no decreciente, así que esto es
    aproximación estocástica de **paso constante** y no converge en el
    sentido de Robbins-Monro: oscila alrededor del equilibrio. Es
    deliberado — con paso decreciente el umbral se congelaría, y aquí se
    espera que la fiabilidad de los motores cambie con el tiempo (cambias
    de modelo, el servidor jev se actualiza, el dominio deriva).

    El equilibrio sí es el correcto: en régimen estacionario la tasa de
    error de lo aceptado tiende a `target_error_rate`, porque
    `p·lr·(1−t) = (1−p)·lr·t` se cumple exactamente en `p = t`.

    Con los valores por defecto (`lr=0.01`, `target_error_rate=0.05`) un
    fallo sube el umbral 0,0095 y un acierto lo baja 0,0005; sobre el rango
    útil `[0.5, 0.97]` eso es un 2 % por fallo. Con el `lr=0.05` anterior
    era un 10 % por fallo: un único error aislado movía el umbral
    demasiado. El precio de bajarlo es que hacen falta unas cinco veces más
    observaciones para recorrer la misma distancia.
    """

    def __init__(
        self,
        default: float = 0.7,
        lr: float = 0.01,
        target_error_rate: float = 0.05,
        lo: float = 0.5,
        hi: float = 0.97,
    ) -> None:
        self.default = default
        self.lr = lr
        self.target_error_rate = target_error_rate
        self.lo = lo
        self.hi = hi
        self._thresholds: dict[str, float] = {}

    def get(self, task_type: str) -> float:
        return self._thresholds.get(task_type, self.default)

    def update_on_accepted_outcome(self, task_type: str, correct: bool) -> None:
        current = self.get(task_type)
        error = 0.0 if correct else 1.0
        new = current + self.lr * (error - self.target_error_rate)
        self._thresholds[task_type] = min(max(new, self.lo), self.hi)

    def state_dict(self) -> dict:
        return dict(self._thresholds)

    def load_state_dict(self, state: dict) -> None:
        self._thresholds = {k: float(v) for k, v in state.items()}
