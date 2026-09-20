"""Calibración online de confianza, por motor y por tipo de tarea.

La confianza cruda de un clasificador tipado, la confianza autoreportada de
un LLM y la certeza fija de una regla NO son comparables entre sí sin
calibrar — es exactamente el problema que la investigación previa encontró
formalizado solo a medias (routers de incertidumbre entre variantes de LLM
como CP-Router, arXiv:2505.19970, o LEC para predicción selectiva,
arXiv:2512.01556) pero nunca entre motores de naturaleza heterogénea.

La técnica usada aquí es histogram binning Beta-Bernoulli online: sin
gradientes, sin reentrenamiento, actualización bayesiana cerrada en O(1)
por observación. Es deliberadamente simple para que cualquiera pueda
auditarla leyendo el código.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Bin:
    alpha: float = 1.0  # prior Beta(1,1) = uniforme (sin sesgo inicial)
    beta: float = 1.0

    @property
    def n(self) -> float:
        return self.alpha + self.beta - 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, correct: bool) -> None:
        if correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class ConfidenceCalibrator:
    """Calibra la confianza cruda de UN motor para UN task_type.

    En frío (pocas observaciones en el bin correspondiente), la confianza
    calibrada se acerca a la cruda (no hay evidencia para corregirla). A
    medida que se acumulan outcomes reales, se desplaza hacia la tasa de
    acierto empírica observada para ese rango de confianza — esto es
    literalmente "recalibrarse solo": cada `update()` es la única acción
    necesaria, sin batch ni intervención manual.
    """

    def __init__(self, n_bins: int = 10, cold_start_n: float = 5.0) -> None:
        self.n_bins = n_bins
        self.cold_start_n = cold_start_n
        self._bins: dict[int, _Bin] = {i: _Bin() for i in range(n_bins)}

    def _bin_index(self, raw_confidence: float) -> int:
        idx = int(raw_confidence * self.n_bins)
        return min(max(idx, 0), self.n_bins - 1)

    def calibrate(self, raw_confidence: float) -> float:
        raw_confidence = min(max(raw_confidence, 0.0), 1.0)
        b = self._bins[self._bin_index(raw_confidence)]
        # peso de la evidencia real: crece con n, tope en 0.9 para nunca
        # ignorar del todo la confianza cruda reportada por el motor
        w = min(b.n / (b.n + self.cold_start_n), 0.9)
        return (1 - w) * raw_confidence + w * b.mean

    def update(self, raw_confidence: float, correct: bool) -> None:
        raw_confidence = min(max(raw_confidence, 0.0), 1.0)
        self._bins[self._bin_index(raw_confidence)].update(correct)

    def state_dict(self) -> dict:
        return {str(i): (b.alpha, b.beta) for i, b in self._bins.items()}

    def load_state_dict(self, state: dict) -> None:
        for i, (a, b) in state.items():
            self._bins[int(i)] = _Bin(alpha=a, beta=b)
