"""Online confidence calibration, per engine and per task type.

The raw confidence of a typed classifier, the self-reported confidence of
an LLM and the fixed certainty of a rule are NOT comparable without
calibration -- exactly the problem the prior research found only half
formalised (uncertainty routers between LLM variants such as CP-Router,
arXiv:2505.19970, or LEC for selective prediction, arXiv:2512.01556) but
never across engines of genuinely different kinds.

The technique used here is online Beta-Bernoulli histogram binning: no
gradients, no retraining, a closed-form Bayesian update in O(1) per
observation. It is deliberately simple so that anyone can audit it by
reading the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Bin:
    alpha: float = 1.0  # Beta(1,1) prior = uniform (no initial bias)
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
    """Calibrates the raw confidence of ONE engine for ONE task_type.

    On a cold start -- few observations in the relevant bin -- the
    calibrated confidence stays close to the raw one, because there is no
    evidence to correct it with. As real outcomes accumulate it shifts
    towards the empirical accuracy observed for that confidence range.
    That is literally "recalibrating itself": each `update()` is the only
    action required, with no batch job and no manual intervention.
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
        # Weight given to real evidence: grows with n, capped at 0.9 so
        # the engine's own raw confidence is never ignored entirely.
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
