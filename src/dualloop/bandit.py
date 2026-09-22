"""Learned arbitration across heterogeneous engines: which engine to
consult first (ReliabilityBandit) and when to stop escalating
(AdaptiveThreshold).

No reviewed commercial router (RouteLLM, OpenRouter, Martian, Not Diamond)
nor research meta-controller (Meta-Reasoner, arXiv:2502.19918; AAMC)
arbitrates between engines of genuinely different kinds -- they all route
between LLM variants. Here "engine" is an opaque label: it can be an LLM, a
typed classifier or a rule. The bandit does not need to know the
difference, only whether it was right.
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
    """Beta-Bernoulli Thompson sampling per (task_type, engine).

    Used only to ORDER the cascade -- which engine to try first -- never to
    decide the final answer; that is what each vote's calibrated confidence
    does. On a cold start, with no observations at all, a mild optimistic
    bias favours the cheaper engines (`cost_by_engine`) so the cascade
    begins by exploring the cheap engine rather than at random. The bias
    dilutes quickly once real outcomes arrive.
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
    """Per-task_type acceptance threshold, adjusted online by constant-step
    stochastic approximation.

    Every time an answer is accepted and its real outcome becomes known,
    the threshold moves towards the point where the error rate of accepted
    answers equals `target_error_rate`: if the observed error exceeds the
    target it rises (stricter, escalates more); if it is lower it falls a
    little (less needless escalation, cheaper).

    **On the step.** `lr` is constant, not decreasing, so this is
    **constant-step** stochastic approximation and does not converge in the
    Robbins-Monro sense: it oscillates around the equilibrium. That is
    deliberate -- with a decreasing step the threshold would freeze, and
    engine reliability is expected to drift over time (you change model,
    the jev server is updated, the domain shifts).

    **On the bounds.** `lo` is the system's real floor: when the engines
    perform better than `target_error_rate`, the threshold falls until it
    rests on `lo` and stays there, so past that point it is `lo` -- not
    `target_error_rate` -- that decides what gets accepted. That is why the
    default floor is 0.8 rather than 0.5: accepting an answer at 50%
    calibrated confidence is rarely what anyone wants, and with a low `lo`
    abstention almost never fires. `default` starts above the floor so that
    downward adaptation has room; with `default == lo` the threshold could
    only ever rise.

    The equilibrium is nonetheless the right one: in steady state the error
    rate of accepted answers tends to `target_error_rate`, because
    `p*lr*(1-t) = (1-p)*lr*t` holds exactly at `p = t`.

    With the defaults (`lr=0.01`, `target_error_rate=0.05`) a wrong outcome
    raises the threshold by 0.0095 and a correct one lowers it by 0.0005.
    Over the default range `[0.8, 0.97]` that is 5.6% of the possible
    travel per error; over a wider range such as `[0.5, 0.97]` it would be
    2%. The previous `lr=0.05` moved it about 10% of that wider range per
    error -- a single isolated mistake shifted the threshold too far. The
    price of lowering it is needing roughly five times as many
    observations to travel the same distance.
    """

    def __init__(
        self,
        default: float = 0.85,
        lr: float = 0.01,
        target_error_rate: float = 0.05,
        lo: float = 0.8,
        hi: float = 0.97,
    ) -> None:
        # The floor overrides the starting point: asking for a `default`
        # outside [lo, hi] is not a caller error, it is a policy the bounds
        # correct. Without this, a default below the floor would give an
        # initial threshold that no later update could ever return to --
        # incoherent, and silently so.
        self.default = min(max(default, lo), hi)
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
