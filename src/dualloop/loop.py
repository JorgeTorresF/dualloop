"""Punto de entrada único de la librería.

`DualLoop` combina motores heterogéneos (LLM razonador, clasificador
tipado tipo Jev, reglas...) bajo un árbitro aprendido y auditable, y cierra
el bucle decisión -> resultado -> recalibración a través de
`report_outcome()`, sin reentrenar ningún modelo: cada llamada actualiza
tres estadísticos online (calibración por motor, fiabilidad por motor,
umbral de aceptación por tipo de tarea) con una actualización bayesiana
cerrada en O(1).
"""

from __future__ import annotations

from typing import Optional

from .arbiter import Arbiter
from .bandit import AdaptiveThreshold, ReliabilityBandit
from .calibration import ConfidenceCalibrator
from .engines.base import BaseEngine
from .heuristics import OutcomeHeuristic
from .store import InMemoryStore, Store
from .types import Decision, Outcome, Question


class DualLoop:
    def __init__(
        self,
        engines: list[BaseEngine],
        *,
        store: Optional[Store] = None,
        seed: Optional[int] = None,
        default_threshold: float = 0.7,
        max_engines_per_decision: Optional[int] = None,
        heuristics: Optional[list[OutcomeHeuristic]] = None,
    ) -> None:
        if not engines:
            raise ValueError("DualLoop necesita al menos un motor")

        self.store = store or InMemoryStore()
        by_name = {e.name: e for e in engines}
        if len(by_name) != len(engines):
            raise ValueError("Los nombres de los motores deben ser unicos")
        # orden inicial por coste ascendente: informa el prior optimista
        # del bandit (ver ReliabilityBandit), no determina el resultado
        self._engines_by_name = dict(sorted(by_name.items(), key=lambda kv: kv[1].relative_cost))

        self._calibrators: dict[tuple[str, str], ConfidenceCalibrator] = {}
        cost_by_engine = {name: e.relative_cost for name, e in self._engines_by_name.items()}
        self._bandit = ReliabilityBandit(cost_by_engine=cost_by_engine, seed=seed)
        self._threshold = AdaptiveThreshold(default=default_threshold)
        self._arbiter = Arbiter(
            self._engines_by_name,
            self._calibrators,
            self._bandit,
            self._threshold,
            max_engines_per_decision=max_engines_per_decision,
        )
        self._heuristics = heuristics or []
        self._load_state()

    # ---------------------------------------------------------------- decidir
    def decide(
        self,
        task_type: str,
        question: Question,
        engine_subset: Optional[list[str]] = None,
    ) -> Decision:
        result = self._arbiter.decide(task_type, question, engine_subset=engine_subset)
        self.store.save_decision(result.decision)
        return result.decision

    # ------------------------------------------------------- cerrar el bucle
    def report_outcome(
        self,
        decision_id: str,
        correct: Optional[bool] = None,
        *,
        reward: Optional[float] = None,
        note: str = "",
        ground_truth: object = None,
        source: str = "explicit",
        per_engine_correct: Optional[dict[str, bool]] = None,
    ) -> Outcome:
        """Reporta lo que pasó de verdad tras una decisión.

        `correct=None` registra el outcome como auditoría sin recalibrar
        (resultado ambiguo o aún desconocido). `per_engine_correct` permite
        dar la verdad por motor cuando se conoce (p.ej. en un set de
        evaluación offline) para una recalibración más precisa que la
        aproximación por defecto (solo el motor elegido recibe la señal
        exacta; los demás se aproximan por si coincidieron con la
        respuesta elegida).
        """
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(f"No existe una decision con id={decision_id!r}")

        outcome = Outcome(
            decision_id=decision_id,
            correct=correct,
            reward=reward,
            note=note,
            ground_truth=ground_truth,
            source=source,
        )
        self.store.save_outcome(outcome)

        if correct is not None or per_engine_correct:
            self._recalibrate(decision, correct, per_engine_correct)

        return outcome

    def try_infer_outcomes(self, decision_id: str, context: dict) -> list[Outcome]:
        """Ejecuta los heuristicos opcionales registrados sobre una
        decision. No se llama automaticamente: quien integra la libreria
        decide cuando invocarlo."""
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(f"No existe una decision con id={decision_id!r}")

        inferred: list[Outcome] = []
        for heuristic in self._heuristics:
            outcome = heuristic.try_infer(decision, context)
            if outcome is not None:
                self.store.save_outcome(outcome)
                if outcome.correct is not None:
                    self._recalibrate(decision, outcome.correct, None)
                inferred.append(outcome)
        return inferred

    def _recalibrate(
        self,
        decision: Decision,
        correct: Optional[bool],
        per_engine_correct: Optional[dict[str, bool]],
    ) -> None:
        for vote in decision.votes:
            if per_engine_correct and vote.engine_name in per_engine_correct:
                vote_correct = per_engine_correct[vote.engine_name]
            elif correct is not None:
                vote_correct = (
                    correct
                    if vote.engine_name == decision.chosen_engine
                    else (vote.value == decision.chosen_value and correct)
                )
            else:
                continue
            calibrator = self._calibrators.setdefault(
                (decision.task_type, vote.engine_name), ConfidenceCalibrator()
            )
            calibrator.update(vote.raw_confidence, vote_correct)

        if correct is not None:
            self._bandit.update(decision.task_type, decision.chosen_engine, correct)
            self._threshold.update_on_accepted_outcome(decision.task_type, correct)

        self._save_state()

    # ----------------------------------------------------------- auditoría
    def explain(self, decision_id: str) -> dict:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(f"No existe una decision con id={decision_id!r}")
        outcomes = self.store.get_outcomes(decision_id)
        return {
            "decision": decision,
            "outcomes": outcomes,
            "reliability_at_decision_time": {
                v.engine_name: self._bandit.mean(decision.task_type, v.engine_name)
                for v in decision.votes
            },
            "current_threshold": self._threshold.get(decision.task_type),
        }

    # --------------------------------------------- persistencia del estado
    def _save_state(self) -> None:
        self.store.save_state_blob("bandit", self._bandit.state_dict())
        self.store.save_state_blob("threshold", self._threshold.state_dict())
        self.store.save_state_blob(
            "calibrators",
            {f"{t}||{e}": c.state_dict() for (t, e), c in self._calibrators.items()},
        )

    def _load_state(self) -> None:
        bandit_state = self.store.load_state_blob("bandit")
        if bandit_state:
            self._bandit.load_state_dict(bandit_state)

        threshold_state = self.store.load_state_blob("threshold")
        if threshold_state:
            self._threshold.load_state_dict(threshold_state)

        calibrators_state = self.store.load_state_blob("calibrators")
        if calibrators_state:
            for key, state in calibrators_state.items():
                t, e = key.split("||", 1)
                calibrator = ConfidenceCalibrator()
                calibrator.load_state_dict(state)
                self._calibrators[(t, e)] = calibrator
