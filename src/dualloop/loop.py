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
from .types import Decision, Outcome, Question, Vote


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
        abstain_below_threshold: bool = False,
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
            abstain_below_threshold=abstain_below_threshold,
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
        inferencia por defecto, que solo tiene certeza sobre el motor
        elegido y sobre los que coincidieron con él; ver
        `_infer_vote_correct`, que deja sin actualizar los votos cuya
        verdad no se puede deducir.
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

    @staticmethod
    def _infer_vote_correct(
        decision: Decision, vote: Vote, correct: bool
    ) -> Optional[bool]:
        """Infiere si un voto acerto, a partir del outcome de la elegida.

        El motor elegido recibe la senal exacta. Para los demas:

        - si la elegida acerto, quien discrepo fallo;
        - si la elegida fallo, quien coincidio fallo;
        - si la elegida fallo y el motor discrepo, depende de cuantas
          respuestas posibles habia. Con dos (``choice`` binario o
          ``noul``) el discrepante acerto necesariamente. Con mas de dos,
          saber que la elegida era falsa NO identifica cual era la buena.

        Devuelve ``None`` cuando la verdad del voto es desconocida; el
        llamante no actualiza nada en ese caso. Es deliberadamente
        conservador: alimentar al calibrador con una etiqueta inventada
        sesga el sistema hacia el consenso, que es exactamente lo que un
        arbitro entre motores heterogeneos no debe hacer. Si prefieres mas
        senal a cambio de suponer uniformidad entre las opciones
        restantes, esta es la unica linea que hay que cambiar (por ejemplo
        devolviendo una actualizacion ponderada por 1/(n_options - 1)).
        """
        if vote.engine_name == decision.chosen_engine:
            return correct

        agreed = vote.value == decision.chosen_value
        if correct:
            return agreed
        if agreed:
            return False

        n_options = (
            2
            if decision.question.type == "noul"
            else len(decision.question.criteria or ())
        )
        return True if n_options == 2 else None

    def _recalibrate(
        self,
        decision: Decision,
        correct: Optional[bool],
        per_engine_correct: Optional[dict[str, bool]],
    ) -> None:
        for vote in decision.votes:
            if per_engine_correct and vote.engine_name in per_engine_correct:
                vote_correct: Optional[bool] = per_engine_correct[vote.engine_name]
            elif correct is not None:
                vote_correct = self._infer_vote_correct(decision, vote, correct)
            else:
                continue
            if vote_correct is None:
                continue
            calibrator = self._calibrators.setdefault(
                (decision.task_type, vote.engine_name), ConfidenceCalibrator()
            )
            calibrator.update(vote.raw_confidence, vote_correct)

        if correct is not None:
            self._bandit.update(decision.task_type, decision.chosen_engine, correct)
            # El umbral persigue la tasa de error de lo ACEPTADO. Una
            # decision en la que el loop se abstuvo no fue aceptada, asi
            # que alimentarla aqui rompe la semantica del umbral: lo
            # subiria por un error que el sistema ya habia senalado como
            # dudoso. Los calibradores y el bandit si aprenden de ella.
            if not decision.abstained:
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
        # Los calibradores se guardan en un unico blob, que se reescribe
        # entero en cada outcome: el coste es O(pares task_type x motor).
        # Es deliberado. Guardar un blob por calibrador obligaria a
        # mantener ademas un indice, porque el protocolo `Store` no expone
        # forma de enumerar claves, y el ahorro solo se nota con cientos de
        # task_types. Si algun dia lo hace, ese es el cambio: un blob
        # "calibrators::index" con las claves, mas uno por calibrador.
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
