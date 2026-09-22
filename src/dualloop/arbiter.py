"""El árbitro: orquesta la cascada de consulta entre motores heterogéneos.

Flujo por cada decisión:
1. Ordena los motores disponibles para este task_type por fiabilidad
   aprendida (ReliabilityBandit.rank), con sesgo inicial a favor de los
   baratos mientras no hay evidencia.
2. Consulta el primero, calibra su confianza cruda (ConfidenceCalibrator).
3. Si la confianza calibrada alcanza el umbral adaptativo del task_type,
   acepta esa respuesta y NO sigue escalando (ahorro de coste).
4. Si no, pasa al siguiente motor en el orden aprendido, hasta agotar la
   lista o el límite `max_engines_per_decision`.
5. Se queda con el voto de mayor confianza calibrada entre todos los
   consultados, y marca `disagreement=True` si los motores no coincidieron.

Todo el proceso queda registrado en el `Decision` resultante — qué motores
se consultaron, en qué orden, sus salidas crudas y calibradas, y el umbral
usado — para que `DualLoop.explain()` pueda reconstruir por qué se decidió
lo que se decidió.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bandit import AdaptiveThreshold, ReliabilityBandit
from .calibration import ConfidenceCalibrator
from .engines.base import BaseEngine
from .types import Decision, EngineOutput, Question, Vote


@dataclass
class ArbitrationResult:
    decision: Decision
    consulted_engines: list[str]


class Arbiter:
    def __init__(
        self,
        engines: dict[str, BaseEngine],
        calibrators: dict[tuple[str, str], ConfidenceCalibrator],
        bandit: ReliabilityBandit,
        threshold: AdaptiveThreshold,
        *,
        max_engines_per_decision: int | None = None,
        abstain_below_threshold: bool = False,
    ) -> None:
        self._engines = engines
        self._calibrators = calibrators
        self._bandit = bandit
        self._threshold = threshold
        self._max_engines = max_engines_per_decision or len(engines)
        self._abstain_below_threshold = abstain_below_threshold

    def _calibrator_for(self, task_type: str, engine_name: str) -> ConfidenceCalibrator:
        key = (task_type, engine_name)
        if key not in self._calibrators:
            self._calibrators[key] = ConfidenceCalibrator()
        return self._calibrators[key]

    def decide(
        self,
        task_type: str,
        question: Question,
        engine_subset: list[str] | None = None,
    ) -> ArbitrationResult:
        names = engine_subset or list(self._engines.keys())
        if not names:
            raise ValueError("No hay motores disponibles para arbitrar esta decisión")

        order = self._bandit.rank(task_type, names)
        threshold = self._threshold.get(task_type)

        votes: list[Vote] = []
        consulted: list[str] = []
        best: Vote | None = None

        for engine_name in order[: self._max_engines]:
            engine = self._engines[engine_name]
            output: EngineOutput = engine.decide(task_type, question)
            consulted.append(engine_name)
            if not output.ok:
                continue

            calibrator = self._calibrator_for(task_type, engine_name)
            calibrated = calibrator.calibrate(output.raw_confidence)
            vote = Vote(
                engine_name=engine_name,
                value=output.value,
                raw_confidence=output.raw_confidence,
                calibrated_confidence=calibrated,
                latency_ms=output.latency_ms,
                raw=output.raw,
            )
            votes.append(vote)
            if best is None or vote.calibrated_confidence > best.calibrated_confidence:
                best = vote
            if calibrated >= threshold:
                break  # confianza suficiente: no seguimos escalando

        if best is None:
            raise RuntimeError(
                f"Ningun motor pudo responder para task_type={task_type!r} "
                f"(motores consultados: {consulted})"
            )

        disagreement = len({v.value for v in votes}) > 1
        # Si se agoto la cascada sin que nadie alcanzara el umbral, el
        # sistema no tiene una respuesta en la que confie. Marcarlo es lo
        # honesto: decidir igualmente convierte el umbral en decorativo.
        abstained = (
            self._abstain_below_threshold and best.calibrated_confidence < threshold
        )

        decision = Decision(
            id=Decision.new_id(),
            task_type=task_type,
            question=question,
            votes=votes,
            chosen_engine=best.engine_name,
            chosen_value=best.value,
            chosen_confidence=best.calibrated_confidence,
            escalation_threshold_used=threshold,
            disagreement=disagreement,
            abstained=abstained,
        )
        return ArbitrationResult(decision=decision, consulted_engines=consulted)
