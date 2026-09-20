"""Demo sintética end-to-end de DualLoop.

Simula un mundo de juguete con tres motores heterogéneos de fiabilidad y
coste distintos, y muestra empíricamente que:

1. La cascada resuelve la mayoría de los casos con el motor barato y solo
   escalonda al caro cuando hace falta (ahorro de coste).
2. La calibración de confianza mejora con la experiencia (el motor barato
   parte deliberadamente MAL calibrado -sobreconfiado en casos difíciles-
   para que el efecto de corregirlo sea visible).
3. La precisión de las respuestas aceptadas se mantiene estable o mejora
   con el tiempo, sin ningún reentrenamiento manual: todo pasa a través de
   `report_outcome()`.

IMPORTANTE: esto es un benchmark SINTÉTICO de referencia, no una validación
en producción. Sirve para verificar que el mecanismo de recalibración
funciona como está diseñado, no como prueba de que resuelve un dominio real.
No requiere red ni credenciales: los tres motores son simulaciones locales.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualloop import DualLoop, InMemoryStore, Question  # noqa: E402
from dualloop.engines.base import BaseEngine  # noqa: E402

LABELS = ["A", "B"]
N_DECISIONS = 800
WINDOW = 50
SEED = 7


class SimFastEngine(BaseEngine):
    """Simula un clasificador tipado tipo Jev: barato, rápido, pero
    deliberadamente SOBRECONFIADO en casos difíciles (para que el efecto de
    la calibración sea visible en las métricas)."""

    name = "jev_sim"
    relative_cost = 0.1

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _decide_raw(self, task_type: str, question: Question):
        true_label = question.state.split("|")[1]
        difficulty = question.state.split("|")[2]
        acc = 0.92 if difficulty == "easy" else 0.55
        correct = self._rng.random() < acc
        value = true_label if correct else next(o for o in LABELS if o != true_label)
        if difficulty == "easy":
            confidence = self._rng.uniform(0.78, 0.98)
        else:
            # sobreconfiado incluso cuando se equivoca: esto es lo que la
            # calibración debe corregir con la experiencia
            confidence = self._rng.uniform(0.65, 0.92)
        return value, confidence, {"sim": "fast", "correct": correct}


class SimSlowEngine(BaseEngine):
    """Simula un LLM de razonamiento: caro, lento, mejor calibrado y más
    fiable en casos difíciles."""

    name = "llm_sim"
    relative_cost = 5.0

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _decide_raw(self, task_type: str, question: Question):
        true_label = question.state.split("|")[1]
        difficulty = question.state.split("|")[2]
        acc = 0.85 if difficulty == "easy" else 0.90
        correct = self._rng.random() < acc
        value = true_label if correct else next(o for o in LABELS if o != true_label)
        confidence = self._rng.uniform(0.7, 0.95) if correct else self._rng.uniform(0.4, 0.7)
        return value, confidence, {"sim": "slow", "correct": correct}


class SimRuleEngine(BaseEngine):
    """Simula una regla de negocio: gratis e instantánea, pero solo cubre
    el subconjunto de casos marcados como 'obvious'."""

    name = "rules_sim"
    relative_cost = 0.01

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _decide_raw(self, task_type: str, question: Question):
        true_label, difficulty, obvious = question.state.split("|")[1:4]
        if obvious != "yes":
            raise ValueError("La regla no cubre este caso (no es obvio)")
        correct = self._rng.random() < 0.98
        value = true_label if correct else next(o for o in LABELS if o != true_label)
        return value, 0.99, {"sim": "rule", "correct": correct}


def make_question(true_label: str, difficulty: str, obvious: str) -> Question:
    # el estado codifica la verdad simulada para que los motores "hagan
    # trampa" de forma controlada; en un caso real el estado sería el
    # contexto de negocio real, no la respuesta
    state = f"caso_sintetico|{true_label}|{difficulty}|{obvious}"
    return Question(
        type="choice",
        instructions="Elige la categoria correcta para este caso sintetico.",
        state=state,
        criteria={"A": None, "B": None},
    )


def windowed(values: list[float], window: int) -> list[float]:
    return [statistics.mean(values[i : i + window]) for i in range(0, len(values), window)]


def main() -> None:
    rng = random.Random(SEED)
    loop = DualLoop(
        engines=[SimFastEngine(rng), SimSlowEngine(rng), SimRuleEngine(rng)],
        store=InMemoryStore(),
        seed=SEED,
        default_threshold=0.75,
    )

    accuracy_series: list[float] = []
    n_engines_series: list[int] = []
    calib_error_series: list[float] = []

    for i in range(N_DECISIONS):
        true_label = rng.choice(LABELS)
        difficulty = "easy" if rng.random() < 0.7 else "hard"
        obvious = "yes" if rng.random() < 0.15 else "no"
        question = make_question(true_label, difficulty, obvious)

        decision = loop.decide("caso_sintetico", question)
        correct = decision.chosen_value == true_label

        per_engine_correct = {v.engine_name: (v.value == true_label) for v in decision.votes}
        loop.report_outcome(decision.id, correct=correct, per_engine_correct=per_engine_correct)

        accuracy_series.append(1.0 if correct else 0.0)
        n_engines_series.append(len(decision.votes))
        calib_error_series.append(abs(decision.chosen_confidence - (1.0 if correct else 0.0)))

    acc_w = windowed(accuracy_series, WINDOW)
    n_w = windowed([float(n) for n in n_engines_series], WINDOW)
    calib_w = windowed(calib_error_series, WINDOW)

    print(f"DualLoop - demo sintetica ({N_DECISIONS} decisiones, ventana={WINDOW})\n")
    print(f"{'ventana':>8} | {'precision':>9} | {'motores/decision':>16} | {'error calibracion':>18}")
    print("-" * 62)
    for idx in (0, len(acc_w) // 2, len(acc_w) - 1):
        print(f"{idx * WINDOW:>8} | {acc_w[idx]:>9.3f} | {n_w[idx]:>16.2f} | {calib_w[idx]:>18.3f}")

    print("\nComparacion primera ventana vs ultima ventana:")
    print(f"  precision:            {acc_w[0]:.3f} -> {acc_w[-1]:.3f}")
    print(f"  motores por decision: {n_w[0]:.2f} -> {n_w[-1]:.2f}  (mas bajo = mas barato)")
    print(f"  error de calibracion: {calib_w[0]:.3f} -> {calib_w[-1]:.3f}  (mas bajo = mejor calibrado)")

    # ---- auditoría de una decisión concreta ----
    last_decision_id = decision.id
    explanation = loop.explain(last_decision_id)
    print(f"\nAuditoria de la ultima decision ({last_decision_id[:8]}...):")
    print(f"  task_type: {explanation['decision'].task_type}")
    print(f"  motor elegido: {explanation['decision'].chosen_engine}")
    print(f"  confianza calibrada: {explanation['decision'].chosen_confidence:.3f}")
    print(f"  umbral usado: {explanation['decision'].escalation_threshold_used:.3f}")
    print(f"  fiabilidad aprendida por motor: {explanation['reliability_at_decision_time']}")
    print(f"  outcomes registrados: {len(explanation['outcomes'])}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        x = [i * WINDOW for i in range(len(acc_w))]
        axes[0].plot(x, acc_w)
        axes[0].set_ylabel("precision")
        axes[1].plot(x, n_w)
        axes[1].set_ylabel("motores/decision")
        axes[2].plot(x, calib_w)
        axes[2].set_ylabel("error calibracion")
        axes[2].set_xlabel("decision #")
        fig.suptitle("DualLoop - recalibracion online sobre datos sinteticos")
        out_path = Path(__file__).with_name("demo_synthetic_result.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\nGrafico guardado en {out_path}")
    except ImportError:
        print("\n(instala el extra 'demo' -> pip install '.[demo]' para generar el grafico)")


if __name__ == "__main__":
    main()
