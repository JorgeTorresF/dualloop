from dualloop.bandit import AdaptiveThreshold, ReliabilityBandit


def test_cheap_engine_biased_first_in_cold_start():
    bandit = ReliabilityBandit(cost_by_engine={"cheap": 0.1, "expensive": 5.0}, seed=42)
    # sin ninguna observacion, en agregado sobre muchos samples el barato
    # deberia salir primero con mucha mas frecuencia que el caro
    first_counts = {"cheap": 0, "expensive": 0}
    for _ in range(200):
        order = bandit.rank("t", ["cheap", "expensive"])
        first_counts[order[0]] += 1
    assert first_counts["cheap"] > first_counts["expensive"]


def test_bandit_learns_reliability_from_updates():
    bandit = ReliabilityBandit(seed=1)
    for _ in range(30):
        bandit.update("t", "good", correct=True)
        bandit.update("t", "bad", correct=False)
    assert bandit.mean("t", "good") > bandit.mean("t", "bad")


def test_bandit_state_roundtrip():
    bandit = ReliabilityBandit(seed=1)
    bandit.update("t", "a", correct=True)
    bandit.update("t", "a", correct=True)
    state = bandit.state_dict()

    restored = ReliabilityBandit(seed=1)
    restored.load_state_dict(state)
    assert restored.mean("t", "a") == bandit.mean("t", "a")


def test_adaptive_threshold_rises_on_errors_and_falls_on_success():
    # `lo` explicito: este test prueba la MECANICA del ajuste (sube con
    # fallos, baja con aciertos), no la politica. Con el suelo por defecto
    # (0.8) un arranque en 0.7 quedaria recortado y no podria bajar.
    threshold = AdaptiveThreshold(default=0.7, lr=0.1, target_error_rate=0.05, lo=0.5)
    base = threshold.get("t")
    threshold.update_on_accepted_outcome("t", correct=False)
    assert threshold.get("t") > base

    threshold2 = AdaptiveThreshold(default=0.7, lr=0.1, target_error_rate=0.05, lo=0.5)
    threshold2.update_on_accepted_outcome("t", correct=True)
    assert threshold2.get("t") < base


def test_adaptive_threshold_respects_bounds():
    threshold = AdaptiveThreshold(default=0.95, lr=1.0, lo=0.5, hi=0.97)
    for _ in range(20):
        threshold.update_on_accepted_outcome("t", correct=False)
    assert threshold.get("t") <= 0.97


def test_el_suelo_recorta_un_default_mas_bajo():
    # Pedir un arranque por debajo del suelo no es un error del llamante:
    # el suelo es politica y recorta. Lo contrario daria un umbral inicial
    # al que ninguna actualizacion posterior podria volver.
    threshold = AdaptiveThreshold(default=0.5, lo=0.8)
    assert threshold.get("t") == 0.8

    threshold_alto = AdaptiveThreshold(default=0.99, hi=0.97)
    assert threshold_alto.get("t") == 0.97


def test_suelo_por_defecto_es_08():
    # Con motores mejores que target_error_rate el umbral baja hasta
    # pegarse al suelo, asi que `lo` es quien decide de verdad que se
    # acepta. 0.8 y no 0.5: aceptar con un 50% de confianza calibrada
    # rara vez es lo que se quiere.
    threshold = AdaptiveThreshold()
    assert threshold.lo == 0.8
    for _ in range(500):
        threshold.update_on_accepted_outcome("t", correct=True)
    assert threshold.get("t") == 0.8, "baja hasta el suelo y se queda ahi"
