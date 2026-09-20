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
    threshold = AdaptiveThreshold(default=0.7, lr=0.1, target_error_rate=0.05)
    base = threshold.get("t")
    threshold.update_on_accepted_outcome("t", correct=False)
    assert threshold.get("t") > base

    threshold2 = AdaptiveThreshold(default=0.7, lr=0.1, target_error_rate=0.05)
    threshold2.update_on_accepted_outcome("t", correct=True)
    assert threshold2.get("t") < base


def test_adaptive_threshold_respects_bounds():
    threshold = AdaptiveThreshold(default=0.95, lr=1.0, lo=0.5, hi=0.97)
    for _ in range(20):
        threshold.update_on_accepted_outcome("t", correct=False)
    assert threshold.get("t") <= 0.97
