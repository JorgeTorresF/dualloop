from dualloop.bandit import AdaptiveThreshold, ReliabilityBandit


def test_cheap_engine_biased_first_in_cold_start():
    bandit = ReliabilityBandit(cost_by_engine={"cheap": 0.1, "expensive": 5.0}, seed=42)
    # With no observations at all, aggregated over many samples the cheap
    # engine should come first far more often than the expensive one.
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
    # Explicit `lo`: this test exercises the MECHANICS of the adjustment
    # (rises on errors, falls on successes), not the policy. Under the
    # default floor (0.8) a start at 0.7 would be clamped and could not fall.
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


def test_the_floor_clamps_a_lower_default():
    # Asking for a start below the floor is not a caller error: the floor
    # is policy and it clamps. Otherwise you would get an initial threshold
    # that no later update could ever return to.
    threshold = AdaptiveThreshold(default=0.5, lo=0.8)
    assert threshold.get("t") == 0.8

    threshold_alto = AdaptiveThreshold(default=0.99, hi=0.97)
    assert threshold_alto.get("t") == 0.97


def test_default_floor_is_08():
    # With engines better than target_error_rate the threshold falls until
    # it rests on the floor, so `lo` is what really decides what gets
    # accepted. 0.8 and not 0.5: accepting at 50% calibrated confidence is
    # rarely what anyone wants.
    threshold = AdaptiveThreshold()
    assert threshold.lo == 0.8
    for _ in range(500):
        threshold.update_on_accepted_outcome("t", correct=True)
    assert threshold.get("t") == 0.8, "falls to the floor and stays there"
