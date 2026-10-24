import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_sim import simulate_review


def test_deterministic_given_seeded_rng():
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    results_a = [simulate_review(True, rng_a) for _ in range(20)]
    results_b = [simulate_review(True, rng_b) for _ in range(20)]
    assert results_a == results_b


def test_error_rate_zero_is_always_accurate():
    rng = random.Random(1)
    for _ in range(50):
        assert simulate_review(True, rng, error_rate=0.0) is True
        assert simulate_review(False, rng, error_rate=0.0) is False


def test_error_rate_one_always_flips():
    rng = random.Random(1)
    for _ in range(50):
        assert simulate_review(True, rng, error_rate=1.0) is False
        assert simulate_review(False, rng, error_rate=1.0) is True


def test_approximate_error_rate_over_many_trials():
    rng = random.Random(7)
    flips = sum(1 for _ in range(5000) if simulate_review(True, rng, error_rate=0.07) is False)
    rate = flips / 5000
    assert 0.04 < rate < 0.10
