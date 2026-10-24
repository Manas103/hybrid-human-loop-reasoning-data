"""Simulated human review: a noisy labeler standing in for a paid annotator.

Each review flips the true defective/clean label with probability
`error_rate`, independently per trace, from a seeded RNG. This is stated as
simulated on the resume itself, not hidden in a footnote (see README).
"""
from __future__ import annotations

import random


def simulate_review(true_is_defective: bool, rng: random.Random, error_rate: float = 0.07) -> bool:
    if rng.random() < error_rate:
        return not true_is_defective
    return true_is_defective
