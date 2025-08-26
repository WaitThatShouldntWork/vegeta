from __future__ import annotations

import random
from typing import Literal


def noisy_yes_no(p_true: float = 0.7, error_rate: float = 0.1, seed: int | None = None) -> Literal["yes", "no"]:
    rng = random.Random(seed)
    truth = rng.random() < p_true
    if rng.random() < error_rate:
        truth = not truth
    return "yes" if truth else "no"


