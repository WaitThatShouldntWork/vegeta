from __future__ import annotations

from typing import Dict


def score_answer(confidence: float, risk_weight: float = 1.0) -> Dict[str, float]:
    # Increase EIG slightly for higher confidence and reduce cost to let answers compete
    eig = max(0.0, confidence - 0.5) * 0.15
    cost = (1.0 - confidence) * risk_weight * 0.08
    return {"eig": round(eig, 3), "cost": round(cost, 3), "net": round(eig - cost, 3)}


def score_ask(entropy: float = 1.0, k: int = 12) -> Dict[str, float]:
    # Slightly reduce the attractiveness of asking so that confident answers can win
    eig = min(0.25, entropy * 0.2)
    cost = 0.12
    return {"eig": round(eig, 3), "cost": cost, "net": round(eig - cost, 3)}


def score_search(recency_flag: bool = False) -> Dict[str, float]:
    eig = 0.15 + (0.03 if recency_flag else 0.0)
    cost = 0.15
    return {"eig": round(eig, 3), "cost": cost, "net": round(eig - cost, 3)}


def estimate_entropy_drop_for_ask(current_entropy: float, plausible_outcomes: int = 3) -> float:
    """Crude entropy drop estimator: proportional to log(outcomes)."""
    import math
    drop = min(current_entropy, 0.5 * math.log2(max(2, plausible_outcomes)))
    return float(round(drop, 3))


def expected_entropy_drop_bayes(prior_dist: Dict, slot: str) -> float:
    try:
        from decider.bayes import expected_entropy_drop_for_slot  # local import
        return float(round(expected_entropy_drop_for_slot(prior_dist, slot), 3))
    except Exception:
        return 0.0


