from __future__ import annotations

import math
from typing import Dict, Tuple, Any, Optional

# Latent state v = (risk_class, kev, exp)
# risk_class in {"low","med","high"}; kev in {0,1}; exp in {0,1}
State = Tuple[str, int, int]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _risk_dist(risk_scalar: float) -> Dict[str, float]:
    s = _clamp01(risk_scalar)
    # Triangular weights around centers 0.17 (low), 0.5 (med), 0.83 (high)
    centers = {"low": 0.17, "med": 0.5, "high": 0.83}
    weights: Dict[str, float] = {}
    for k, c in centers.items():
        d = abs(s - c)
        w = _clamp01(1.0 - 3.0 * d)  # linear falloff
        weights[k] = w
    z = sum(weights.values()) or 1.0
    return {k: v / z for k, v in weights.items()}


def build_prior_from_signals(
    signals: Dict[str, Any],
    resolved_slots: Optional[Dict[str, Any]] = None,
) -> Dict[State, float]:
    cvss = signals.get("cvss")
    epss = signals.get("epss")
    kev_flag = bool(signals.get("kev_flag", False))
    cvss_norm = (float(cvss) / 10.0) if isinstance(cvss, (int, float)) else 0.5
    epss_scalar = float(epss) if isinstance(epss, (int, float)) else 0.3
    risk_scalar = _clamp01(0.5 * cvss_norm + 0.5 * epss_scalar)
    risk_p = _risk_dist(risk_scalar)

    kev_p1 = 0.9 if kev_flag else 0.1
    kev_p = {1: kev_p1, 0: 1.0 - kev_p1}

    if resolved_slots and "internet_exposed" in resolved_slots:
        exp_true = bool(resolved_slots["internet_exposed"])
        exp_p = {1: 0.9 if exp_true else 0.1, 0: 0.1 if exp_true else 0.9}
    else:
        exp_p = {1: 0.5, 0: 0.5}

    prior: Dict[State, float] = {}
    for r, pr in risk_p.items():
        for k, pk in kev_p.items():
            for e, pe in exp_p.items():
                prior[(r, k, e)] = pr * pk * pe
    # Normalize (defensive)
    z = sum(prior.values()) or 1.0
    return {s: p / z for s, p in prior.items()}


def entropy(dist: Dict[State, float]) -> float:
    h = 0.0
    for p in dist.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return float(h)


def _likelihood(slot: str, answer: str, state: State) -> float:
    r, kev, exp = state
    if slot == "actively_exploited":  # yes/no
        p_yes = 0.95 if kev == 1 else 0.05
        return p_yes if answer == "yes" else (1.0 - p_yes)
    if slot == "epss_high":  # yes/no
        base = {"low": 0.2, "med": 0.5, "high": 0.8}[r]
        return base if answer == "yes" else (1.0 - base)
    if slot == "internet_exposed":  # yes/no
        p_yes = 0.9 if exp == 1 else 0.1
        return p_yes if answer == "yes" else (1.0 - p_yes)
    # default neutral
    return 0.5


def posterior(prior: Dict[State, float], slot: str, answer: str) -> Dict[State, float]:
    post: Dict[State, float] = {}
    for s, p in prior.items():
        post[s] = p * _likelihood(slot, answer, s)
    z = sum(post.values()) or 1.0
    return {s: v / z for s, v in post.items()}


def expected_entropy_drop_for_slot(prior: Dict[State, float], slot: str) -> float:
    h0 = entropy(prior)
    # binary answers yes/no
    drops = []
    for ans in ("yes", "no"):
        # p(ans) = sum_v p(ans|v) p(v)
        p_ans = sum(_likelihood(slot, ans, s) * p for s, p in prior.items())
        post = posterior(prior, slot, ans)
        h_post = entropy(post)
        drops.append(p_ans * h_post)
    h_exp = sum(drops)
    return max(0.0, h0 - h_exp)


