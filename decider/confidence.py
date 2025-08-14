from __future__ import annotations

from typing import Dict


def compute_answer_confidence(signals: Dict[str, float]) -> float:
    """Tiny heuristic confidence: bounded in [0,1].

    signals may include: cvss_norm, epss, kev_flag (0 or 1)
    """
    cvss = max(0.0, min(1.0, signals.get("cvss_norm", 0.0)))
    epss = max(0.0, min(1.0, signals.get("epss", 0.0)))
    kev = 1.0 if signals.get("kev_flag", 0.0) else 0.0
    # slightly higher weight on EPSS to prefer answering on likely exploitation
    s = 0.45 * cvss + 0.45 * epss + 0.1 * kev
    return max(0.0, min(1.0, s))


