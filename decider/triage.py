from __future__ import annotations

from typing import Dict, Tuple


def compute_risk_score(cvss: float | None, epss: float | None, kev_flag: bool) -> float:
    cvss_norm = 0.0 if cvss is None else max(0.0, min(1.0, cvss / 10.0))
    epss_val = 0.0 if epss is None else max(0.0, min(1.0, epss))
    kev_val = 1.0 if kev_flag else 0.0
    return 0.5 * cvss_norm + 0.4 * epss_val + 0.1 * kev_val


def recommend_action(
    *,
    cvss: float | None,
    epss: float | None,
    kev_flag: bool,
    asset_present: bool,
    internet_exposed: bool,
    business_critical: bool,
) -> Tuple[str, Dict[str, float]]:
    """Return (action, details) where action in {'patch_now','plan_patch','monitor'}.

    Simple policy: risk score then escalate if exposed/critical.
    """
    risk = compute_risk_score(cvss, epss, kev_flag)
    # Base decision
    if risk >= 0.6:
        action = "patch_now"
    elif risk >= 0.3:
        action = "plan_patch"
    else:
        action = "monitor"

    # Escalation for exposure/criticality
    if action != "patch_now" and asset_present and (internet_exposed or business_critical):
        action = "patch_now" if risk >= 0.4 else "plan_patch"

    return action, {"risk": round(risk, 3)}


