from __future__ import annotations

from typing import Any, Dict


def resolve_slots_from_signals(signals: Dict[str, Any]) -> Dict[str, Any]:
    """Derive simple question slots from CVE graph signals.

    - actively_exploited: bool from kev_flag
    - epss_high: bool from epss > 0.5
    - severity_high: bool from cvss >= 7.0
    """
    kev_flag = bool(signals.get("kev_flag", False))
    epss = signals.get("epss")
    try:
        epss_val = float(epss) if epss is not None else None
    except (TypeError, ValueError):
        epss_val = None
    cvss = signals.get("cvss")
    try:
        cvss_val = float(cvss) if cvss is not None else None
    except (TypeError, ValueError):
        cvss_val = None

    slots: Dict[str, Any] = {
        "actively_exploited": kev_flag,
        "epss_high": bool(epss_val is not None and epss_val > 0.5),
        "severity_high": bool(cvss_val is not None and cvss_val >= 7.0),
    }
    return slots


