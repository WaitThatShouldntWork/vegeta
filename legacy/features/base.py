from __future__ import annotations

from typing import Any, Dict


def clamp01(x: float) -> float:
	return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def as_float(value: Any) -> float | None:
	try:
		if value is None:
			return None
		return float(value)
	except Exception:
		return None


def default_features() -> Dict[str, float]:
	return {
		"bias": 1.0,
		"cvss_norm": 0.0,
		"epss": 0.0,
		"kev_flag": 0.0,
		"risk": 0.0,
		"entropy": 1.0,
		"centrality": 0.0,
		"slot_actively_exploited": 0.0,
		"slot_epss_high": 0.0,
		"slot_severity_high": 0.0,
		"ask_eig_heur": 0.0,
		"bayes_drop_kev": 0.0,
		"bayes_drop_epss": 0.0,
	}


