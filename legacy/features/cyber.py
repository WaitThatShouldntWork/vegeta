from __future__ import annotations

from typing import Any, Dict

from features.base import default_features, clamp01, as_float
from decider.triage import compute_risk_score


def build_features_from_signals(*, signals: Dict[str, Any] | None, entropy: float | None, resolved_slots: Dict[str, Any] | None, priors: Dict[str, float] | None, heuristics: Dict[str, float] | None) -> Dict[str, float]:
	feat = default_features()
	if signals:
		cvss_val = as_float(signals.get("cvss"))
		epss_val = as_float(signals.get("epss"))
		kev_flag = bool(signals.get("kev_flag", False))
		feat["cvss_norm"] = clamp01(0.0 if cvss_val is None else cvss_val / 10.0)
		feat["epss"] = clamp01(0.0 if epss_val is None else epss_val)
		feat["kev_flag"] = 1.0 if kev_flag else 0.0
		feat["risk"] = compute_risk_score(cvss=cvss_val, epss=epss_val, kev_flag=kev_flag)
	if entropy is not None:
		feat["entropy"] = clamp01(entropy)
	if priors and "centrality" in priors:
		feat["centrality"] = clamp01(float(priors["centrality"]))
	if resolved_slots:
		feat["slot_actively_exploited"] = 1.0 if resolved_slots.get("actively_exploited") else 0.0
		feat["slot_epss_high"] = 1.0 if resolved_slots.get("epss_high") else 0.0
		feat["slot_severity_high"] = 1.0 if resolved_slots.get("severity_high") else 0.0
	if heuristics:
		feat["ask_eig_heur"] = float(heuristics.get("ask_eig_heur", 0.0))
		feat["bayes_drop_kev"] = float(heuristics.get("bayes_drop_kev", 0.0))
		feat["bayes_drop_epss"] = float(heuristics.get("bayes_drop_epss", 0.0))
	return feat


