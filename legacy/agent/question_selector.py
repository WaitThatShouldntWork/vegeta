from __future__ import annotations

from typing import Any, Dict, Tuple

from decider.bayes import build_prior_from_signals, expected_entropy_drop_for_slot


def pick_next_slot_by_eig(
    *,
    signals: Dict[str, Any] | None,
    resolved_slots: Dict[str, Any] | None = None,
    candidates: Tuple[str, ...] = ("actively_exploited", "epss_high", "internet_exposed", "asset_present"),
    exclude: Tuple[str, ...] = tuple(),
) -> str:
	prior = build_prior_from_signals(signals or {}, resolved_slots=resolved_slots)
	slots = [s for s in candidates if s not in set(exclude)] or list(candidates)
	slots.sort(key=lambda s: expected_entropy_drop_for_slot(prior, s), reverse=True)
	return slots[0]


