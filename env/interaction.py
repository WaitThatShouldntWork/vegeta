from __future__ import annotations

from typing import Any, Dict, Tuple

from eval.noisy_user import noisy_yes_no
from decider.bayes import build_prior_from_signals, expected_entropy_drop_for_slot


class AskEnv:
	def __init__(self, *, signals: Dict[str, Any] | None, error_rate: float = 0.1) -> None:
		self.signals = signals or {}
		self.error_rate = float(error_rate)
		self.steps = 0
		self.max_steps = 5

	def step(self, slot: str, seed: int) -> Tuple[str, float]:
		# sample noisy yes/no
		ans = noisy_yes_no(p_true=0.5, error_rate=self.error_rate, seed=seed)
		# reward is negative small cost for asking
		self.steps += 1
		reward = -0.05
		return ans, reward

	def best_slot_by_eig(self) -> str:
		prior = build_prior_from_signals(self.signals)
		cands = ["actively_exploited", "epss_high", "internet_exposed"]
		scored = [(s, expected_entropy_drop_for_slot(prior, s)) for s in cands]
		scored.sort(key=lambda t: t[1], reverse=True)
		return scored[0][0]


