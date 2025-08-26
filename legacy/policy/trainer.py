from __future__ import annotations

from typing import Dict, List, Tuple

from policy.bandit import SoftmaxBandit, Action


def reward_weighted_update(bandit: SoftmaxBandit, episodes: List[Tuple[Dict[str, float], Action, float]], lr: float = 0.05) -> SoftmaxBandit:
	# Simple per-feature additive update: w_a += lr * r * x; w_other -= lr * r * x / (A-1)
	actions = bandit.actions
	for x, a_taken, r in episodes:
		for a in actions:
			for k, v in x.items():
				w = bandit.weights.setdefault(a, {}).get(k, 0.0)
				if a == a_taken:
					bandit.weights[a][k] = w + lr * r * v
				else:
					bandit.weights[a][k] = w - lr * r * v / max(1, (len(actions) - 1))
	return bandit


