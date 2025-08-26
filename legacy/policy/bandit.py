from __future__ import annotations

import json
from typing import Dict, List

Action = str  # one of {"answer","ask","search"}


class SoftmaxBandit:
	def __init__(self, actions: List[Action], weights: Dict[Action, Dict[str, float]] | None = None) -> None:
		self.actions = actions
		self.weights = weights or {a: {} for a in actions}

	def score(self, features: Dict[str, float]) -> Dict[Action, float]:
		return {a: sum(features.get(k, 0.0) * w for k, w in self.weights.get(a, {}).items()) for a in self.actions}

	def probs(self, features: Dict[str, float]) -> Dict[Action, float]:
		import math
		s = self.score(features)
		mx = max(s.values()) if s else 0.0
		exp = {a: math.exp(v - mx) for a, v in s.items()}
		t = sum(exp.values()) or 1.0
		return {a: v / t for a, v in exp.items()}

	def choose(self, features: Dict[str, float]) -> Action:
		p = self.probs(features)
		return max(p.items(), key=lambda kv: kv[1])[0]

	def save(self, path: str) -> None:
		with open(path, "w", encoding="utf-8") as f:
			json.dump({"actions": self.actions, "weights": self.weights}, f, indent=2)

	@staticmethod
	def load(path: str) -> "SoftmaxBandit":
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		return SoftmaxBandit(actions=data["actions"], weights=data["weights"])


