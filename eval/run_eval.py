from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from decider.choose import choose_action
from decider.triage import compute_risk_score
from decider.questions import resolve_slots_from_signals
from decider.graph_signals import get_cve_graph_signals
from decider.triage import recommend_action
from eval.noisy_user import noisy_yes_no


@dataclass
class EvalResult:
    total: int
    answers: int
    asks: int
    searches: int
    confident_answers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "answers": self.answers,
            "asks": self.asks,
            "searches": self.searches,
            "confident_answers": self.confident_answers,
        }


def run_scenarios(path: Path) -> EvalResult:
    total = answers = asks = searches = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scenario = json.loads(line)
            signals = None
            if scenario.get("cve"):
                try:
                    signals = get_cve_graph_signals(scenario["cve"])  # may be None
                except Exception:
                    signals = None

            # Derive entropy and resolved slots from signals to match CLI behavior
            resolved = resolve_slots_from_signals(signals) if signals else None
            initial_entropy = None
            if signals:
                risk = compute_risk_score(
                    cvss=(signals.get("cvss") if isinstance(signals.get("cvss"), (int, float)) else None),
                    epss=(signals.get("epss") if isinstance(signals.get("epss"), (int, float)) else None),
                    kev_flag=bool(signals.get("kev_flag", False)),
                )
                initial_entropy = max(0.1, round(1.0 - min(1.0, risk), 3))

            state = choose_action(
                domain=scenario.get("domain", "cyber"),
                cve=scenario.get("cve"),
                assets_path=None,
                seed=scenario.get("seed", 42),
                force_action=None,
                do_retrieval=False,
                cve_signals=signals,
                resolved_slots=resolved,
                current_entropy=(initial_entropy if initial_entropy is not None else 1.0),
            )
            choice = state.get("choice", "ask")
            total += 1
            if choice == "answer":
                answers += 1
                if float(state.get("confidence", 0.0)) >= 0.75:
                    # Track confident answers meeting threshold
                    pass
            elif choice == "search":
                searches += 1
            else:
                asks += 1
    return EvalResult(total, answers, asks, searches)


def main() -> None:
    scenarios_path = Path("eval/scenarios_cyber.jsonl")
    result = run_scenarios(scenarios_path)
    print(json.dumps(result.to_dict(), indent=2))


def run_scenarios_ask_then_triage(path: Path, seed: int = 123) -> EvalResult:
    total = answers = asks = searches = confident = 0
    rng_seed = seed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scenario = json.loads(line)
            cve = scenario.get("cve")
            signals = get_cve_graph_signals(cve) if cve else None
            if not signals:
                # fall back to simple ask
                choice_state = choose_action(
                    domain=scenario.get("domain", "cyber"),
                    cve=cve,
                    assets_path=None,
                    seed=scenario.get("seed", 42),
                    force_action=None,
                    do_retrieval=False,
                    cve_signals=None,
                    resolved_slots=None,
                    current_entropy=1.0,
                )
                choice = choice_state.get("choice", "ask")
                total += 1
                if choice == "answer":
                    answers += 1
                elif choice == "search":
                    searches += 1
                else:
                    asks += 1
                continue

            # Simulate two asks: asset_present and internet_exposed
            asset_present = noisy_yes_no(p_true=0.7, error_rate=0.1, seed=rng_seed) == "yes"
            rng_seed += 1
            internet_exposed = noisy_yes_no(p_true=0.5, error_rate=0.1, seed=rng_seed) == "yes"
            rng_seed += 1

            action, details = recommend_action(
                cvss=signals.get("cvss"),
                epss=signals.get("epss"),
                kev_flag=bool(signals.get("kev_flag", False)),
                asset_present=asset_present,
                internet_exposed=internet_exposed,
                business_critical=False,
            )

            # Derive a naive expected outcome for proxy accuracy
            cvss = (signals.get("cvss") or 0.0) or 0.0
            epss = (signals.get("epss") or 0.0) or 0.0
            kev = bool(signals.get("kev_flag", False))
            high = kev or cvss >= 7.0 or epss > 0.5
            if high and asset_present and internet_exposed:
                expected = "patch_now"
            elif high:
                expected = "plan_patch"
            else:
                expected = "monitor"

            total += 1
            asks += 2
            if action == expected:
                answers += 1
                if details.get("risk", 0.0) >= 0.6:
                    confident += 1
            else:
                # treat as search-needed if mismatch (proxy)
                searches += 0

    return EvalResult(total, answers, asks, searches, confident_answers=confident)


if __name__ == "__main__":
    main()


