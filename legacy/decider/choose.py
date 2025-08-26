from __future__ import annotations

from typing import Any, Dict, Optional, List
from pathlib import Path

try:
    # Local import; safe during early scaffolding
    from retrieval.local_snapshot import LocalSnapshotRetriever  # type: ignore
except Exception:  # pragma: no cover - retrieval layer may be absent in some envs
    LocalSnapshotRetriever = None  # type: ignore


def choose_action(
    domain: str,
    cve: Optional[str],
    assets_path: Optional[str],
    seed: int,
    force_action: Optional[str] = None,
    do_retrieval: bool = False,
    top_k: int = 2,
    answer_confidence_threshold: float = 0.6,
    cve_signals: Optional[Dict[str, Any]] = None,
    resolved_slots: Optional[Dict[str, Any]] = None,
    current_entropy: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a minimal deterministic decision stub.

    Mirrors the CLI's previous hardcoded output so downstream UI stays identical.
    """
    belief_entropy_value = 1.0 if current_entropy is None else float(current_entropy)
    state: Dict[str, Any] = {
        "domain": domain,
        "cve": cve,
        "assets_path": assets_path,
        "seed": seed,
        "belief_entropy": belief_entropy_value,
        "actions": {},
        "choice": "ask",
        "question": {
            "id": "internet_exposed",
            "text": "Is the affected asset exposed to the public internet?",
            "answers": ["yes", "no", "unknown"],
        },
    }

    # Simple scoring
    try:
        from decider.confidence import compute_answer_confidence  # local import
        from decider.eig import score_answer, score_ask, score_search, estimate_entropy_drop_for_ask, expected_entropy_drop_bayes
        from decider.bayes import build_prior_from_signals
        from features.cyber import build_features_from_signals
        from policy.bandit import SoftmaxBandit

        # derive confidence from signals if available
        if cve_signals is not None:
            cvss = cve_signals.get("cvss")
            cvss_norm = float(cvss) / 10.0 if isinstance(cvss, (int, float)) else 0.75
            epss = float(cve_signals.get("epss") or 0.2)
            kev_flag = bool(cve_signals.get("kev_flag", False))
        else:
            cvss_norm = 0.75
            epss = 0.2
            kev_flag = False

        conf = compute_answer_confidence({"cvss_norm": cvss_norm, "epss": epss, "kev_flag": 1 if kev_flag else 0})

        # Boost confidence from resolved slots if provided
        if resolved_slots:
            if resolved_slots.get("actively_exploited"):
                conf += 0.2
            if resolved_slots.get("epss_high"):
                conf += 0.1
            if resolved_slots.get("severity_high"):
                conf += 0.1
            conf = min(1.0, conf)

        # ask and search scores with calibrated heuristics
        entropy = float(state.get("belief_entropy", 1.0))
        # Base heuristic drop (unscaled, rough upper bound when entropy=1)
        ask_drop_raw = estimate_entropy_drop_for_ask(entropy, plausible_outcomes=3)
        if cve_signals is not None:
            prior_dist = build_prior_from_signals(cve_signals, resolved_slots=resolved_slots)
            bayes_drop_kev = expected_entropy_drop_bayes(prior_dist, slot="actively_exploited")
            bayes_drop_epss = expected_entropy_drop_bayes(prior_dist, slot="epss_high")
            ask_drop_raw = max(ask_drop_raw, bayes_drop_kev, bayes_drop_epss)
        # Rescale to EIG scale comparable with other actions
        ask_eig = min(0.25, 0.2 * ask_drop_raw)
        ask_cost = 0.12
        ask_scores = {"eig": round(ask_eig, 3), "cost": ask_cost, "net": round(ask_eig - ask_cost, 3)}
        search_scores = score_search(recency_flag=kev_flag)

        actions = {
            "answer": score_answer(confidence=conf),
            "ask": ask_scores,
            "search": search_scores,
        }
        state["actions"] = actions
        # Choose by learned policy if available; else by net score
        choice_by_net = max(actions.items(), key=lambda kv: kv[1]["net"])[0]
        choice_by_policy: Optional[str] = None
        try:
            weights_path = Path("policy/weights.json")
            if weights_path.exists():
                heur = {
                    "ask_eig_heur": float(ask_scores.get("eig", 0.0)),
                    "bayes_drop_kev": 0.0,
                    "bayes_drop_epss": 0.0,
                }
                if cve_signals is not None:
                    prior_dist = build_prior_from_signals(cve_signals, resolved_slots=resolved_slots)
                    bayes_drop_kev = expected_entropy_drop_bayes(prior_dist, slot="actively_exploited")
                    bayes_drop_epss = expected_entropy_drop_bayes(prior_dist, slot="epss_high")
                    heur["bayes_drop_kev"] = float(bayes_drop_kev)
                    heur["bayes_drop_epss"] = float(bayes_drop_epss)
                feat = build_features_from_signals(
                    signals=cve_signals,
                    entropy=state.get("belief_entropy"),
                    resolved_slots=resolved_slots,
                    priors=None,
                    heuristics=heur,
                )
                bandit = SoftmaxBandit.load(str(weights_path))
                choice_by_policy = bandit.choose(feat)
        except Exception:
            choice_by_policy = None

        # Direct answer shortcuts for strong graph evidence
        if force_action is None and cve_signals is not None:
            cvss_val = cve_signals.get("cvss")
            epss_val = cve_signals.get("epss")
            kev_flag = bool(cve_signals.get("kev_flag", False))
            if kev_flag or (isinstance(cvss_val, (int, float)) and cvss_val >= 8.0) or (
                isinstance(epss_val, (int, float)) and epss_val >= 0.7
            ):
                state["choice"] = "answer"
            else:
                chosen = choice_by_policy or choice_by_net
                state["choice"] = (
                    "answer" if conf >= answer_confidence_threshold and force_action is None else chosen
                )
        else:
            chosen = choice_by_policy or choice_by_net
            state["choice"] = (
                "answer" if conf >= answer_confidence_threshold and force_action is None else chosen
            )
        state["confidence"] = round(conf, 3)
        if resolved_slots:
            state["slots"] = resolved_slots
    except Exception:
        # Fallback to static actions if scoring unavailable
        state["actions"] = {
            "answer": {"eig": 0.05, "cost": 0.05, "net": 0.0},
            "ask": {"eig": 0.25, "cost": 0.1, "net": 0.15},
            "search": {"eig": 0.2, "cost": 0.15, "net": 0.05},
        }

    if force_action in {"answer", "ask", "search"}:
        state["choice"] = force_action

    if state["choice"] == "search" and do_retrieval:
        preview: List[Dict[str, Any]] = []
        if LocalSnapshotRetriever is not None:
            retriever = LocalSnapshotRetriever(corpus_name=f"{domain}-snapshot")
            docs = retriever.retrieve(query=cve or "generic", k=top_k)
            for d in docs:
                preview.append({"text": d.text, "metadata": d.metadata})
        state["retrieval_preview"] = preview

    # Simple EIG-inspired adjustment: prefer Ask if high entropy and no assets known,
    # but only when we lack graph signals (otherwise rely on scores)
    try:
        entropy = float(state.get("belief_entropy", 1.0))
        has_assets = bool(state.get("assets_preview"))
        if entropy > 0.8 and not has_assets and force_action is None and cve_signals is None:
            state["choice"] = "ask"
    except Exception:
        pass

    return state


