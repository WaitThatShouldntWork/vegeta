from __future__ import annotations

import json
import time
from typing import Optional, Any, Dict
import random

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TextColumn
from decider.choose import choose_action
from config.settings import get_aura_settings
from ingest.cyber_etl import (
    sample_inputs_mini,
    build_snapshot_from_inputs,
    upsert_snapshot_to_neo4j,
)
from ingest.nvd_fetch import fetch_nvd_recent, extract_cves, extract_cpe_map
from ingest.kev_fetch import fetch_kev_json, extract_kev_cves
from ingest.epss_fetch import fetch_epss_csv, parse_epss_csv_gz
from db.neo4j_client import get_driver
from decider.triage import recommend_action, compute_risk_score
from pathlib import Path
from eval.run_eval import run_scenarios
from eval.run_eval import run_scenarios_ask_then_triage
from decider.graph_signals import get_cve_graph_signals, get_centrality_prior, get_cve_graph_signals_batch
from decider.questions import resolve_slots_from_signals
from ingest.working_set import build_working_set_from_neo4j
from features.cyber import build_features_from_signals
from policy.bandit import SoftmaxBandit
from decider.bayes import build_prior_from_signals
from decider.eig import expected_entropy_drop_bayes
from retrieval.wikipedia import search_wikipedia
from db.writeback import upsert_fact_with_source, fact_exists, has_slot_value
from agent.question_generator import render_question_from_slot, render_from_domain_pack
from agent.question_selector import pick_next_slot_by_eig
from emg.loader import load_seed_entities
from emg.query import find_films_by_slots, get_entity_types, list_slots_for_type, list_common_slots


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """VEGETA: Expected Information Gain demo CLI."""
    return


@app.command()
def decide(
    domain: str = typer.Option("cyber", "--domain", help="Domain pack to use"),
    cve: Optional[str] = typer.Option(None, "--cve", help="CVE identifier for cyber domain"),
    assets: Optional[str] = typer.Option(None, "--assets", help="Path to assets JSON"),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed"),
    force_action: Optional[str] = typer.Option(
        None,
        "--force-action",
        help="Override choice to one of: answer|ask|search",
    ),
    retrieve: bool = typer.Option(
        False, "--retrieve", help="If choice is search, include retrieval preview"
    ),
    top_k: int = typer.Option(2, "--top-k", help="Number of preview docs when retrieving"),
) -> None:
    """Stream a stub decision showing the policy wiring."""
    assets_preview: Optional[Dict[str, Any]] = None
    assets_count: Optional[int] = None
    cve_signals: Optional[Dict[str, Any]] = None
    initial_entropy: Optional[float] = None
    if assets:
        try:
            with open(assets, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "assets" in data and isinstance(data["assets"], list):
                assets_count = len(data["assets"])
                assets_preview = {"sample": data["assets"][:2]}
            elif isinstance(data, list):
                assets_count = len(data)
                assets_preview = {"sample": data[:2]}
            else:
                assets_preview = {"keys": list(data)[:3] if isinstance(data, dict) else None}
        except Exception as exc:
            assets_preview = {"error": str(exc)}

    # Try to fetch graph signals if CVE provided
    if cve:
        try:
            cve_signals = get_cve_graph_signals(cve)
        except Exception:
            cve_signals = None
        # derive a simple initial entropy from graph risk if available
        if cve_signals:
            risk = compute_risk_score(
                cvss=(cve_signals.get("cvss") if isinstance(cve_signals.get("cvss"), (int, float)) else None),
                epss=(cve_signals.get("epss") if isinstance(cve_signals.get("epss"), (int, float)) else None),
                kev_flag=bool(cve_signals.get("kev_flag", False)),
            )
            # Higher risk → lower entropy; clamp to [0.1, 1.0]
            initial_entropy = max(0.1, round(1.0 - min(1.0, risk), 3))
        # Blend a tiny centrality prior to nudge entropy further if graph is very peaky
        try:
            centrality = get_centrality_prior(cve)
            if centrality is not None and initial_entropy is not None:
                # Reduce entropy by up to 0.1 based on centrality
                initial_entropy = max(0.1, round(initial_entropy * (1.0 - 0.1 * centrality), 3))
        except Exception:
            pass

    state = choose_action(
        domain=domain,
        cve=cve,
        assets_path=assets,
        seed=seed,
        force_action=force_action,
        do_retrieval=retrieve,
        top_k=top_k,
        cve_signals=cve_signals,
        resolved_slots=(resolve_slots_from_signals(cve_signals) if cve_signals else None),
        current_entropy=initial_entropy,
    )
    if assets_preview is not None:
        state["assets_preview"] = assets_preview
    if assets_count is not None:
        state["assets_count"] = assets_count
    if cve_signals is not None:
        state["cve_signals"] = cve_signals
        state["resolved_slots"] = resolve_slots_from_signals(cve_signals)

    table = Table(title="VEGETA Action Scores", show_lines=True)
    table.add_column("Action")
    table.add_column("EIG", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Net", justify="right")

    for action, scores in state["actions"].items():
        table.add_row(action, f"{scores['eig']:.3f}", f"{scores['cost']:.3f}", f"{scores['net']:.3f}")

    console.rule("Belief")
    console.print(f"Entropy: {state['belief_entropy']:.3f}")

    # Show AuraDS availability (env-driven) for transparency
    aura = get_aura_settings()
    if aura.uri and aura.user:
        console.print("AuraDS: configured (URI + USER detected)")
    else:
        console.print("AuraDS: not configured")
    console.rule("Policy")
    console.print(table)
    console.rule("Decision")
    if state["choice"] == "ask":
        console.print(
            f"Choice: [bold]{state['choice'].upper()}[/bold] → Q: {state['question']['text']}"
        )
    else:
        console.print(f"Choice: [bold]{state['choice'].upper()}[/bold]")

    console.rule("JSON")
    console.print_json(data=state)


@app.command(name="ingest-cyber")
def ingest_cyber(
    mini: bool = typer.Option(True, "--mini/--no-mini", help="Use mini sample inputs"),
    fetch_nvd: bool = typer.Option(False, "--fetch-nvd/--no-fetch-nvd", help="Fetch recent NVD CVEs"),
    limit: int = typer.Option(50, "--limit", help="NVD results per page (approx)"),
    include_attack: bool = typer.Option(False, "--include-attack/--no-include-attack", help="Append a small ATT&CK subset"),
    upsert: bool = typer.Option(False, "--upsert/--no-upsert", help="Write to Aura if configured"),
    snapshot_label: Optional[str] = typer.Option(None, "--snapshot-label", help="Optional label to tag nodes"),
) -> None:
    """Build a tiny cyber snapshot (and optionally upsert to Aura)."""
    if fetch_nvd:
        raw = fetch_nvd_recent(limit=limit)
        cves = extract_cves(raw)
        cpe_by_cve = extract_cpe_map(raw)
        kev = extract_kev_cves(fetch_kev_json())
        try:
            epss_scores, epss_pcts = parse_epss_csv_gz(fetch_epss_csv())
        except Exception:
            epss_scores, epss_pcts = {}, {}
        inputs = {"cves": cves, "cpe_by_cve": cpe_by_cve, "kev_cves": kev, "epss_by_cve": epss_scores}
    else:
        inputs = sample_inputs_mini() if mini else sample_inputs_mini()
    snapshot = build_snapshot_from_inputs(
        cves=inputs["cves"],
        cpe_by_cve=inputs["cpe_by_cve"],
        kev_cves=inputs["kev_cves"],
        epss_by_cve=inputs["epss_by_cve"],
        epss_pct_by_cve=locals().get("epss_pcts", {}),
    )

    # Optional: append ATT&CK stub if requested (flag currently not exposed)
    # from ingest.attack_stub import build_attack_stub
    # atk = build_attack_stub()
    # snapshot["nodes"].extend(atk["nodes"])  # type: ignore[index]
    # snapshot["relationships"].extend(atk["relationships"])  # type: ignore[index]

    # Optional ATT&CK include
    if include_attack:
        try:
            from ingest.attack_stix import fetch_attack_stix, parse_attack
            atk_nodes, atk_rels = parse_attack(fetch_attack_stix(), limit=100)
            snapshot["nodes"].extend(atk_nodes)
            snapshot["relationships"].extend(atk_rels)
        except Exception as exc:
            console.print(f"ATT&CK include skipped: {exc}")

    console.rule("Snapshot")
    console.print({
        "nodes": len(snapshot["nodes"]),
        "relationships": len(snapshot["relationships"]),
    })

    if upsert:
        console.rule("Upsert")
        result = upsert_snapshot_to_neo4j(snapshot, snapshot_label=snapshot_label)
        console.print(result)


@app.command(name="graph-counts")
def graph_counts() -> None:
    """Print simple counts by label from Neo4j (if configured)."""
    driver = get_driver()
    if driver is None:
        console.print("Neo4j not configured or driver missing.")
        raise typer.Exit(code=1)
    with driver.session() as session:
        q = """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS cnt
        ORDER BY cnt DESC
        """
        rows = session.run(q).data()
        table = Table(title="Graph counts by label", show_lines=True)
        table.add_column("Label")
        table.add_column("Count", justify="right")
        for r in rows:
            table.add_row(str(r["label"]), str(r["cnt"]))
        console.print(table)


@app.command(name="ws")
def working_set(
    cve: str = typer.Option(..., "--cve", help="CVE identifier"),
    max_nodes: int = typer.Option(100, "--max-nodes", help="Max nodes"),
    max_rels: int = typer.Option(500, "--max-rels", help="Max relationships"),
) -> None:
    """Print a small working set summary pulled from Neo4j around a CVE."""
    ws = build_working_set_from_neo4j(cve_id=cve, max_nodes=max_nodes, max_rels=max_rels)
    console.rule("Working Set")
    console.print({
        "nodes": len(ws.get("nodes", [])),
        "relationships": len(ws.get("relationships", [])),
        "source": ws.get("meta", {}).get("source"),
    })


@app.command(name="triage")
def triage(
    cvss: Optional[float] = typer.Option(None, "--cvss", help="CVSS base score (0-10)"),
    epss: Optional[float] = typer.Option(None, "--epss", help="EPSS probability (0-1)"),
    kev: bool = typer.Option(False, "--kev/--no-kev", help="CISA KEV exploited"),
    asset_present: bool = typer.Option(True, "--asset-present/--no-asset-present"),
    internet_exposed: bool = typer.Option(False, "--internet-exposed/--no-internet-exposed"),
    business_critical: bool = typer.Option(False, "--business-critical/--no-business-critical"),
) -> None:
    """Minimal triage recommendation based on risk and exposure inputs."""
    action, details = recommend_action(
        cvss=cvss,
        epss=epss,
        kev_flag=kev,
        asset_present=asset_present,
        internet_exposed=internet_exposed,
        business_critical=business_critical,
    )
    console.rule("Triage")
    console.print({"action": action, **details})


@app.command(name="ask-then-triage")
def ask_then_triage(
    # emulate two common asks before triage
    asset_present: bool = typer.Option(True, "--asset-present/--no-asset-present"),
    internet_exposed: bool = typer.Option(False, "--internet-exposed/--no-internet-exposed"),
    cvss: Optional[float] = typer.Option(None, "--cvss"),
    epss: Optional[float] = typer.Option(None, "--epss"),
    kev: bool = typer.Option(False, "--kev/--no-kev"),
) -> None:
    """Simulate two Ask slots, then produce a recommendation."""
    action, details = recommend_action(
        cvss=cvss,
        epss=epss,
        kev_flag=kev,
        asset_present=asset_present,
        internet_exposed=internet_exposed,
        business_critical=False,
    )
    console.rule("Ask → Triage")
    console.print({
        "asked": {"asset_present": asset_present, "internet_exposed": internet_exposed},
        "action": action,
        **details,
    })


@app.command(name="eval")
def eval_cmd(
    scenarios: Path = typer.Option(Path("eval/scenarios_cyber.jsonl"), "--scenarios", help="Path to scenarios JSONL"),
    save: Optional[Path] = typer.Option(None, "--save", help="Optional path to save JSON summary"),
) -> None:
    """Run the tiny eval harness and print a summary."""
    result = run_scenarios(scenarios)
    console.rule("Eval Summary")
    answer_rate = (result.answers / result.total) if result.total else 0.0
    avg_asks = (result.asks / result.total) if result.total else 0.0
    avg_searches = (result.searches / result.total) if result.total else 0.0
    # For one-step eval we track answer_rate; correctness is evaluated in eval-ask
    targets = {"answer_rate>=": 0.2, "avg_asks<=": 3.0, "avg_searches<=": 0.7}
    summary = {
        "total": result.total,
        "answers": result.answers,
        "asks": result.asks,
        "searches": result.searches,
        "answer_rate": round(answer_rate, 3),
        "avg_asks": round(avg_asks, 3),
        "avg_searches": round(avg_searches, 3),
        "targets": targets,
        "pass": bool(answer_rate >= 0.2 and avg_asks <= 3.0 and avg_searches <= 0.7),
    }
    console.print(summary)
    console.print(f"Result: {'PASS' if summary['pass'] else 'FAIL'} (targets: {targets})")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        save.write_text(_json.dumps(summary, indent=2), encoding="utf-8")


@app.command(name="eval-ask")
def eval_ask_then_triage_cmd(
    scenarios: Path = typer.Option(Path("eval/scenarios_cyber.jsonl"), "--scenarios", help="Path to scenarios JSONL"),
    seed: int = typer.Option(123, "--seed", help="Noisy user seed"),
    save: Optional[Path] = typer.Option(None, "--save", help="Optional path to save JSON summary"),
) -> None:
    """Run eval with a simple ask-then-triage simulation and print a summary."""
    result = run_scenarios_ask_then_triage(scenarios, seed=seed)
    console.rule("Eval (Ask → Triage) Summary")
    accuracy = (result.answers / result.total) if result.total else 0.0
    avg_asks = (result.asks / result.total) if result.total else 0.0
    avg_searches = (result.searches / result.total) if result.total else 0.0
    targets = {"accuracy>=": 0.85, "avg_asks<=": 3.0, "avg_searches<=": 0.7}
    summary = {
        "total": result.total,
        "answers": result.answers,
        "asks": result.asks,
        "searches": result.searches,
        "confident_answers": result.confident_answers,
        "accuracy": round(accuracy, 3),
        "avg_asks": round(avg_asks, 3),
        "avg_searches": round(avg_searches, 3),
        "targets": targets,
        "pass": bool(accuracy >= 0.85 and avg_asks <= 3.0 and avg_searches <= 0.7),
    }
    console.print(summary)
    console.print(f"Result: {'PASS' if summary['pass'] else 'FAIL'} (targets: {targets})")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        save.write_text(_json.dumps(summary, indent=2), encoding="utf-8")


@app.command(name="train")
def train(
    scenarios: Path = typer.Option(Path("eval/scenarios_cyber.jsonl"), "--scenarios", help="Path to scenarios JSONL"),
    episodes: int = typer.Option(50, "--episodes", help="Training episodes"),
    save: Path = typer.Option(Path("policy/weights.json"), "--save", help="Where to save the learned weights"),
    log_every: int = typer.Option(1000, "--log-every", help="Log progress every N steps"),
    lr: float = typer.Option(0.02, "--lr", help="Learning rate for updates"),
    subset: Optional[int] = typer.Option(None, "--subset", help="Use only N scenarios per episode (speed)"),
    reset: bool = typer.Option(False, "--reset/--no-reset", help="Ignore existing weights and start fresh"),
) -> None:
    """Train a tiny contextual bandit over actions using reward-weighted updates."""
    try:
        from policy.trainer import reward_weighted_update
    except Exception as exc:
        console.print(f"Trainer unavailable: {exc}")
        raise typer.Exit(code=1)

    actions = ["answer", "ask", "search"]
    bandit = SoftmaxBandit(actions=actions)
    # Warm start if weights already exist at save path (unless reset)
    used_warm_start = False
    try:
        if (not reset) and save.exists():
            bandit = SoftmaxBandit.load(str(save))
            used_warm_start = True
            console.print({"warm_start": str(save)})
    except Exception:
        used_warm_start = False
    # If not warm-starting, seed baseline weights to avoid search dominance
    if not used_warm_start:
        bandit.weights = {
            "answer": {"bias": -0.05, "risk": 0.5, "kev_flag": 0.3, "epss": 0.3},
            "ask": {"bias": -0.02, "entropy": 0.3, "ask_eig_heur": 0.2, "bayes_drop_kev": 0.2, "bayes_drop_epss": 0.2},
            "search": {"bias": -0.2},
        }
        console.print({"init_baseline": True})

    # Simple loop over scenarios; use one-step reward proxy from our existing eval policy
    import json as _json
    rng_seed = 123
    episodes_data = []
    with scenarios.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    # Parse once
    import json as _json
    parsed = [_json.loads(ln) for ln in lines]
    # Optional subset
    if subset is not None and subset > 0 and subset < len(parsed):
        parsed = random.sample(parsed, k=subset)
    # Batch-fetch signals once
    cve_ids = [p.get("cve") for p in parsed if p.get("cve")]
    signals_map = get_cve_graph_signals_batch(cve_ids) if cve_ids else {}
    console.print({
        "scenarios_total": len(lines),
        "parsed": len(parsed),
        "with_cve": len(cve_ids),
    })
    # Prebuild features once
    feats: list[Dict[str, float]] = []
    for p in parsed:
        cve = p.get("cve")
        sig = signals_map.get(cve) if cve else None
        resolved = resolve_slots_from_signals(sig) if sig else None
        heur = {"ask_eig_heur": 0.2, "bayes_drop_kev": 0.0, "bayes_drop_epss": 0.0}
        try:
            if sig:
                prior = build_prior_from_signals(sig, resolved_slots=resolved)
                heur["bayes_drop_kev"] = float(expected_entropy_drop_bayes(prior, slot="actively_exploited"))
                heur["bayes_drop_epss"] = float(expected_entropy_drop_bayes(prior, slot="epss_high"))
        except Exception:
            pass
        feat = build_features_from_signals(signals=sig, entropy=1.0, resolved_slots=resolved, priors=None, heuristics=heur)
        feats.append(feat)
    steps_per_episode = len(feats)
    total_steps = max(1, episodes) * max(1, steps_per_episode)
    console.print({"episodes": episodes, "steps_per_episode": steps_per_episode, "total_steps": total_steps})
    start = time.perf_counter()
    step = 0
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*progress_columns, transient=False) as progress:
        task_id = progress.add_task("Training", total=total_steps)
        for ep in range(1, max(1, episodes) + 1):
            for feat in feats:
                a = bandit.choose(feat)
                # crude reward: prefer answer if high risk/kev/epss; reward ask by predicted EIG; penalize search
                risk = feat.get("risk", 0.0)
                kev = feat.get("kev_flag", 0.0)
                ask_info = feat.get("bayes_drop_kev", 0.0) + feat.get("bayes_drop_epss", 0.0)
                reward = 0.0
                if a == "answer":
                    reward = 1.0 if (risk >= 0.6 or kev >= 0.5) else -1.0
                elif a == "ask":
                    reward = 0.1 * ask_info - 0.05
                else:
                    reward = -0.15
                episodes_data.append((feat, a, reward))
                rng_seed += 1
                step += 1
                progress.advance(task_id, 1)
                if step % max(1, log_every) == 0:
                    elapsed = time.perf_counter() - start
                    rate = step / elapsed if elapsed > 0 else 0.0
                    progress.update(task_id, description=f"Training (ep {ep}, {rate:.1f} steps/s)")

    console.print("Updating policy weights...")
    bandit = reward_weighted_update(bandit, episodes_data, lr=lr)
    elapsed = time.perf_counter() - start
    console.print({
        "done": True,
        "elapsed_s": round(elapsed, 2),
        "avg_steps_per_s": round((step / elapsed) if elapsed > 0 else 0.0, 2),
    })
    save.parent.mkdir(parents=True, exist_ok=True)
    bandit.save(str(save))
    console.print({"saved": str(save)})


@app.command(name="simulate")
def simulate(
    cve: str = typer.Option("CVE-2025-TEST", "--cve"),
    steps: int = typer.Option(3, "--steps"),
) -> None:
    """Run a short Ask simulation with the noisy-user env and show picked slots."""
    from env.interaction import AskEnv
    signals = get_cve_graph_signals(cve)
    env = AskEnv(signals=signals, error_rate=0.1)
    picked = []
    seed = 1
    for _ in range(max(1, steps)):
        slot = env.best_slot_by_eig()
        ans, _ = env.step(slot, seed=seed)
        picked.append({"slot": slot, "answer": ans})
        seed += 1
    console.print({"cve": cve, "trace": picked})


@app.command(name="loop")
def active_loop(
    cve: Optional[str] = typer.Option(None, "--cve"),
    steps: int = typer.Option(3, "--steps"),
    write_back: bool = typer.Option(False, "--write-back/--no-write-back"),
    use_ollama: bool = typer.Option(False, "--ollama/--no-ollama", help="Use local Ollama for question gen"),
    early_stop: bool = typer.Option(True, "--early-stop/--no-early-stop", help="Stop if evidence already exists"),
    assets: Optional[str] = typer.Option(None, "--assets", help="Path to assets JSON (optional)"),
) -> None:
    """Minimal uncertainty→question→Wikipedia search loop.

    Picks a slot by EIG, renders a question (template), searches Wikipedia with CVE as query (or question text),
    and optionally writes back a toy Fact with the top result's title.
    """
    from env.interaction import AskEnv
    signals = get_cve_graph_signals(cve) if cve else None
    resolved = resolve_slots_from_signals(signals) if signals else {}
    # Optional asset context for template filling
    asset_name: Optional[str] = None
    if assets:
        try:
            import json as _json
            with open(assets, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if isinstance(data, dict) and isinstance(data.get("assets"), list) and data["assets"]:
                first = data["assets"][0]
                asset_name = str(first.get("name") or first.get("id") or "asset-1")
            elif isinstance(data, list) and data:
                first = data[0]
                asset_name = str(first.get("name") or first.get("id") or "asset-1") if isinstance(first, dict) else str(first)
        except Exception:
            asset_name = None
    env = AskEnv(signals=signals, error_rate=0.1)
    log = []
    asked: list[str] = []
    for i in range(max(1, steps)):
        already_true = tuple(k for k, v in (resolved or {}).items() if bool(v))
        # Choose a candidate slot first
        candidate = pick_next_slot_by_eig(signals=signals, resolved_slots=resolved, exclude=tuple(asked) + already_true)
        # Skip slots already known in graph (if we have a concrete subject)
        if cve and candidate in ("actively_exploited", "epss_high", "internet_exposed"):
            if has_slot_value(entity_name=cve, slot=candidate, min_confidence=0.7):
                asked.append(candidate)
                continue
        slot = candidate
        question = render_from_domain_pack(slot, context={"cve_id": cve or "unknown", "asset_name": asset_name or "the asset", "use_ollama": use_ollama})
        # Private vs public slots: ask-only for private; search for public
        private_slots = {"internet_exposed", "asset_present", "business_critical"}
        public_slots = {"actively_exploited", "epss_high", "patch_available", "workaround_available"}
        top = None
        if slot in public_slots:
            # Build a slightly more specific query to avoid always hitting the CVE root page
            query = cve or question
            if cve:
                if slot == "actively_exploited":
                    query = f"{cve} CISA Known Exploited Vulnerabilities"
                elif slot == "epss_high":
                    query = f"{cve} EPSS"
                elif slot == "patch_available":
                    query = f"{cve} vendor advisory patch"
                elif slot == "workaround_available":
                    query = f"{cve} vendor advisory workaround"
            docs = search_wikipedia(query=query, k=1)
            top = docs[0] if docs else None
        else:
            query = question
        entry = {"step": i + 1, "slot": slot, "question": question, "query": query, "top": (top.metadata if top else None)}
        if write_back and top is not None:
            title = top.metadata.get("title") or "Wikipedia"
            url = top.metadata.get("url") or ""
            if not fact_exists(subject_name=cve or "Unknown", relation_name="RELATED_TO", object_name=title):
                wb = upsert_fact_with_source(subject_name=cve or "Unknown", relation_name="RELATED_TO", object_name=title, source_url=url)
            else:
                wb = {"skipped": True, "reason": "exists", "object": title}
            # also create lexical scaffolding and a MENTIONS link to the subject
            try:
                from db.writeback import upsert_document_with_sentence, link_sentence_mentions_entity
                sent = upsert_document_with_sentence(source_url=url or f"https://en.wikipedia.org/wiki/{title}", title=title, text_excerpt=top.text[:180])
                if sent is not None:
                    link_sentence_mentions_entity(source_url=sent["source_url"], sentence_order=1, entity_name=cve or title)
                entry["write_back_lex"] = sent
            except Exception:
                pass
            entry["write_back"] = wb
        log.append(entry)
        asked.append(slot)
        if early_stop and isinstance(entry.get("write_back"), dict):
            # Stop when we either wrote a new fact or confirmed one exists
            if (not entry["write_back"].get("skipped")) or (entry["write_back"].get("reason") == "exists"):
                break
    console.print({"cve": cve, "log": log})


@app.command(name="bootstrap")
def bootstrap() -> None:
    """Apply schema/bootstrap.cypher to Neo4j (constraints and indexes)."""
    try:
        from db.bootstrap import apply_bootstrap
    except Exception as exc:
        console.print(f"Bootstrap unavailable: {exc}")
        raise typer.Exit(code=1)
    count = apply_bootstrap()
    if count == 0:
        console.print("Neo4j not configured or driver missing.")
        raise typer.Exit(code=1)
    console.print({"statements_applied": count})


@app.command(name="loop-emg")
def loop_emg(
    episodes: Path = typer.Option(Path("data/emg/movie_episodes.jsonl"), "--episodes", help="Path to EMG episodes JSONL"),
    steps: int = typer.Option(3, "--steps", help="Max steps per episode"),
    write_back: bool = typer.Option(True, "--write-back/--no-write-back"),
    save_json: Optional[Path] = typer.Option(None, "--save-json", help="Save per-episode results to JSONL"),
    save_csv: Optional[Path] = typer.Option(None, "--save-csv", help="Save per-episode results to CSV"),
    ollama: bool = typer.Option(False, "--ollama/--no-ollama", help="Use local LLM (Ollama) to generate questions"),
) -> None:
    """Run a tiny EMG loop over episodes (identify film) and report steps/asks/searches.

    Each episode: {"task":"identify_film","target":"Spirited Away","known":{...}}
    """
    import json as _json

    def _iter_jsonl(p: Path):
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                yield _json.loads(ln)

    total = 0
    total_steps = 0
    total_asks = 0
    total_searches = 0
    successes = 0
    rows: list[dict[str, Any]] = []

    for ep in _iter_jsonl(episodes):
        task = ep.get("task")
        target = str(ep.get("target"))
        if task == "identify_film":
            known = ep.get("known") or {}
            asked: list[str] = []
            steps_done = 0
            asks_done = 0
            searches_done = 0
            trace: list[Dict[str, Any]] = []
            # Try graph filter for known slots; if only one candidate remains, verify via retrieval
            candidates = find_films_by_slots(known)
            if len(candidates) == 1 and candidates[0].lower() == target.lower():
                from agent.question_generator import llm_generate_search_query as _llm_sq
                ver_q = _llm_sq(
                    slot="title",
                    known=known,
                    last_question="Guess what film I'm thinking.",
                    context={"llm_model": "gemma:4b"},
                    use_ollama=ollama,
                    llm_model="gemma:4b",
                )
                docs = search_wikipedia(query=ver_q, k=1)
                top = docs[0] if docs else None
                verified = bool(top and target.lower() in (top.metadata.get("title") or "").lower())
                steps_done = 1
                asks_done = 0
                searches_done = 1
                trace = [
                    {"action": "early_success", "reason": "unique_graph_match", "candidates": candidates},
                    {
                        "action": "verification_search",
                        "query": ver_q,
                        "top_title": (top.metadata.get("title") if top else None),
                        "top_url": (top.metadata.get("url") if top else None),
                    },
                ]
                total += 1
                total_steps += steps_done
                total_asks += asks_done
                total_searches += searches_done
                rows.append({
                    "task": task,
                    "target": target,
                    "success": bool(verified),
                    "steps": steps_done,
                    "asks": asks_done,
                    "searches": searches_done,
                    "trace": trace,
                })
                if verified:
                    successes += 1
                continue
            # Derive candidate slots dynamically from graph schema
            type_names = get_entity_types(target)
            derived: list[str] = []
            for t in type_names or ["Film"]:
                for s in list_slots_for_type(t):
                    if s not in derived:
                        derived.append(s)
            if not derived:
                derived = list_common_slots(limit=8)
            unknowns = [s for s in derived if s not in known]
            # Step loop
            success = False
            for _ in range(max(1, steps)):
                # If we still have unknowns, ask one; otherwise try a search
                if unknowns:
                    slot = unknowns.pop(0)
                    asked.append(slot)
                    asks_done += 1
                    # Generate a question for the slot (LLM when enabled)
                    try:
                        from agent.question_generator import llm_generate_question as _llm_q
                        ctx = {"llm_model": "gemma:4b", "task": task, "known_keys": list(known.keys())}
                        qtext = _llm_q(slot, context=ctx, use_ollama=ollama)
                    except Exception:
                        qtext = f"Provide the film's {slot}."
                    trace.append({"action": "ask", "slot": slot, "question": qtext})
                    # Auto-respond from graph when possible to simulate a user reply
                    try:
                        from emg.query import get_slot_value as _get_sv
                        ans = _get_sv(target, slot)
                    except Exception:
                        ans = None
                    if ans is not None:
                        known[slot] = str(ans)
                        trace.append({"action": "answer", "slot": slot, "value": known[slot], "source": "graph"})
                # Always search Wikipedia using an LLM-generated query from the last asked question and known slots
                from agent.question_generator import llm_generate_search_query as _llm_sq
                last_q = trace[-1]["question"] if trace and trace[-1].get("action") == "ask" else "Guess what film I'm thinking."
                query = _llm_sq(slot=slot, known=known, last_question=last_q, context={"llm_model": "gemma:4b"}, use_ollama=ollama, llm_model="gemma:4b")
                docs = search_wikipedia(query=query, k=1)
                top = docs[0] if docs else None
                searches_done += 1
                steps_done += 1
                trace.append({
                    "action": "search",
                    "query": query,
                    "top_title": (top.metadata.get("title") if top else None),
                    "top_url": (top.metadata.get("url") if top else None),
                })
                if top is not None and write_back:
                    title = top.metadata.get("title") or "Wikipedia"
                    url = top.metadata.get("url") or ""
                    if not fact_exists(subject_name=target, relation_name="RELATED_TO", object_name=title):
                        upsert_fact_with_source(subject_name=target, relation_name="RELATED_TO", object_name=title, source_url=url)
                        trace.append({"action": "write_back", "object": title, "url": url, "status": "created"})
                    else:
                        trace.append({"action": "write_back", "object": title, "url": url, "status": "exists"})
                    # Consider success if the top title contains the target name (very simple heuristic)
                    if target.lower() in title.lower():
                        success = True
                        break
            total += 1
            total_steps += steps_done
            total_asks += asks_done
            total_searches += searches_done
            rows.append({"task": task, "target": target, "success": bool(success), "steps": steps_done, "asks": asks_done, "searches": searches_done, "trace": trace})
            if success:
                successes += 1
        elif task == "retrieve_award":
            # Heuristic: search for awards page and write back RELATED_TO; success if title/url hints at awards
            steps_done = 0
            searches_done = 0
            asks_done = 0
            trace: list[Dict[str, Any]] = []
            query = f"{target} awards"
            docs = search_wikipedia(query=query, k=1)
            top = docs[0] if docs else None
            searches_done += 1
            steps_done += 1
            trace.append({
                "action": "search",
                "query": query,
                "top_title": (top.metadata.get("title") if top else None),
                "top_url": (top.metadata.get("url") if top else None),
            })
            success = False
            if top is not None and write_back:
                title = top.metadata.get("title") or "Wikipedia"
                url = top.metadata.get("url") or ""
                if not fact_exists(subject_name=target, relation_name="RELATED_TO", object_name=title):
                    upsert_fact_with_source(subject_name=target, relation_name="RELATED_TO", object_name=title, source_url=url)
                    trace.append({"action": "write_back", "object": title, "url": url, "status": "created"})
                else:
                    trace.append({"action": "write_back", "object": title, "url": url, "status": "exists"})
                if ("award" in (title or "").lower()) or ("award" in (url or "").lower()):
                    success = True
            total += 1
            total_steps += steps_done
            total_asks += asks_done
            total_searches += searches_done
            rows.append({"task": task, "target": target, "success": bool(success), "steps": steps_done, "asks": asks_done, "searches": searches_done, "trace": trace})
            if success:
                successes += 1
        elif task == "qa_fact":
            # QA over a known slot: {task:"qa_fact", target:"Spirited Away", slot:"Country", expect_value:"Japan"}
            slot = str(ep.get("slot") or "")
            expect_value = str(ep.get("expect_value") or "")
            steps_done = 0
            asks_done = 0
            searches_done = 0
            trace: list[Dict[str, Any]] = []
            val = None
            try:
                from emg.query import get_slot_value as _get_sv
                val = _get_sv(target, slot)
            except Exception:
                val = None
            trace.append({"action": "read_graph", "slot": slot, "value": val})
            success = bool(val and expect_value and str(val).lower() == expect_value.lower())
            # If not found, attempt a single search (no parsing yet), for provenance/write-back only
            if not success:
                query = f"{target} {slot}"
                docs = search_wikipedia(query=query, k=1)
                top = docs[0] if docs else None
                searches_done += 1
                steps_done += 1
                trace.append({
                    "action": "search",
                    "query": query,
                    "top_title": (top.metadata.get("title") if top else None),
                    "top_url": (top.metadata.get("url") if top else None),
                })
                if top is not None and write_back:
                    title = top.metadata.get("title") or "Wikipedia"
                    url = top.metadata.get("url") or ""
                    if not fact_exists(subject_name=target, relation_name="RELATED_TO", object_name=title):
                        upsert_fact_with_source(subject_name=target, relation_name="RELATED_TO", object_name=title, source_url=url)
                        trace.append({"action": "write_back", "object": title, "url": url, "status": "created"})
                    else:
                        trace.append({"action": "write_back", "object": title, "url": url, "status": "exists"})
            total += 1
            total_steps += steps_done
            total_asks += asks_done
            total_searches += searches_done
            rows.append({"task": task, "target": target, "slot": slot, "success": bool(success), "steps": steps_done, "asks": asks_done, "searches": searches_done, "trace": trace})
            if success:
                successes += 1
        elif task == "procedure":
            # Minimal procedure: compute a score from AwardsSignal and Studio
            # Preconditions: Studio, AwardsSignal present as slots
            steps_done = 0
            asks_done = 0
            searches_done = 0
            trace: list[Dict[str, Any]] = []
            needed = ["Studio", "AwardsSignal"]
            from emg.query import get_slot_value as _get_sv
            have: Dict[str, str] = {}
            missing: list[str] = []
            for s in needed:
                v = _get_sv(target, s)
                if v:
                    have[s] = str(v)
                else:
                    missing.append(s)
            if missing:
                for s in missing:
                    trace.append({"action": "ask", "slot": s})
                    asks_done += 1
            # Score: AwardsSignal High=3, Medium=2, Low=1; Studio known +1
            score = 0
            sig = have.get("AwardsSignal")
            if sig:
                score += {"High": 3, "Medium": 2, "Low": 1}.get(sig, 1)
            if have.get("Studio"):
                score += 1
            success = bool(score > 0)
            steps_done = asks_done  # each ask counted as a step here
            total += 1
            total_steps += steps_done
            total_asks += asks_done
            total_searches += searches_done
            rows.append({"task": task, "target": target, "success": bool(success), "steps": steps_done, "asks": asks_done, "searches": searches_done, "score": score, "missing": missing, "trace": trace})
            if success:
                successes += 1

    summary = {
        "episodes": total,
        "successes": successes,
        "success_rate": round((successes / total) if total else 0.0, 3),
        "avg_steps": round((total_steps / total) if total else 0.0, 3),
        "avg_asks": round((total_asks / total) if total else 0.0, 3),
        "avg_searches": round((total_searches / total) if total else 0.0, 3),
    }
    console.print(summary)
    # Optional saves
    if save_json:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with save_json.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(_json.dumps(r) + "\n")
    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with save_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["task", "target", "slot", "success", "steps", "asks", "searches", "score", "trace"]
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in fieldnames}
                # stringify trace for CSV readability
                if isinstance(row.get("trace"), list):
                    row["trace"] = _json.dumps(row["trace"])  # type: ignore[index]
                w.writerow(row)


@app.command(name="emg-seed")
def emg_seed(
    path: Path = typer.Option(Path("data/emg/seed_entities.jsonl"), "--path", help="Path to EMG seed JSONL"),
) -> None:
    """Load EMG seed entities (Entities with Type and SlotValues)."""
    try:
        num = load_seed_entities(path)
    except Exception as exc:
        console.print({"loaded": 0, "error": str(exc)})
        raise typer.Exit(code=1)
    console.print({"loaded": num, "path": str(path)})


@app.command(name="emg-interactive")
def emg_interactive(
    steps: int = typer.Option(10, "--steps", help="Max turns"),
    ollama: bool = typer.Option(False, "--ollama/--no-ollama", help="Use local LLM (Ollama) for question/query gen"),
    llm_model: str = typer.Option("gemma:4b", "--llm-model", help="Model tag when using Ollama"),
    show_candidates: bool = typer.Option(True, "--show-candidates/--hide-candidates", help="Show candidate films each turn"),
    k: int = typer.Option(1, "--k", help="Docs to preview on retrieval"),
) -> None:
    """Interactive loop for identifying a film via human Q&A.

    Flow:
    - You type your own initial question.
    - Assistant picks a slot, generates a natural question, and proposes a short search query.
    - You answer; we update known slot-values and show evolving candidates.
    - Stops when unique candidate remains or steps exhausted.
    Commands during answer: /skip, /done, /guess
    """
    from agent.question_generator import llm_generate_question as _llm_q
    from agent.question_generator import llm_generate_search_query as _llm_sq

    console.rule("Interactive: Guess the film")
    user_q = input("You: ").strip()
    if not user_q:
        user_q = "Guess what film I'm thinking."
    console.print({"user_question": user_q})

    # Known slot-values accumulate from your answers
    known: Dict[str, str] = {}
    trace: list[Dict[str, Any]] = []
    chosen_slots: list[str] = []

    # Candidate slot schema derived from graph for Type 'Film'
    derived_slots = list_slots_for_type("Film") or list_common_slots(limit=8)

    for turn in range(1, max(1, steps) + 1):
        # Compute candidates given what we know so far
        candidates = find_films_by_slots(known)
        if show_candidates:
            console.print({"turn": turn, "candidates": candidates[:5], "candidates_total": len(candidates)})
        if len(candidates) == 1:
            console.rule("Guess")
            console.print({"film": candidates[0], "reason": "unique match from provided answers"})
            break

        # Pick next slot: prefer most common slots first, excluding answered
        next_slot = next((s for s in derived_slots if s not in known and s not in chosen_slots), None)
        if next_slot is None:
            # Nothing left to ask; stop
            console.print({"done": True, "reason": "no more informative slots"})
            break
        chosen_slots.append(next_slot)

        # Generate a natural question
        qctx = {"llm_model": llm_model, "task": "identify_film", "known_keys": list(known.keys())}
        try:
            qtext = _llm_q(next_slot, context=qctx, use_ollama=ollama)
        except Exception:
            qtext = f"Could you share more about the {next_slot}?"
        console.rule(f"Step {turn}")
        console.print({"slot": next_slot, "question": qtext})
        trace.append({"action": "ask", "slot": next_slot, "question": qtext})

        # Get your answer or a command
        ans = input("Your answer (/skip, /done, /guess): ").strip()
        if ans.lower() == "/done":
            console.print({"stopped": True, "reason": "user"})
            break
        if ans.lower() == "/guess":
            console.rule("Current Guess Candidates")
            console.print(candidates[:10])
            # Continue loop without consuming a turn
            chosen_slots.pop()
            continue
        if ans.lower() != "/skip" and ans:
            known[next_slot] = ans
            trace.append({"action": "answer", "slot": next_slot, "value": ans, "source": "user"})

        # Generate a short retrieval query from the latest question + known
        try:
            query = _llm_sq(slot=next_slot, known=known, last_question=qtext, context={"llm_model": llm_model}, use_ollama=ollama, llm_model=llm_model)
        except Exception:
            # Heuristic join of known values
            query = " ".join({v for v in known.values() if v}) or qtext
        docs = search_wikipedia(query=query, k=max(1, k))
        top = docs[0] if docs else None
        trace.append({
            "action": "search",
            "query": query,
            "top_title": (top.metadata.get("title") if top else None),
            "top_url": (top.metadata.get("url") if top else None),
        })
        console.print({"query": query, "top": (top.metadata if top else None)})

    console.rule("Session Summary")
    console.print({"known": known, "turns": len(trace), "trace": trace})

if __name__ == "__main__":
    app()


