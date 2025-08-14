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
from db.writeback import upsert_fact_with_source, fact_exists
from agent.question_generator import render_question_from_slot, render_from_domain_pack
from agent.question_selector import pick_next_slot_by_eig


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
) -> None:
    """Minimal uncertainty→question→Wikipedia search loop.

    Picks a slot by EIG, renders a question (template), searches Wikipedia with CVE as query (or question text),
    and optionally writes back a toy Fact with the top result's title.
    """
    from env.interaction import AskEnv
    signals = get_cve_graph_signals(cve) if cve else None
    env = AskEnv(signals=signals, error_rate=0.1)
    log = []
    asked: list[str] = []
    for i in range(max(1, steps)):
        slot = pick_next_slot_by_eig(signals=signals, resolved_slots=None, exclude=tuple(asked))
        question = render_from_domain_pack(slot, context={"cve_id": cve or "unknown", "use_ollama": use_ollama})
        query = cve or question
        docs = search_wikipedia(query=query, k=1)
        top = docs[0] if docs else None
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

if __name__ == "__main__":
    app()


