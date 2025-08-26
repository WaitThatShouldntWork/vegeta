# %% [markdown]
# # Future Work: Memory-Augmented Planner–Executor atop Bayesian Decider
# source: https://arxiv.org/pdf/2502.01562
# This document sketches concrete next steps to incorporate episodic case-based memory, a learned
# retrieval policy, and a planner–executor loop while preserving the Bayesian decider’s likelihood-first
# scoring and EIG-based meta-control.

# Best path: merge them. Keep your likelihood/EIG as the “truth meter,” use their case memory to adjust priors and plans from outcomes, and gate answers with your uncertainty while letting the planner–executor handle OOD tool use

#How it collapses ambiguity (and its key weakness, huge upfront inference time. our bayesian system should collapse ambiguity MUCH quicker.
# additionally, we're storing all information in the graph (rather than just the case memories) - as we've witnessed, we have huge performance gains on repeated runs.

# How it collapses ambiguity
# Plan→Execute→Evaluate→Replan: runs short loops; if not solved, it fetches more discriminative evidence (search/crawl/code), updates Subtask/Tool Memory, and tries again.
# Case-guided planning: when cases exist, TopK retrieved plans bias decomposition and tool choices, narrowing the branch factor.
# Learned retrieval: a parametric Q(s,c) ranks cases by expected utility, further reducing search.
# Cold start (no previous solved cases)
# K=0 fallback: planner runs zero-shot CoT planning with tools (their ablations report K=0 baselines), executes, evaluates outcome.
# First run writes a case (success or failure). Future similar tasks then benefit from retrieval.
# Until memory accrues, progress relies on the base LLM’s planning quality and tool feedback loops, not uncertainty estimates.

# %% [markdown]
# ## Objectives
# - Add an episodic Case Bank with Write/Read to enable continual, non-parametric learning.
# - Learn a parametric case-retrieval scorer Q(s, c; θ) online (binary outcome), used for TopK selection.
# - Split into Planner (case-guided plan generation) and Executor (tool calls), with per-subtask memories.
# - Formalize a minimal Memory-augmented MDP (M-MDP) abstraction to log state, action, reward.
# - Use retrieved cases to tilt priors and EIG thresholds in the existing Bayesian pipeline (no model finetune).
# - Evaluate continual gains over iterations; control K (≈4) to avoid swamping.

# %% [markdown]
# ## Scope and Non-Goals
# - Scope: Planning and retrieval policies; memory curation; tool protocol; evaluation harness.
# - Non-goals: Training base LLMs; replacing the Bayesian likelihood; complex RL beyond online case scoring.

# %% [markdown]
# # 1) Memory: Case Bank (Episodic)
# We store compact cases at task completion to guide future planning.
#
# Schema (per case):
# - s: state summary (task text + key context; embedded for retrieval)
# - a: plan (decomposition outline; executor-facing subtasks)
# - r: outcome (binary success/failure; optional partial score)
# - trace_id/provenance: minimal pointer to logs
#
# Operations:
# - Write(s, a, r): append after task completes (keep both successes and failures)
# - ReadNP(s, K): TopK by cosine(sim(enc(s), enc(si)))
# - ReadP(s, K): TopK by learned Q(s, ci; θ) with θ updated online (single-step CE on r)
#
# Curation:
# - Cap size (e.g., 5k–20k); maintain recency and diversity; decay low-utility cases
# - Prefer small K (≈4) per step, per ablations (quality over quantity)

# %% [markdown]
# # 2) Minimal Memory‑Augmented MDP (M‑MDP)
# Tuple ⟨S, A, P, R, γ, M⟩ used for logging and analysis, not heavy RL.
# - State s: user task + short history + current evidence summary
# - Action a: planner’s plan (or executor tool call), but we focus on planner’s retrieval/plan
# - Reward r: task success (EM/PM/slot completeness/posterior test)
# - Memory M: episodic Case Bank
# - Policy: π(a|s, M) via planner conditioned on K retrieved cases
# - Retrieval policy: µ(c|s, M) ∝ softmax(Q(s, c; θ)/α), but we operationalize as TopK for stability

# %% [markdown]
# # 3) Planner–Executor Split
# Planner (LLM):
# - Input: (task, history, retrieved cases K)
# - Output: decomposed plan (subtasks), decision to replan/stop
# - Uses cases to bias structure/templates and to highlight discriminative evidence to seek
#
# Executor (tools):
# - Executes subtasks; logs Tool Memory per subtask (inputs, outputs, provenance)
# - Supported tools (initial): graph/Neo4j, web search, crawl/fetch, code, math; standardize I/O schemas
#
# Subtask Memory:
# - Append-only list of (subtask, result, status); used by planner for replanning

# %% [markdown]
# # 4) Integration with Bayesian Core (Single Source of Truth for Scoring)
# - Use retrieved cases to set prior nudges: p(z_checklist), p(z_goal), p(z_subgraph), σ², τ
# - Feed case-derived lexicon/slots into expected u′_terms/u′_struct for better δ_terms/δ_struct alignment
# - Ask/Search:
#   - Rank ASK slots by H_r × discriminativeness across retrieved cases (split top-2)
#   - Build targeted SEARCH queries from disagreements between cases and current top subgraphs
# - Keep candidate ranking tied to the existing likelihood: log p(u|v) with δ_sem/δ_struct/δ_terms

# %% [markdown]
# # 5) Outcome Signals and Rewarding
# - Success (r=1): EM/PM thresholds met; or posterior confidence ≥ θ and posterior predictive check passes
# - Failure (r=0): incorrect; or posterior predictive check fails; or budget exceeded
# - Partial credit: record PM/slot-completeness for analysis, but keep r binary for Q training

# %% [markdown]
# # 6) Memory Read/Write Policy Details
# Defaults:
# - Encoder: SimCSE or equivalent for s
# - K: 4 (tune 2–8)
# - Selection: TopK (non-param) → TopK(Q) (param)
# - Write only final state/action after completion to reduce redundancy
# - Balance successes and failures; avoid swamping by near-duplicate failures

# %% [markdown]
# # 7) Continual Learning Loop (Iterations)
# For a small benchmark suite:
# - Iteration i: run tasks with current Case Bank; Write new cases; update Q(s, c; θ)
# - Report curves: F1/PM/EM per dataset; improvement vs w/o CBR; OOD tests
# - Monitor K sensitivity; token/tool cost; average subtasks/tool calls per task

# %% [markdown]
# # 8) Tool Protocol and Safety
# - Standardize tool signatures (JSON schemas), timeouts, retry/backoff
# - Whitelist code execution libs; sandbox; persist artifacts per task
# - Per-subtask Tool Memory for reproducibility and audits

# %% [markdown]
# # 9) Calibration and Cost Control
# - Confidence head: map [q_top1, margin, H, completeness, provenance] → predicted accuracy
# - Adjust θ_conf and EIG thresholds by predicted accuracy and budget
# - Track tokens and tool costs; enforce soft caps per level/dataset

# %% [markdown]
# # 10) Evaluation Plan
# Datasets (initial):
# - Internal graph tasks (entity linking, fact verification, subgraph identification)
# - A small web QA subset for OOD checks (3–5 categories)
#
# Protocol:
# - Baselines: Offline executor; Online tools only; Planner without CBR; Planner with non-param CBR; with param CBR
# - Metrics: EM/PM/F1; cost; tool calls; pass@1; learning curves over 3–5 iterations

# %% [markdown]
# # 11) Risks and Mitigations
# - Case swamping → cap size, deduplicate, diversify, prefer TopK small
# - Misleading failures → downweight via Q(s, c; θ) trained on r
# - Planner verbosity → enforce concise templates; short plans with clear subtasks
# - Tool instability → retries, caching, provenance scoring

# %% [markdown]
# # 12) Minimal MVP (1–2 sprints)
# - Non-param Case Bank (Write/ReadNP) + K=4
# - Planner that concatenates TopK cases to produce a plan
# - Executor for graph + web search + crawl; Subtask/Tool Memory logs
# - Rewarding with EM/PM threshold + posterior predictive check gate
# - One iterative run showing improvement vs no-case baseline

# %% [markdown]
# # 13) Stretch: Parametric Case Selection
# - Q(s, c; θ): two-layer MLP on [enc(s) ⊕ enc(case_s), simple overlaps/features]
# - Train online with binary CE on outcome r; TopK by Q for stability
# - Compare to kernel/episodic control variant if needed later

# %% [markdown]
# # 14) Checklist
# - [ ] Case Bank: schema + Write/ReadNP
# - [ ] Planner–Executor separation + per-subtask memory
# - [ ] Tool protocol (graph, search, crawl, code, math) + logs
# - [ ] Prior nudges and ASK/SEARCH hooks from cases
# - [ ] Rewarding and posterior predictive check gate
# - [ ] Iteration harness + metrics + curves
# - [ ] Parametric Q(s, c; θ) (stretch)


