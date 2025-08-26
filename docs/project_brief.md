# Project Brief — VEGETA

VEGETA is a Bayesian Active Inference decider that chooses among Answer / Ask / Search using Expected Information Gain (EIG). It uses simple, interpretable likelihood channels over candidate subgraphs (semantic, structure, terms) rather than a heavy neural decoder. Domain‑agnostic; current demo focuses on a generalisable model, starting with movies. See `README.md` for usage; see `TODO.md` for live tasks.

## Getting Started (succinct)

- Setup (PowerShell):
  - `scripts/setup.ps1 -Dev`
  - Activate: `.venv\Scripts\Activate.ps1`
- Run demo CLI:
  - `vegeta decide --domain cyber --cve CVE-2025-TEST`
  - Search preview: `vegeta decide --domain cyber --cve CVE-2025-TEST --force-action search --retrieve --top-k 2`
- Run tests:
  - `scripts/test.ps1`

## Overview of how the brain currently works

In the Bayesian brain framing, the brain maintains beliefs about hidden causes $v$ of sensory data $u$. It continually generates predictions $u'$ from those beliefs via a generative model $p(u|v)$, compares them to actual sensations $u$, and updates beliefs to reduce prediction error. This runs hierarchically: higher levels predict lower levels; lower levels send back residual errors. Learning adjusts parameters so that predictions get less wrong over time.

Mathematically, exact Bayesian inference over $p(v|u)$ is intractable for real brains and real machines. So we use an approximate posterior $q_\phi(v)$ with parameters $\phi$ and optimize a tractable bound on surprise, the variational free energy:

$$F(\phi) = KL[q_\phi(v) \| p(v)] - E_{q_\phi}[\log p(u|v)]$$

Minimizing $F$ is equivalent to maximizing the ELBO. Intuition: keep your beliefs close to the prior unless the data strongly argues otherwise, and make your predictions explain the data.

## Proposed process flow of EIG

Your agent chooses among actions $a \in \{\text{answer}, \text{ask}, \text{search}\}$ using Expected Information Gain (EIG):

1. Maintain a belief over hypotheses $\Theta$ (e.g., which procedure step, which policy branch, which entity). Current entropy $H(\Theta)$ quantifies uncertainty.

2. For each action $a$, predict possible observations $O$ and their probabilities $p(o|a,D)$.

3. Compute expected posterior entropy $E_o[H(\Theta|D,o)]$.

4. EIG is the expected drop in entropy:

$$EIG(a) = H(\Theta|D) - E_o[H(\Theta|D,o)] = I(\Theta;O|a,D)$$

5. Subtract a cost term $c(a)$ for latency, tokens, user annoyance, or risk. Pick the argmax. If confidence in the current best answer already exceeds a threshold, skip the dance and answer.

## Breakdown of UE components

Ship the simplest UX that makes the policy legible:

- **Belief panel**: current top hypotheses with probabilities and entropy
- **Action bar**: scores for Answer, Ask, Search showing EIG, cost, and net
- **Question card**: the next user question, with why it was chosen (expected split)
- **Search trace**: retrieval snippets with provenance and a "recency/risk" badge
- **Procedure path**: current step in the domain procedure graph and next candidates
- **Cost dials**: tunable weights for latency, tokens, user annoyance, risk
- **Session graph**: tiny DAG of actions and observations, to debug decision paths

## Keywords

Bayesian Brain Model; Cognitive Bayesianism; Predictive Coding; Predictive Processing; Free Energy Principle; Variational Free Energy; ELBO; Expected Information Gain; Active Inference; KL Divergence; Hierarchical Generative Models.

## From retrieval to $v$ and predicted $u'$ (current design)

We use a simple, interpretable pipeline over a working‑set subgraph:

- Latents $v$ (factorized): $z_{checklist}$, $z_{goal}$, $z_{subgraph}$, $\{z_{slot_r}\}$, $z_{novelty}$.
- Selective activation (anchors → candidates):
  - Embed the utterance; retrieve k‑NN anchors by semantic and graph embeddings; combine and take top‑K.
  - Expand each anchor into small candidate subgraphs (fixed hops or PPR) and keep top‑M.
  - Define retrieval context $R$ = anchors, candidates, parameters, provenance.
- Predicted channels $u'$ for a candidate $S_j$:
  - $u'_\text{sem}(S_j)$: pooled node embeddings concatenated with a small structure vector; then normalize.
  - $u'_\text{struct}(S_j)$: expected slot/edge counts from the active checklist (what should exist).
  - $u'_\text{terms}(S_j)$: expected key terms from node names/aliases, relation words, and checklist lexicon (optionally LLM‑augmented).
- Observation channels $u$:
  - $u_\text{sem}$: utterance embedding; $u_\text{terms}$: extracted keywords/entities; $u_\text{struct\_obs}(S_j)$: measured counts within $S_j$.
- Likelihood distances per channel:
  - $\delta_{sem} = 1 - \text{cosine}(u_{sem}, u'_{sem})$
  - $\delta_{struct} = \|u_{struct\_obs}(S_j) - u'_{struct}(S_j)\|$ with penalties for required‑missing
  - $\delta_{terms} = 1 - \text{JaccardOverlap}$ (or cosine over term‑embeddings)
- Combined log‑likelihood used everywhere (single source of truth):

  $\log p(u\mid v=S_j) = -\left[\alpha\,\frac{\delta_{sem}}{\sigma_{sem}^2} + \beta\,\frac{\delta_{struct}}{\sigma_{struct}^2} + \gamma\,\frac{\delta_{terms}}{\sigma_{terms}^2} + \text{penalties}\right] + C$

- Priors:
  - $p(z_{checklist})$ from history and frequencies; $p(z_{goal}\mid z_{checklist})$ via a small compatibility table.
  - $p(z_{subgraph}\mid R)$ from retrieval softmax, simplicity, provenance, recency.
  - $p(z_{slot_r})$ from local popularity/type constraints with an "unknown" option.
  - $p(z_{novelty})$ from OOD signals (low similarity, distance from clusters, poor checklist fit).
- Useful defaults: K anchors ≈ 10; M candidates ≈ 20; hops=2 (or PPR top‑N); $\alpha=1.0,\ \beta=0.5,\ \gamma=0.3$; $\sigma^2_{sem}=0.3,\ \sigma^2_{struct}=0.2,\ \sigma^2_{terms}=0.2$; temperature $\tau\in[0.5,1.0]$.

## Variational algorithm, approximate Bayesian inference

Exact Bayes is unmanageable. We approximate with factorized $q(v)$ and minimize free energy:

$$F = KL[q(v) \| p(v)] - E_{q}[\log p(u|v)]$$

In practice, we reweight candidates by $p(u\mid v)\,p(v)$ and normalize (softmax with temperature $\tau$) to obtain $q(v)$; no heavy encoder/decoder is required.

## KL divergence, Shannon angle

KL is relative entropy:

$$KL(q \| p) = \sum_v q(v) \log \frac{q(v)}{p(v)}$$

In "20 questions" terms, entropy tells you the minimum yes/no questions to pin down the truth on average. EIG is how many questions you save by taking an action now.

## Why minimize free energy?

1. **Explain sensory data with your beliefs (fit)**
2. **Keep beliefs simple unless data demands complexity (Occam)**
3. **Make action choice principled**: Active Inference adds actions that are expected to minimize future free energy. In your agent, that's asking a clarifying question or launching a retrieval to reduce uncertainty before acting.

## Active inference in this setup

Action is another knob to drive down expected future free energy. Your policy chooses:

- **Ask** when private variables dominate uncertainty and a question splits the hypothesis set well
- **Search** when public facts dominate and are likely to change or be missing locally  
- **Answer** when expected EIG is low and current confidence is high relative to the cost of delay

### Procedure‑driven next‑best‑question (gating)

Rather than asking generic high‑EIG slots, prefer questions that unblock the current Procedure Step. Model Steps with preconditions (`:Step-[:REQUIRES]->(:SlotValue or Condition)`), and at runtime ask only for missing preconditions. Public preconditions (e.g., “patch available?”) route to Search; private ones (e.g., “internet exposed?”) route to Ask. This focuses interaction on “what unlocks progress next,” reduces question count, and yields logs that can be distilled into a fast‑and‑frugal policy tree.

## Practical recipe (current)

- Build working set via selective activation: utterance → anchors → candidate subgraphs; record retrieval context and provenance.
- Compute per‑candidate features and predicted channels $u'$; measure observed $u_{struct\_obs}$.
- Form priors over $z_{checklist}$, $z_{goal}$, $z_{subgraph}$, $\{z_{slot_r}\}$, $z_{novelty}$.
- Compute $\log p(u\mid v)$ via the channel distances; obtain $q(v) \propto p(u\mid v)\,p(v)$.
- Estimate EIG:
  - Ask: simulate a few likely slot answers; average entropy drop.
  - Search: assume retrieval confirms/denies top hypotheses per calibrated rates; estimate entropy drop.
  - Answer: current confidence minus risk.
- Choose action by Utility = expected_gain − cost − risk; enforce sensible thresholds (confidence, entropy, margin).
- Persist session state: truncated $q(z_{subgraph})$, slot posteriors, novelty flag, retrieval trace, and a short conversation summary.

## Simulated user (explained simply)

- Think of 20 Questions. Our system can ask yes/no questions to reduce uncertainty.
- For testing, we don't need a human every time. We build a tiny "fake user" that answers based on ground-truth we already have in each scenario (e.g., a CVE marked as exploited). To keep it realistic, the fake user is a bit noisy: it answers wrong sometimes (small error rate).
- Each time the policy chooses Ask(slot), the simulator returns "yes" or "no" with that small chance of error; our Bayesian update adjusts beliefs; we continue until we Answer or hit a step limit. Rewards encourage correct answers and penalize asking/searching too much.

## Where do questions come from?

- From required slots in the active checklist and the highest‑entropy unknowns in $\{z_{slot_r}\}$.
- Render concise, targeted questions dynamically; optionally template per slot and let an LLM phrase the text.
- The question selector approximates EIG and chooses the slot that most reduces uncertainty given cost.

## Is our dataset big enough for a contextual bandit?

- Yes for a starter. A contextual bandit here is a light model (softmax over a short feature vector), not an RNN. With a few hundred scenarios and multiple episodes each, you can fit a stable linear policy that outperforms manual thresholds. We can expand scenarios over time and switch to deeper models later if needed.

## Project blueprint: "EIG Decider with Domain Packs"

### 0) Repo and tooling

Create monorepo `eig-decider/` with:

- `schema/` - metamodel and validators
- `domain_packs/` - each domain's data + mappings  
- `ingest/` - ETL to build graphs
- `decider/` - action policy (answer/ask/search)
- `retrieval/` - pluggable search clients
- `eval/` - benchmarks, simulators, metrics
- `cli/` - command-line demo
- `docs/` - architecture notes and runbooks

**Setup:**
- Use Python 3.11+ managed with `uv`; add `pyproject.toml`, ruff, and mypy
- Use Neo4j Aura (16–32 GB RAM). No local Docker required
- Provide PowerShell scripts (`scripts/*.ps1`) for common tasks instead of a Makefile

### 1) Metamodel (stable across domains)

- Define node labels: Entity, Attribute, Event, Procedure, Policy, plus domain labels (e.g., CVE, Product, Technique)
- Define the whitelisted predicates as relationship types: `IS_A`, `SUBCLASS_OF`, `PART_OF`, `LOCATED_IN`, `USES`, `HAS_PROPERTY`, `RELATED_TO`, `CAUSES`, `MITIGATES`, `REQUIRES`, `ALLOWS`, `PROHIBITS`, `HAS_STEP`, `NEXT_IF`, `APPLIES_TO`, `AFFECTS`
- Add temporal qualifiers using reified `:Fact` nodes when needed: `(:Fact {valid_from, valid_until, source, confidence})-[:SUBJECT]->(:Entity)` etc.
- Write `schema/metamodel.yaml` with allowed labels, rels, properties
- Write a schema linter `schema/lint_schema.py` that fails CI if a new rel type appears outside the whitelist

### 2) Neo4j bootstrap

- Init DB with constraints: unique IDs, required properties, indexing for `:Entity(id)`, `:Event(date)`
- For AuraDS, ensure GDS library is available; APOC + n10s as needed
- Add `schema/bootstrap.cypher` to create constraints and small reference taxonomies

### 3) Domain Pack: Cybersecurity (first implementation)

#### Pack structure

- `domain_packs/cyber/config.yaml` describing classes, mappings, and question bank
- `domain_packs/cyber/mappings.py` mapping raw feeds to metamodel
- `domain_packs/cyber/questions.yaml` the 12–15 canonical questions with graph shapes

#### Ingest tasks

- NVD CVE JSON feed → `:CVE` nodes with CVSS vectors
- CPE dictionary → `:Product`/`:Vendor` nodes; `(:CVE)-[:AFFECTS]->(:Product)`
- CISA KEV → flag exploited CVEs; add `:Event` for exploited_in_wild
- EPSS scores → properties on `:CVE`
- MITRE ATT&CK techniques/mitigations → `:Technique`/`:Mitigation`; edges `MITIGATES`, `CAUSES`/`RELATED_TO`
- Write `ingest/cyber_etl.py` to pull, normalize, upsert, snapshot label `vYYYY_MM_DD`

#### Procedure graph

- Define Procedure: VulnTriage with Step nodes: Identify → Validate Exposure → Triage (KEV/EPSS/CVSS) → Choose Action → Implement → Verify
- Attach `REQUIRES` edges for inputs: asset presence, internet exposure, criticality
- Use `NEXT_IF {cond}` for branching (e.g., `KEV=true` or `EPSS>0.7`)

#### Policy/cost

- Create Policy nodes: AskCost, SearchCost, PatchRisk, with properties `{latency_ms, token_cost, user_annoyance, risk_weight}`
- Map CVSS/EPSS/KEV to a scalar risk via a simple initial formula and tune over time:
  - `cvss_norm = CVSS / 10.0`
  - `risk_score = 0.5 * cvss_norm + 0.4 * EPSS + 0.1 * KEV_flag`
  - Cost weights (initial): `ask.user_annoyance = 1.0`, `search.token_cost = 1.0`, `search.latency_weight = 0.001`
  - Document in `docs/policy.md`

### 4) Question-first design (ties schema to utility)

- Finalize the top 12 questions for cyber triage, ordered by expected information gain (EIG)
- Example slots: `asset_present?`, `internet_exposed?`, `business_critical?`, `compensating_control?`
- For each question, define the graph shape it queries and the candidate answers
- Store in `domain_packs/cyber/questions.yaml` with fields: `id`, `text`, `slot`, `shape_cypher`, `answer_values`, `prior_p(answer)`

### 5) Working set builder

Implement `ingest/working_set.py`:

- **Input**: task context (e.g., CVE-2025-XXXX, asset tags)
- **Output**: subgraph limited to whitelisted predicates, 2 hops from seeds, last 90 days of events
- **Enforce size cap**: ≤ 25k nodes, ≤ 250k rels by default (configurable). If exceeded, back off and mark "search required"

### 6) Decider core (EIG/EFE policy)

- Implement `decider/state.py`: tracks known slots, evidence, candidate actions
- Implement `decider/confidence.py`: compute answer confidence via self-consistency or calibrated score from graph signals
- Implement `decider/eig.py`:
  - `score_answer(state)`: confidence minus risk
  - `score_ask(state)`: simulate K plausible user answers for each remaining slot (K=12 by default), estimate entropy drop
  - `score_search(state)`: run a cheap retrieval probe (stub), estimate expected entropy drop, minus cost
- Implement `decider/choose.py`: argmax over actions with thresholds and one-time forced-ask if required slots are empty

 - Optional AuraDS (v1 enabled):
   - Run GDS PageRank and/or fastRP on the working set to produce node scores/embeddings
   - Feed these as priors/features into `decider/confidence.py` to calibrate initial hypothesis probabilities

### 7) Retrieval module (pluggable, starts local)

- Create `retrieval/base.py`: interface `retrieve(query) -> docs`
- Implement `retrieval/local_snapshot.py` that fetches from your ingested snapshot (no network)
- Stub `retrieval/http_clients.py` for later vendor advisories; include cost parameters
- Add heuristic to set "recency triggers" that force search for time-sensitive facts

### 8) Evaluation harness

- Build `eval/scenarios_cyber.jsonl`: 100–300 scenarios with ground-truth actions and expected outcomes
- Implement a noisy user simulator that answers ask-questions with a small error rate
- **Metrics**: accuracy of final recommendation, questions asked, retrieval calls, latency, EIG per step
- `eval/run_eval.py` to batch-run scenarios and output a leaderboard table
- **Set pass/fail bars**: e.g., ≥ 85% correct with ≤ 3 questions average and ≤ 0.7 retrievals per scenario

### 9) CLI demo (handoff-friendly)

`cli/decide.py`:

```bash
--domain cyber --cve CVE-2025-XXXX --assets file.json
```

- Streams the policy's action choice and rationale
- Pretty-print the next question, or the retrieved snippet, or the final answer
- Ensure deterministic mode for tests; seed all randomness

### 10) CI, tests, and guards

- **Unit tests for**: schema linter, ETL mappers, working set builder, EIG math, decision thresholds
- **Snapshot tests**: load a mini-graph fixture and assert Cypher shapes for 5 core questions
- Add pre-commit hooks for formatting, type checks, and schema lint
- GitHub Actions to run tests on PR

### 11) Docs you actually need

- `docs/overview.md` - high-level architecture and metamodel diagram
- `docs/domain_pack_cyber.md` - data sources, mappings, question bank, procedures
- `docs/policy.md` - cost model, thresholds, and how to tune
- `docs/dev_guide.md` - how to run Neo4j, ingest, and evaluate locally

### Domain: Cybersecurity (dataset and questions)

- Dataset slice:
  - Entities: CVE, Product, Vendor, Technique, Event
  - Signals: CVSS (base), EPSS, KEV exploited flag
  - Relationships: `AFFECTS`, `MITIGATES`, `CAUSES`/`RELATED_TO`
- Procedure graph (modeled by us, aligned to ATT&CK where helpful):
  - Process: VulnTriage; Steps: Identify → Validate Exposure → Triage (KEV/EPSS/CVSS) → Choose Action → Implement → Verify
  - Pattern: `(:Process)-[:HAS_STEP]->(:Step)`, `(:Step)-[:REQUIRES]->(:Requirement)`, `(:Step)-[:NEXT_IF {cond}]->(:Step)`
- Policy/cost:
  - Ask: user_annoyance = 1.0; Search: token_cost = 1.0, latency_weight = 0.001; Answer: risk_weight from severity
  - Recency triggers: new vendor advisory, KEV change, EPSS spike
- Initial question bank (12–15):
  - Asset present? internet exposed? business critical? compensating control? patch window?
  - Vendor workaround available? outage/active exploit in the wild?
- Retrieval targets:
  - Vendor advisories, KEV updates, EPSS feeds, recent incidents
- Metamodel mapping: reuse whitelist (`AFFECTS`, `MITIGATES`, `RELATED_TO`, `REQUIRES`, `NEXT_IF`, `APPLIES_TO`)

### 12) Prove portability: second domain pack (3-day sprint)

- Pick one: Public Transit or Open Contracting
- Copy `domain_packs/template/` and fill `config.yaml`, `questions.yaml`
- Minimal ingest using public GTFS/OCDS sample bundles
- Reuse the same metamodel and decider; only swap the question bank and working set seeds
- Run eval with 50 scenarios; compare metrics to cyber to validate the metamodel

### 13) Stretch targets (optional, not required for v1)

- Graph embeddings with AuraDS GDS (fastRP) to provide priors for disambiguation
- Learn a regressor to predict `score_search` from cheap proxies (entity novelty, recency flags)
- **UI**: tiny web panel to visualize the question path and action costs
- Data versioning with DVC for snapshot diffs

## Deliverables by milestone

### Milestone A: Skeleton runs

- Repo + Aura Neo4j (16GB RAM) setup
- Schema linter passes
- Cyber ETL loads a small snapshot
- CLI answers a fixed CVE without search

### Milestone B: Policy online

- Full cyber question bank wired
- Decider picks among answer/ask/search
- Eval runs with metrics

### Milestone C: Portability

- Second domain pack ingested
- Same decider works; only data/questions changed
- Comparative eval report

---

## Ontology Brief — Editable Memory Graph (EMG) for Planning, Gap‑Detection, and Q&A

Purpose: drive an agent loop that retrieves facts/procedures, detects gaps, asks or searches, then writes back provenance‑rich facts. Keep a tiny upper ontology, put uncertainty first‑class, and avoid schema explosion.

Core modeling
- Small predicates (~10): `INSTANCE_OF`, `SUBCLASS_OF`, `HAS_SLOT`, `HAS_ATTRIBUTE` (optional), `SAME_AS`, `SUBJECT`/`PREDICATE`/`OBJECT` (for reified `:Fact`), `HAS_SOURCE`, lexical (`HAS_SECTION`/`HAS_PARAGRAPH`/`HAS_SENTENCE`/`MENTIONS`), procedural (`HAS_STEP`, `HAS_TOOL`), optional `PARTICIPATES_IN`.
- Node labels: `Entity`, `Type`, `Fact`, `Event` (optional), `MetaSlot`, `SlotValue`, `RelationType`, `Document`/`Section`/`Paragraph`/`Sentence`, `Procedure`/`Step`/`Tool`, `User`.
- Uncertainty: `Fact.confidence ∈ [0,1]`; `HAS_SLOT.confidence`; source weights on `HAS_SOURCE`.
- Sources are objects: `:Document` with lexical hierarchy; `:Fact-[:HAS_SOURCE]->:Document`.

Key properties & ids
- `Entity{name, wikidata_id?, embedding?}`; `Type{name, iri?}`; `RelationType{name}`; `Fact{kind, confidence, source, source_ref, timestamp}`; `SlotValue{slot,value}`; `Document{source_url,title}`; lexical nodes scoped by `(doc_url, order)`.
- Uniqueness examples: `Type.name`, `RelationType.name`, `Document.source_url`, `(SlotValue.slot, SlotValue.value)`; lexical scoping: `(Section.doc_url,order)` etc.

Minimal Cypher patterns
```cypher
MERGE (t:Type {name:'Film'})
MERGE (e:Entity {name:'Spirited Away'})-[:INSTANCE_OF]->(t)

MERGE (sv:SlotValue {slot:'Era', value:'2000s'})
MERGE (e)-[hs:HAS_SLOT]->(sv)
  ON CREATE SET hs.confidence=0.9, hs.source='derived'

MERGE (rt:RelationType {name:'WON_AWARD'})
MERGE (fa:Fact {kind:'WON_AWARD', film_id:'Q220677', target_id:'Q103360'})
  ON CREATE SET fa.confidence=0.9, fa.source='wikidata', fa.timestamp=date()
MERGE (fa)-[:SUBJECT]->(e)
MERGE (fa)-[:PREDICATE]->(rt)
MERGE (aw:Entity:Award {name:'Best Animated Feature', wikidata_id:'Q103360'})
MERGE (fa)-[:OBJECT]->(aw)

MERGE (d:Document {source_url:$url}) ON CREATE SET d.title=$title
MERGE (s:Section {doc_url:$url, order:1})
MERGE (p:Paragraph {doc_url:$url, order:1})
MERGE (sen:Sentence {doc_url:$url, order:1, text_excerpt:$text})
MERGE (d)-[:HAS_SECTION]->(s)
MERGE (s)-[:HAS_PARAGRAPH]->(p)
MERGE (p)-[:HAS_SENTENCE]->(sen)
MERGE (sen)-[:MENTIONS {confidence:0.9, via:'ner'}]->(e)
MERGE (fa)-[:HAS_SOURCE {support:'context'}]->(d)
```

Agent loop (incremental)
- Compute uncertainty/coverage via `MetaSlot` + missing/low‑confidence `HAS_SLOT` and entropy on candidates.
- Pick next: Ask (slot question) vs Search (sources) vs Execute (procedure step) by expected entropy drop − cost.
- Search uses Wikipedia/Wikidata first; extract facts; write back `:Fact` + `HAS_SOURCE` + lexical nodes.
- Log every step; later distill logs to a fast‑and‑frugal tree (FFT) and store as `(:Procedure {name:'…Policy.v#'})`.

Implementation status (v1)
- Slot selection via Bayesian EIG (signals) with diversity across steps (exclude asked)
- Schema‑driven question rendering (domain pack templates) + LLM fallback (Ollama)
- Wikipedia retriever + write‑back of reified Fact and lexical scaffolding
- CLI command `vegeta loop` for end‑to‑end uncertainty→question→search→write‑back
- Next: existence checks before retrieval, smarter upserts (HAS_SOURCE accumulation), early stop on evidence threshold, and EMG seed loader for non‑cyber demos
