# %% [markdown]
# # VEGETA
# The goal with Vegeta is to build an agentic system that can pass the GAIA benchmark. 
# We are using an **active Bayesian inference** approach with **predictive coding** on knowledge graph,
# to have the system predict the next best action to take depending on an user input: ASK, SEARCH or ACT. 

# The hypothesis is rather than use Language Models and extended 'thinking' to brute force to the solution 
# (which still often doesn't work); we copy what humans do, which is *detect uncertainty* (gaps in our knowledge) 
# and ask for clarification (when given a task for example) or search for missing knowledge.

# Build a decider that chooses: answer vs ask vs search, using the free energy formula. When asking a question, it uses shannon entropy to 
# determine the best question to ask. 
# The project aims to be domain-agnostic but tested on movie knowledge.
#
# ## Theory Foundation: Bayesian Brain Model
# The Bayesian brain maintains beliefs about hidden causes `h` of sensory data `o`. 
# It generates predictions `o'` from beliefs via generative model `p(o|h)`, compares 
# to actual sensations, and updates beliefs to reduce prediction error.
#
# Exact Bayesian inference `p(h|o)` however, is intractable, so we use 
# approximate posterior `q_φ(h)` and minimize variational free energy:
# ```
# F(φ) = KL[q_φ(h) || p(h)] - E_q[log p(o|h)]
# ```
# This balances complexity (stay close to prior) vs accuracy (explain data).

# %% [markdown]
# ## Quick Reference: The Bayesian Loop
# ```
# User utterance → o_t (observation)
#                ↓
#    Embed & extract → {o_sem, o_terms, o_struct_obs}
#                ↓
#    Retrieval (k-NN) → candidate subgraphs S_j
#                ↓
#    For each S_j, predict → o'_t = {o'_sem, o'_struct, o'_terms}
#                ↓
#    Compute likelihood → p(o_t | h_t = S_j) via δ_sem, δ_struct, δ_terms
#                ↓
#    Combine with prior → q(h_t) ∝ p(o_t | h_t) · p(h_t)
#                ↓
#    Measure uncertainty → H[q(h_t)], H[q(z_slot_r)]
#                ↓
#    Decide action → ANSWER / ASK(slot r) / SEARCH
# ```

# %% [markdown]
# ## Notation Guide (Bayesian Brain → Implementation)
# 
# ### Theoretical Framework (Bayesian inference)
# - **h_t**: hidden state at turn t (the true but unknown situation)
#   - Implementation: h_t = {z_checklist, z_goal, z_subgraph, z_slot_r, z_dialogue_act, z_step, z_novelty}
# - **o_t**: observed features at turn t (what the user said, measured)
#   - Implementation: o_t = {o_sem, o_terms, o_struct_obs, o_meta}
# - **o'_t**: predicted features given hypothesis h_t (what we expect to observe if h_t were true)
#   - Implementation: o'_t = {o'_sem, o'_struct, o'_terms}
# 
# ### Variable Naming Convention
# **During retrieval/selection (query context):**
# - `q_sem`: user utterance embedding used as query vector (same as o_sem, different context)
# - `sem_emb`: pre-computed semantic embeddings stored in graph nodes (retrieval targets)
# - `s_sem`: similarity scores from k-NN retrieval (cosine between q_sem and sem_emb)
# - `s_graph`: graph-structure similarity scores from k-NN on graph embeddings
# 
# **During inference (likelihood computation):**
# - `o_sem`: observed utterance semantic embedding (o_t.sem)
# - `o_terms`: observed keywords/entities from utterance (o_t.terms)
# - `o_struct_obs(S_j)`: observed structure counts in candidate subgraph S_j (o_t.struct)
# - `o'_sem(S_j)`: expected semantic embedding for candidate S_j (o'_t.sem)
# - `o'_struct(S_j)`: expected structure from checklist for S_j (o'_t.struct)
# - `o'_terms(S_j)`: expected terms if S_j is correct (o'_t.terms)
# 
# ### Other symbols
# - t: turn index
# - S_j: candidate subgraph j
# - δ_sem, δ_struct, δ_terms: per-channel distances (observed vs predicted)
# - σ_sem², σ_struct², σ_terms²: noise variances per channel (aleatoric uncertainty)
# - α, β, γ: channel weights in log-likelihood
# - τ_retrieval: temperature for retrieval/candidate prior softmax
# - τ_posterior: temperature when normalizing p(o_t|h_t)·p(h_t) into q(h_t)
# - H[·]: Shannon entropy; EIG: expected information gain
# - R_t: retrieval context at turn t (anchors, params, candidates, provenance)
# 
# ### Complete Mapping Table
# | Concept | Classic | Time-indexed | Implementation |
# |---------|---------|--------------|----------------|
# | **Hidden state** | h | h_t | {z_checklist, z_subgraph, z_slot_r, ...} |
# | **Observation** | o | o_t | {o_sem, o_terms, o_struct_obs, o_meta} |
# | **Prediction** | o' | o'_t | {o'_sem, o'_struct, o'_terms} |
# | **Prior** | p(h) | p(h_t) | p(z_checklist)·p(z_goal\|...)·... |
# | **Likelihood** | p(o\|h) | p(o_t\|h_t) | exp(-[α·δ_sem + β·δ_struct + γ·δ_terms]) |
# | **Posterior** | q(h) | q(h_t) | Softmax(log p(o_t\|h_t) + log p(h_t)) |
# | **Belief state** | - | b_t | q_t(h_t) stored in session |
# | **Free energy** | F | F_t | KL[q(h_t)\|\|p(h_t)] - E_q[log p(o_t\|h_t)] |

# %% [markdown]
# ## Setup

# ### set up database & seed data

# use neo4j driver to Load seed into local graph (neo4j desktop)

# add embeddings to each node property (string, name, description, etc)
# Store in vector index
# add relation aware embeddings with GDS (fastRP, node2vec, graphSAGE, etc) into node property

# %% [markdown]
# # Process user utterance (o_t → o_sem, o_terms)
# 1) **Extract entities/keywords (LLM JSON-structured output)**
#    - Call a small 4B LLM  to return:
#      { canonical_terms (≤15), entities [{surface, normalized, type}], numbers, dates }.
#    - Validate against a strict JSON schema.
#    - Canonicalize: lowercase, lemmatize, dedupe; keep at most 15 canonical_terms.
#    - These canonical_terms feed δ_terms (likelihood). The linked entities (below) feed retrieval only.
# Outputs: {canonical_terms_set, entities[], numbers[], dates[]}
# %% [markdown]
# 2) **Make `o_terms`**:
#    - **Set form:** `o_terms_set` = canonical_terms_set.
#      δ_terms default = 1 − Jaccard(o_terms_set, expected_terms_set(S_j)).
# Outputs: {o_terms_set}
# %% [markdown]
# 3) **Link to graph ** (symbolic lookup)
#    - For each extracted entity. normalized, query Neo4j full-text over :Entity(name, aliases).
#    - Keep top 1–3 entity ids per entity if scores pass a small threshold; else leave as raw string.
#
# Notes:
# - δ_terms is a likelihood component only. Retrieval scores must not be reused inside the likelihood.
# - Outputs feed two places:
#     (a) **Retrieval (R):** linked_entity_ids boost anchors alongside semantic kNN.
#     (b) **Likelihood (later):** o_terms_set (and optional o_terms_vec) drive the terms channel.

###############################################################################################################
# %% [markdown]
# # Selective activation
# ## Initial subgraph anchor candidates
# 1) Embed user utterance → o_sem (e.g., 768-dim using nomic-embed-text via ollama).
# 2) Use o_sem as query vector (q_sem) for k-NN **against sem_emb** (node embeddings in DB) 
#    to get initial recall scores s_sem.                                        [index: idx_sem, metric: cosine]
# 3) For reranking, k-NN **against graph_emb** to get s_graph.                 [index: idx_graph, metric: cosine]
# 4) Normalize scores (z-score per list), then combine:
#       s_combined = 0.7 * z_sem + 0.3 * z_graph
#    Note: temperature τ is used later only when forming p(h_t|anchors) via softmax; not here.
#    Clarification: τ_retrieval denotes the retrieval softmax temperature used for p(z_subgraph|anchors).
# 5) Rank by s_combined and take top-K anchor nodes.                           [K = 10]
# 
# **Note:** q_sem and o_sem are the same vector; q_sem just emphasizes we're using it as a query.
#
# ## Checklist & Procedure Extraction (binary decision during retrieval)
# During selective activation, we also determine if a **procedural checklist** applies:
# - Detect if :Checklist nodes in Neo4j are similar to the user utterance
# - If semantic/structural similarity to a checklist with HAS_STEP relations exceeds threshold (e.g., 0.65):
#   → **Extract procedure = TRUE**, activate that checklist, initialize z_step to first step
# - Else: **Extract procedure = FALSE**, z_checklist may still be inferred (non-procedural), z_step = None
#
# This is an **all-or-nothing commitment** based on brain model: when humans recognize a familiar procedure
# (mortgage application, recipe, troubleshooting), we switch to procedural mode and follow the script.
# When we don't recognize one, we fall back to flexible inference.
#
# Implementation detail:
# - Store extracted_procedure flag in session state
# - If TRUE: policy uses step-gating (only ask/search for REQUIRES of current step)
# - If FALSE: policy is purely EIG-driven across all slots
# - Re-evaluate this flag only on major topic shifts (detected via semantic distance drop)

# %% [markdown]
# ## Neighborhood signature (for subgraph vectors)
# For each anchor, summarize its local structure so we can compare subgraphs:
# - Collect k-hop neighborhood (e.g., k=1..2).
# - Count node labels and edge types (a "bag of types").
# - Optionally hash/encode counts into a fixed vector using WL-style features. [Weisfeiler–Lehman hashing]
#   This captures patterns like "Person–Loan–Date" appearing together, useful for checklists.
# - **Pooling and fusion:**
#     - L2-normalize node embeddings, average with uniform weights → pooled_node_emb.
#     - Build structure_vector from label/edge-type counts.
#       - Vocabulary: fixed, ordered list of node labels and edge types (kept constant across runs).
#       - Counts: use log1p(count) to compress heavy tails.
#       - Normalization: divide by the max component per vector to scale into [0,1].
#       - Dimension: |labels| + |edge_types|.
#     - Concatenate vec_subgraph = [pooled_node_emb || 0.5·structure_vector]; then L2-normalize the concat.

# %% [markdown]
# # Expand into neighborhoods
# For each anchor, induce a small candidate subgraph:
# - Method A: fixed-radius expansion (e.g., 2 hops, typed edge filter).
# - Method B: personalized PageRank seeded by anchors (budgeted top-N nodes).
# - Add variance: different anchors, radii, or seed mixes to avoid clones.

# %% [markdown]
# ## Score candidate subgraphs
# Guard: retrieval-derived signals live only in priors; likelihood compares observed vs expected channels.
# For each candidate subgraph S_j, use the likelihood pieces directly (single source of truth):
# - δ_sem    = 1 - cosine(o_sem, o'_sem(S_j))
# - δ_struct = || log1p(o_struct_obs(S_j)) − log1p(o'_struct(S_j)) ||_2
# - δ_terms  = 1 − Jaccard(o_terms_set, expected_terms_set(S_j)); if min set size < 3, use 0.5·(1−Jaccard) + 0.5·(1−cosine(avg(term_embeds)))
# - penalties: +0.3 per missing required slot; hubbiness penalties (degree caps, detour penalties)
#
# Define the log-likelihood-based score used for ranking:
#   log p(o | h = S_j) = - [ α * δ_sem / σ_sem²
#                         + β * δ_struct / σ_struct²
#                         + γ * δ_terms / σ_terms²
#                         + penalties ]  + const
#
# Use this as the only scoring function for candidate ranking.
#
# ### Likelihood Function Clarification (implementation pseudocode)
# ```python
# def log_likelihood(o_t, o_prime_t, sigma_sq, alpha, beta, gamma, penalties):
#     """
#     Compute log p(o_t | h_t = S_j) for candidate subgraph S_j.
#     
#     Args:
#         o_t: observed features {sem, struct_obs, terms_set}
#              → implementation: {o_sem, o_struct_obs(S_j), o_terms}
#         o_prime_t: predicted features from S_j {sem, struct_expected, terms_set}
#              → implementation: {o'_sem(S_j), o'_struct(S_j), o'_terms(S_j)}
#         sigma_sq: dict {sem: float, struct: float, terms: float}
#         alpha, beta, gamma: channel weights
#         penalties: total penalty score
#     
#     Returns:
#         log_likelihood ∈ (-∞, 0]  (0 = perfect match)
#     """
#     # Semantic channel: cosine distance ∈ [0, 1]
#     # δ_sem = 1 - cos(o_sem, o'_sem(S_j))
#     delta_sem = 1.0 - cosine_similarity(o_t.sem, o_prime_t.sem)  # both L2-normalized
#     
#     # Structural channel: L2 on log-counts, normalized to [0, 1]
#     # δ_struct = ||log(o_struct_obs) - log(o'_struct)||
#     struct_obs_log = np.log1p(o_t.struct_obs)  # observed counts in S_j
#     struct_exp_log = np.log1p(o_prime_t.struct_expected)  # checklist expectations
#     delta_struct = np.linalg.norm(struct_obs_log - struct_exp_log)
#     delta_struct = np.clip(delta_struct / MAX_STRUCT_NORM, 0, 1)  # normalize, MAX_STRUCT_NORM ≈ 5.0
#     
#     # Terms channel: Jaccard or blended ∈ [0, 1]
#     # δ_terms = 1 - Jaccard(o_terms, o'_terms(S_j))
#     min_set_size = min(len(o_t.terms_set), len(o_prime_t.terms_set))
#     if min_set_size >= 3:
#         delta_terms = 1.0 - jaccard(o_t.terms_set, o_prime_t.terms_set)
#     else:
#         # Fallback for tiny sets: blend Jaccard with cosine of term embeddings
#         jacc = jaccard(o_t.terms_set, o_prime_t.terms_set)
#         cos = cosine_similarity(avg_embed(o_t.terms_set), avg_embed(o_prime_t.terms_set))
#         delta_terms = 0.5 * (1 - jacc) + 0.5 * (1 - cos)
#     
#     # Combine channels (Gaussian noise model)
#     raw_score = - (alpha * delta_sem / sigma_sq['sem']
#                  + beta * delta_struct / sigma_sq['struct']
#                  + gamma * delta_terms / sigma_sq['terms']
#                  + penalties)
#     
#     # Constant term (log normalizer) cancels when comparing candidates, omit or set to 0
#     return raw_score  # ∈ (-∞, 0]
# ```
# **Key properties:**
# - All δ values are in [0,1] by construction → prevents scale mismatches
# - Higher σ² = more forgiving (wider noise tolerance)
# - Penalties are additive in the exponent (equivalent to multiplying likelihood by exp(-penalty))
# - This is the **single source of truth** for scoring; do not create separate ranking functions 

# %% [markdown]
# ### Penalty definitions (plain language)
# - Missing required slot (per slot r):
#   - If the checklist says slot r is required and S_j has no valid filler → add λ_missing.
#   - Default λ_missing = 0.30. (Tweak later if it over/under-shoots.)
#
# - Hub node penalty (per node n with degree d):
#   - If a node is too “hubby,” penalize it: penalty_hub(n) = λ_hub · softplus(d − d_cap).
#   - Sum this over all nodes in S_j.
#   - Defaults: d_cap = 40, λ_hub = 0.02.  (Swap softplus for max(0, d − d_cap) if you want linear.)
#
# - Optional detour penalty:
#   - Add a small extra penalty when S_j takes long, irrelevant side paths (off by default).
#
# - Total penalties:
#   - total_penalty = Σ penalty_missing  +  Σ penalty_hub  +  optional_detour_penalty

# %% [markdown]
# ## Carry forward top-M candidates (no early collapse)
# By 'candidate (M)' we mean a subgraph with a set of nodes and relationships.
# - Keep top-M by score(S) (e.g., M=10–50). Do not pick a single winner yet.
# - Persist these candidates and their features to be used when computing the likelihood p(o|h) and the posterior q(z_subgraph).
# - Optional: lightweight reranking to improve the top-M ordering without collapsing.

# %% [markdown]
# ## Retrieval context R (definition)
# We denote the retrieval context as R:
# - R = {anchors, expansion params, candidate subgraphs S_j, vec_subgraph(S_j), o_struct_obs(S_j), provenance}
# - R is only used to build priors and per-candidate features, not part of observation o_t.
# - Used for: p(z_subgraph|anchors), slot priors, o_struct_obs(S_j), vec_subgraph(S_j).

# %% [markdown]
# ## Keep a retrieval log (provenance)
# Persist (JSON):
# - session_id, turn_id, timestamp, utterance_raw, o_terms_set
# - anchors: [{node_id, s_sem, s_graph, s_combined}], expansion params
# - candidates: [{subgraph_id, size, provenance_score}]
# - posterior summary: top‑K q(z_subgraph), slot entropies, channel deltas per candidate, novelty signals
# - decision: EIG numbers, chosen action, and outcome when available
# This makes results reproducible and debuggable.

# %% [markdown]
# ## Defaults (initial values)
# - K anchors = 10
# - M candidates = 20
# - hops = 2  OR  PPR top-N = 50
# - τ (retrieval softmax temperature) = 0.7
# - α = 1.0, β = 0.5, γ = 0.3
# - σ_sem² = 0.3, σ_struct² = 0.2, σ_terms² = 0.2
# - τ_retrieval (retrieval softmax temperature) = 0.7
# - τ_posterior (posterior softmax temperature) = 0.7
# - N_terms_max (canonical utterance terms) = 15
# - N_expected (cap for o'_terms) = 20
# - small_set_threshold = 3; small_set_blend = 0.5
# - λ_missing = 0.30; d_cap = 40; λ_hub = 0.02

###############################################################################################################
# %% [markdown]
# # Creating the prior p(h_t) over hidden states
# We keep a belief over hidden causes/states h_t that could explain the user's utterance.
# Include optional procedure fields in x: procedure_active ∈ {0,1}, step ∈ Steps ∪ {None}
# 
# Hidden state components (h_t):
# - z_checklist: which task template applies (e.g., MovieQuery, LoanApp, None)
# - z_goal: user's intent (identify, recommend, verify, explore, act)
# - z_subgraph: which candidate subgraph S_j explains the utterance
# - z_slot_r: value for each required slot r
# - z_dialogue_act: conversational move (clarify, confirm, request, provide)
# - z_step: current procedure step (if checklist has HAS_STEP relations)
# - z_novelty: out-of-distribution flag (low/high)

# ## Where do the priors come from?
# z_checklist prior & z_subgraph prior are fed from selective activation step: 
# - z_checklist prior: nudged by history summary + recent window + domain frequencies.
# - z_subgraph prior: seeded by our anchor retrieval scores from selective activation.
# Inference then updates both using the actual observation (the utterance and retrieved candidates).

# These are **independent-ish dials** turned in a sensible order:
# 1) p(z_checklist): which checklist fits? (loan application vs movie query vs none)
# 2) p(z_goal | ...): within that checklist, what does the user want right now?
# 3) p(z_subgraph | ...): among the candidates we retrieved, which chunk of the graph explains the utterance best?
# 4) p(z_step | z_checklist, z_goal, history): which procedure step are we in? (gates questions)
# 5) p(z_slot_r | z_subgraph, z_step): for required slots, what values look right? (and which are missing)
# 6) p(z_dialogue_act | history, utterance): clarify/confirm/request/provide; biases Ask vs Answer wording/thresholds
# 7) p(z_novelty | anchors, distances): are we out-of-distribution (i.e., probably need SEARCH)?

# %% [markdown]
# ## Cold Start Priors (t=0, no history)
# When the session begins (turn 0), we have no previous posteriors to carry forward.
# Initialize all priors from base rates and the first utterance:
#
# **p_0(z_checklist):**
# - Uniform over {common checklists + None}, weighted by global frequency
# - If domain is known from context (e.g., URL, user profile), boost relevant checklists
# - Always keep p_0(z_checklist = None) ≥ 0.15 to avoid forced fits
# - Example: p_0 = {MovieQuery: 0.25, LoanApp: 0.10, None: 0.15, ...normalize...}
#
# **p_0(z_goal):**
# - Run intent classifier on first utterance only (no history)
# - Use uniform prior over {identify, recommend, verify, explore, act} if classifier fails
# - Apply checklist↔goal compatibility Ψ after checklist is inferred
#
# **p_0(z_subgraph):**
# - Purely from retrieval scores (selective activation)
# - No recency boost (nothing is recent yet)
# - Use simplicity + provenance only: smaller, well-sourced subgraphs slightly favored
# - Softmax over anchor scores with τ_retrieval
#
# **p_0(z_step):**
# - If procedure extracted: z_step = first step in HAS_STEP chain (order=1)
# - Else: z_step = None
#
# **p_0(z_slot_r):**
# - Uniform over available fillers in candidate subgraphs + UNKNOWN
# - Weight by popularity (degree centrality) if filler appears in multiple candidates
# - Type constraints enforced: impossible types get p=0
#
# **p_0(z_dialogue_act):**
# - Heuristics only (no dialogue history):
#   - Question marks / wh-words → request or clarify
#   - Declarative + entity → provide
#   - Default: uniform over {clarify: 0.3, request: 0.4, provide: 0.2, confirm: 0.1}
#
# **p_0(z_novelty):**
# - Check if max(cosine(o_0.sem, all_known_embeddings)) < 0.35 → high novelty
# - Mahalanobis distance to embedding cluster > 3σ → high novelty
# - Default: p_0(novelty=low) = 0.7, p_0(novelty=high) = 0.3
#
# **Transition to t=1:**
# After first action (ANSWER/ASK/SEARCH), update to q_0(·) using Bayes rule,
# then use q_0 as input to p_1 via carryover (see "Lifecycle of posterior" section).

# %% [markdown]
# ## Checklist Definition (graph schema)
# Checklists are reusable templates stored in Neo4j that define typed requirement sets for answering specific questions.
# They serve as explicit procedural memory that can be introspected and improved over time.
# (See docs/ontology.md for full specification)
#
# **Structure:**
# - `:Checklist {name, description}` — e.g., "IdentifyFilm", "AssessCyberRisk"
# - `:SlotSpec {name, expect_labels, rel, required, cardinality}` — individual requirements
# - Relationship: `(:Checklist)-[:REQUIRES]->(:SlotSpec)`
#
# **SlotSpec properties:**
# - `name`: slot identifier (e.g., "film", "year", "applicant")
# - `expect_labels`: array of node labels expected (e.g., ["Film"], ["Person"])
# - `rel`: relationship type to validate (e.g., "INSTANCE_OF", "ACTED_IN")
# - `required`: boolean (true = must be filled for complete answer)
# - `cardinality`: "ONE" or "MANY" (how many fillers expected)
#
# ### Example: IdentifyFilm checklist
# ```cypher
# CREATE (:Checklist {name: "IdentifyFilm", description: "Determine which specific film the user is asking about"})
# CREATE (:SlotSpec {name: "film", expect_labels: ["Film"], rel: "INSTANCE_OF", required: true, cardinality: "ONE"})
# CREATE (:SlotSpec {name: "year", expect_labels: ["Year","Date"], rel: "RELEASED_IN", required: false, cardinality: "ONE"})
# CREATE (:SlotSpec {name: "director", expect_labels: ["Person"], rel: "DIRECTED_BY", required: false, cardinality: "ONE"})
# CREATE (:SlotSpec {name: "actor", expect_labels: ["Person"], rel: "ACTED_IN", required: false, cardinality: "MANY"})
#
# MATCH (c:Checklist {name: "IdentifyFilm"}), (s:SlotSpec) WHERE s.name IN ["film", "year", "director", "actor"]
# CREATE (c)-[:REQUIRES]->(s)
# ```
#
# ### Example: Procedural checklist with steps
# ```cypher
# CREATE (:Checklist {name: "LoanApplication", description: "Multi-step loan application workflow"})
# CREATE (:SlotSpec {name: "applicant", expect_labels: ["Person"], required: true, cardinality: "ONE"})
# CREATE (:SlotSpec {name: "amount", expect_labels: ["Currency"], required: true, cardinality: "ONE"})
# CREATE (:SlotSpec {name: "income", expect_labels: ["Currency"], required: true, cardinality: "ONE"})
#
# CREATE (:Step {name: "CollectBasicInfo", order: 1})
# CREATE (:Step {name: "AssessEligibility", order: 2})
#
# MATCH (c:Checklist {name: "LoanApplication"}), (s:SlotSpec) WHERE s.name IN ["applicant", "amount", "income"]
# CREATE (c)-[:REQUIRES]->(s)
#
# MATCH (c:Checklist {name: "LoanApplication"}), (step1:Step {order: 1}), (step2:Step {order: 2})
# CREATE (c)-[:HAS_STEP]->(step1)
# CREATE (step1)-[:REQUIRES {slot: "applicant"}]->()
# CREATE (step1)-[:REQUIRES {slot: "amount"}]->()
# CREATE (step1)-[:NEXT_IF {condition: "has(applicant) AND has(amount)"}]->(step2)
# ```
#
# **Benefits:**
# - Explicit "completeness" definition for each task type
# - Introspectable reasoning (why did we ask X?)
# - Foundation for learned policies that skip steps when confidence is high
# - Clear separation from world model (no ghost edges in entity graph)
#
# **Extraction during retrieval:**
# If semantic/structural similarity to a checklist template exceeds threshold (e.g., 0.65),
# activate that checklist and its procedure graph (if HAS_STEP relations exist).

# %% [markdown]
# ## Prior over checklist p(z_checklist)
# What kind of problem are we trying to solve? (pick the playbook):
# Examples: LoanApplication vs MovieQuery vs EligibilityCheck vs None.
# - **Frequency**: which checklists are common in our logs. (Counts → probabilities)
# - **Recency/context**: things discussed in the last few turns get a boost (with time decay).
# - **User profile**: tiny nudge if we know the user's domain.
# - **Null option**: always keep "none-of-the-above" nonzero to avoid forcing a bad fit.

# %% [markdown]
# ## Prior over goal p(z_goal | history)
# What the user is trying to do right now; combined with checklist via a small compatibility table (Ψ[checklist, goal]).
# Multiply by Ψ[checklist, goal] to down-weight incompatible pairs.
# Examples: identify a film, get a recommendation, verify a fact, play 20Q, ask the agent to act, (recommend me a movie, will score low against verify a fact etc)
# Built from: prompting a small LLM "“Classify the user’s intent into one of: identify, recommend, verify, explore, act. Return a JSON with scores.”
# over the last 1–3 turns + checklist↔goal compatibility table.
# Long term, when we have 1–2k labeled snippets, we'd train a supervised classifier and retire the LLM for this task (it will be cheaper and lower latency to run)

# %% [markdown]
# ## Prior over subgraph p(z_subgraph | anchors)
# Which concrete chunk of the graph is most likely BEFORE we score in detail? (*after retrieval, before semantic/structural scoring*.):
# Retrieval context R provides anchors, candidates, and params used here.
# - **Retrieval softmax**: normalize your combined anchor/candidate scores (temperature τ).
# - **Simplicity**: smaller subgraphs get a gentle boost (don’t over-explain).
# - **Provenance**: edges/facts with better sources raise the prior.
# - **Recency**: nodes touched recently get a tiny bump.

# %% [markdown]
# ## Prior over dialogue act p(z_dialogue_act | history, utterance)
# What conversational move is likely right now?
# - Values: {clarify, confirm, request, provide}. Simple 4-way tag.
# - Signals (cheap):
#   - Imperatives/questions → request/clarify ("can you", "what is", "which")
#   - Declaratives with entity mention → provide/confirm
#   - Punctuation + wh-words; last agent move; presence of options in context
# - Why it helps:
#   - **Bias Ask vs Answer**: if 'provide' is likely, tolerate lower answer thresholds; if 'clarify', prefer targeted Ask.
#   - **Wording**: shapes question templates (confirm vs open clarify).
# - Implementation: tiny classifier (LLM?) over n-grams + dialogue features (last 1–2 turns). Start with heuristics; upgrade to a linear model later.

# %% [markdown]
# ## Prior over step p(z_step | z_checklist, z_goal, history)
# Which procedure step are we currently in, within the active checklist?
# - Model steps explicitly: `(:Process)-[:HAS_STEP]->(:Step)` with `:REQUIRES` preconditions and `:NEXT_IF {cond}` branches.
# - Signals:
#   - Recently satisfied preconditions; last answered slot; typical step orders
#   - Goal ↔ step compatibility (e.g., "Validate Exposure" before "Triage")
# - Why it helps (gating):
#   - Ask only for slots that **unlock the current step** (missing REQUIRES).
#   - Prefer SEARCH when the step requires public facts; ASK when private facts are missing.
#   - MOMDP consistency note:
#       - Steps live in observed x. When no procedure applies: x.procedure_active=0 and x.step=None.
#       - T_x handles both cases (advance when REQUIRES met; else identity; identity when step=None).
#       - We always keep belief only over hidden y and use action-conditioned O; gating is a policy choice, not a model constraint.
# Step gating toggle:
# - has_procedure = (z_checklist != None) and (z_step != None)
# - When has_procedure == True → gate ASK/SEARCH to actions that satisfy REQUIRES(current_step) or unlock NEXT_IF.
# - When False → no gating; use EIG to choose the single highest information question.
# - Result: fewer, more surgical questions; faster path to a decision.
# - Procedure-agnostic fallback (simple):
#   - Treat z_step as an always-on prior feature. If no procedure applies, set its score to 0.
#   - This way, nothing breaks and there is no special gating to maintain.
#   - Still keep a tiny topic agenda to guide Ask/Search without a formal procedure graph.

# %% [markdown]
# ## Slot priors p(z_slot_r)
# How we start each required slot (initial guesses for the blanks you must fill.)
# - **Uniform** if we truly know nothing.
# - **Popularity nearby**: if a filler appears often in the candidate neighborhood, give it prior mass.
# - **Type constraints**: impossible types get zero probability from the start (saves time and dumb questions) 
# -- Example: - Slot: `:Review.stars` expects an **int in {1..5}**. Candidate is 8.7. → impossible.
# -- Example:  - Relation: expects `(User)-[:RATED]->(Movie)`. Candidate is `(Movie)-[:RATED]->(User)`. → impossible.
# Slots describe what *should* exist in a complete answer, not pre-created placeholders.
# - We look at the candidate subgraph and nearby nodes as possible fillers for each REQUIRED slot.
# - Slot prior = an initial probability over **available** fillers + an **"unknown"** option.
# - If nothing nearby fits a slot’s type/constraints, that slot’s entropy stays high → perfect ASK target.
# The brain analogy holds: no one stores empty boxes; you predict a box *should* be there, then look.

# %% [markdown]
# ## Novelty/OOD prior p(z_novelty)
# Does this look outside of our comfort zone? (do we even have the right playbook, or should we go look stuff up?):
# - Heuristics:
#   - max cosine(o_sem, anchors) < 0.35 → novelty high (Low max similarity to all anchors)
#   - Mahalanobis distance to global embedding mean/cov > 3 → novelty high (High distance from known cluster)
#   - poor checklist likelihood (utterance doesn’t fit any checklist language).
# - Effect: raise p(z_novelty), bump σ² across channels, bias policy toward SEARCH.
# - Mapping: novelty ∈ [0,1] → inflate = 1 + novelty; σ_*²_eff = inflate · σ_*² (cap at ×2).

#################################################################################

# %% [markdown]
# # Observation o_t (what we measure from the user's input)
# 
# Theoretical: o_t is the observed feature vector at turn t
# 
# Implementation components:
# - **o_sem**: utterance semantic embedding (768-dim vector from nomic-embed-text)
#   - What the user said, encoded as dense vector
# - **o_terms**: utterance keywords/entities (canonical_terms_set, size ≤ 15)
#   - Extracted via LLM, canonicalized (lowercase, lemmatized, deduped)
# - **o_struct_obs(S_j)**: observed structure counts in candidate subgraph S_j
#   - Counts of node labels and edge types actually present in S_j
#   - Measured AFTER retrieval, specific to each candidate
# - **o_meta**: context metadata (history summary, time, user prefs)
#   - Used to bias priors only; NOT scored in likelihood
#
# **Mapping:** o_t = {o_sem, o_terms, o_struct_obs(S_j), o_meta}

# %% [markdown]
# ## Where each component is used:
# - **o_sem** → δ_sem channel: distance to o'_sem(S_j)
# - **o_terms** → δ_terms channel: Jaccard/cosine to o'_terms(S_j)
# - **o_struct_obs(S_j)** → δ_struct channel: L2 distance to o'_struct(S_j)
# - **o_meta** → prior biasing only (not in likelihood)
# - **R_t** (retrieval context: anchors, S_j candidates) → builds priors, slot distributions
#   - R_t is NOT part of observation o_t

#################################################################################

#################################################################################

# %% [markdown]
# # Prediction o'_t = g(h_t) (what we expect to observe if hypothesis h_t were true)
# 
# Theoretical: o'_t is the predicted feature vector given hidden state h_t
# 
# Think of g(h_t) as a simple, deterministic *predictor* that generates expected observations.
# For a candidate subgraph S_j (h_t includes z_subgraph = S_j), we predict:
# 
# Implementation components:
# 
# **1) o'_sem(S_j)** - Expected semantic embedding:
#   - Weighted pooling of node embeddings in candidate subgraph S_j
#   - Concatenate with structure vector (label/edge-type counts)
#   - L2-normalize the result
#   - Dimensions: same as o_sem (768-dim)
# 
# **2) o'_struct(S_j)** - Expected structure pattern:
#   - Counts of node labels and edge types REQUIRED by the Checklist
#   - Cardinality expectations (ONE vs MANY per slot)
#   - What SHOULD exist in a complete answer
# 
# **3) o'_terms(S_j)** - Expected key terms:
#   - Sources:
#     * Names/aliases from nodes in S_j
#     * Relation names (edge types like ACTED_IN, DIRECTED_BY)
#     * Slot names from checklist ("applicant," "release year")
#     * Checklist-specific lexicon ("interest rate," "eligibility")
#     * Optional: TF-IDF tokens from linked documents (HAS_SOURCE)
#     * Optional: LLM-generated terms for this subgraph
#   - Processing:
#     * Dedupe, lowercase, lemmatize/stem
#     * Cap to N_expected terms (e.g., 20)
#     * Store as set or embed for cosine comparison
# 
# **Mapping:** o'_t = {o'_sem(S_j), o'_struct(S_j), o'_terms(S_j)}
# 
# Then likelihood compares observed vs predicted:
# - δ_sem = 1 - cos(o_sem, o'_sem)
# - δ_struct = ||log(o_struct_obs) - log(o'_struct)||
# - δ_terms = 1 - Jaccard(o_terms, o'_terms)

# Normalize channels before weighting:
# - δ_sem: clamp to [0,1] (cosine distance on L2-normalized vectors).
# - δ_struct: L2 on normalized log1p-count vectors; scale to [0,1] by a fixed norm cap.
# - δ_terms: Jaccard-based ∈ [0,1]; small-set fallback blend stays in [0,1].
# Calibrate α, β, γ so terms contribute similarly on a validation set.

# %% [markdown]
# # Summary: g(h_t) → o'_t (prediction function)
# For a hypothesis h_t where z_subgraph = S_j:
# 
# **Predicted channels:**
# - o'_sem(S_j)    : pooled node embeddings ⊕ structure vector → L2-normalize
# - o'_struct(S_j) : expected slot/edge counts from Checklist (what SHOULD exist)
# - o'_terms(S_j)  : expected terms (names/aliases, relation types, slot names, lexicon)
#
# **Likelihood via per-channel distances:**
# - δ_sem    = 1 - cosine(o_sem, o'_sem(S_j))
# - δ_struct = ||log(o_struct_obs(S_j)) - log(o'_struct(S_j))||  [+ penalties for missing required]
# - δ_terms  = 1 - Jaccard(o_terms, o'_terms(S_j))  [or blended with cosine for small sets]
# 
# **Combined likelihood:**
# log p(o_t | h_t) = -[α·δ_sem/σ_sem² + β·δ_struct/σ_struct² + γ·δ_terms/σ_terms² + penalties]
# 
# This is the single source of truth for candidate scoring.


# For each channel, we'll apply our noise model σ²:
# Heuristics to set them:

# Start with fixed values (e.g., 0.3, 0.2, 0.2) and calibrate on a small validation set.
# Adapt online:
# Short, vague utterance → increase σ_sem² and σ_terms² (be more forgiving).
# High OOD/novelty → increase all σ² (don’t overcommit).
# Very long, specific utterance → decrease σ² (be stricter).
# Likelihood shape (conceptually):

# log p(o | h) = - [ α * δ_sem / σ_sem²
#                 + β * δ_struct / σ_struct²
#                 + γ * δ_terms / σ_terms²
#                 + penalties ]  + const


# Where δ_* are distances (e.g., 1 − cosine for semantic, L2 for counts, 1 − Jaccard for terms)

#################################################################################

# %% [markdown]
# # Prediction error and likelihood
# See the likelihood definition above. Candidate scoring uses the same δ_* distances and penalties;
# do not maintain a separate scoring formula.

# %% [markdown]
# # Posterior q(h_t | o_t)
# ## Approximate posterior over hidden states (our updated belief after seeing evidence o_t)
# 
# We maintain a factorized posterior distribution:
# 
# q(h_t) = q(z_checklist)
#        · q(z_goal | z_checklist, history)
#        · q(z_subgraph | z_checklist, anchors)
#        · q(z_step | z_checklist, z_goal, history)    # optional; None if no procedure
#        · ∏_r q(z_slot_r | z_subgraph, z_step)        # for each required slot r
#        · q(z_dialogue_act | history, o_t)
#        · q(z_novelty | anchors, distances)
# 
# Each factor is a probability distribution over possible values of that hidden variable.

#################################################################################

# %% [markdown]
# # Variational update (free energy minimization)
# We maintain approximate posterior q(h_t) and update it by minimizing free energy.
# 
# **Free energy:**
#   F = KL[q(h_t) || p(h_t)] - E_q[log p(o_t | h_t)]
# 
# **Intuition:**
# - **Accuracy term:** E_q[log p(o_t | h_t)] measures how well our beliefs explain the observation
#   - Low → data doesn't match beliefs → F increases → must update
# - **Complexity term:** KL[q(h_t) || p(h_t)] measures deviation from prior
#   - High → beliefs moved far from prior without evidence → F increases
# - We minimize F to balance fitting data vs staying grounded in priors
# 
# **Implementation (discrete case):**
# For each candidate hypothesis (e.g., subgraph S_j):
#   q(h_t = S_j) ∝ p(o_t | h_t = S_j) · p(h_t = S_j)
# 
# Normalize via softmax with temperature τ_posterior:
#   q(h_t = S_j) = exp(log p(o_t|S_j) + log p(S_j)) / Z
# 
# **Note:** Use τ_posterior for posterior inference; τ_retrieval is only for retrieval priors.

# %% [markdown]
# # Quantifying uncertainty (what do we know, what do we not?)
# We need numbers that tell us "I don't know" vs "I kind of know."
# 
# ## Global uncertainty over hypotheses
# - **Entropy:** H[q(h_t)] over candidate hypotheses (subgraphs/checklists)
#   - H = -Σ q(h_t) log q(h_t)  [Shannon entropy]
#   - High H → belief spread thin → high uncertainty
# - **Margin:** q(top1) - q(top2)
#   - Small margin → two strong competing hypotheses
#   - Cheap proxy for uncertainty
#
# ## Slot-level uncertainty (known unknowns)
# - For each required slot r: maintain distribution q(z_slot_r)
# - **Slot entropy:** H_r = -Σ q(z_slot_r=v) log q(z_slot_r=v)
#   - High H_r → don't know this slot → good ASK target
#
# ## Epistemic vs aleatoric uncertainty
# - **Epistemic (model uncertainty):** use ensembles or dropout to get variance over scores
#   - "We don't know because we haven't seen enough data"
# - **Aleatoric (data noise):** σ² in likelihood noise model
#   - "The observation itself is inherently noisy"
#
# **These metrics drive the decision policy:** ANSWER vs ASK vs SEARCH

# %% [markdown]
# # Expected Information Gain (EIG) for actions
# We estimate how much an action would reduce uncertainty.
# 
# ## Actions (kept simple)
# - ANSWER: say the best answer now. Utility ≈ confidence − risk − delay_cost.
# - ASK: ask the user a precise question to fill the most uncertain slot (taking into consideration hybrid policy).
# EIG under hybrid policy:
# - Procedural: compute EIG_ASK(r) only for r ∈ EligibleAskSlots; others = 0.
# - Non-procedural: compute EIG_ASK(r) over all plausible slots; include discriminativeness across top-2 subgraphs.
# - SEARCH: model outcomes {success, fail} with p_success; on success, observe slot values/prune subgraphs; compute expected entropy drop.
# - SEARCH (Local): look in the graph/snapshot for facts (low cost/latency).
# - SEARCH (Web): look on the internet for facts (higher cost/latency).
#
# ## EIG definition (conceptual)
#   EIG(action) = expected reduction in entropy after the observation we get from that action.
#   EIG = H[current] - E_{outcome}[ H[posterior after outcome] ].
#
# ## Cheap approximations (because life is short)
# - For ASK(slot r): sample top-3 plausible values + "unknown", locally reweight q and average the entropy drop → EIG_ASK(r).
# - For SEARCH(Local/Web): assume p_success = 0.5; ΔH_success = 0.30, ΔH_fail = 0.05 → EIG_SEARCH = 0.175.
#
# We compare EIG against action costs to decide.

# %% [markdown]
# # Cost, risk, and utility
# Perfect curiosity is cute; users have time budgets.
# Define a simple utility for each action:
# 
#   Utility(action) = expected_gain(action) - cost(action) - risk(action)
# 
# - cost(ASK): 0.10 (friction of bothering user, extra turns, token budget)
# - cost(SEARCH): 0.20 (latency, API cost, distraction risk)
# - risk(ANSWER): (1 − q_top1) * 1.0        [calibrate via past accuracy vs. confidence]
#
# Choose the action with highest utility. If everything is junk, default to ASK minimal clarification.

# %% [markdown]
# # Decision policy (answer vs ask vs search)

# A) Decision policy
# Check for procedure-first, otherwise EIG-first
# - If a procedure is extracted and an active step is identified (z_checklist, z_step not None):
#     - EligibleAskSlots = { r ∈ REQUIRED(current_checklist) : REQUIRES_r ⊄ filled_slots }
#     - Score only r ∈ EligibleAskSlots using action-conditioned EIG (ASK) and SEARCH targeting facts that satisfy REQUIRES.
#     - Add unlock_bonus to actions that satisfy NEXT_IF preconditions.
# - Else (no procedure applies this turn):
#     - Compute EIG over discriminative slots (across top-2 subgraphs) without gating.
#     - Consider SEARCH when EIG_ASK is low or novelty/provenance flags are high.
#       State features: [q_top1, margin, H(q_checklist), H(q_subgraph), max_slot_entropy,
#                    OOD flag, provenance score, expected cost(time/$), turn_count]
#     - Actions: {ANSWER, ASK(slot r*), SEARCH(pattern p*)}
#     - Reward: +1 if final answer accepted/correct; -λ_time per turn; -λ_$ per API call; large -C if wrong answer.
# - Learn with: contextual bandit or tiny actor-critic (2–3 layer MLP). No need to backprop through the graph.

# %% [markdown]
# # Hybrid selection:
# - Procedural mode: rank EligibleAskSlots by score_r = EIG_ASK(r) - cost_r + unlock_bonus.
# - Non-procedural mode: rank all candidate slots by score_r = EIG_ASK(r) - cost_r (no step-gating: e.g procedural steps).
# - If all scores ≤ 0 and uncertainty remains, consider SEARCH with p_success, else ANSWER if confident.

# %% [markdown]
# ## 0) Inputs we already have
# - Active checklist (from z_checklist top-1).
# - Candidate subgraphs S_j (from retrieval R) with q(z_subgraph = S_j).
# - Slot specs (required slots only): allowed types/relations.
# - o_terms (utterance terms) for light name overlap checks.

# %% [markdown]
# ## 1) Build a slot table (required slots only)
# For each required slot r in the active checklist:
# - For each candidate S_j:
#   - Enumerate **candidate fillers** near the instance that satisfy SlotSpec (type + relation).
#   - If none exist, use special value **UNKNOWN**.
# Result:
#   slot_table[r] = [{ "S": S1, "fillers": [v1, v3] },
#                    { "S": S2, "fillers": [v2]     },
#                    { "S": S3, "fillers": ["UNKNOWN"] }]

# %% [markdown]
# ## 2) Local score per filler (inside each candidate)
# Simple, fast, no rocket science:
#   provenance ∈ [0,1]     # mean source reliability along the path
#   closeness  ∈ [0,1]     # hop distance or PPR weight, normalized
#   term_match ∈ [0,1]     # quick overlap with o_terms if filler has a name
# Local score:
#   loc(r, v | S_j) = 0.5*provenance + 0.3*closeness + 0.2*term_match
# If S_j has no filler:
#   loc(r, UNKNOWN | S_j) = 1.0

# %% [markdown]
# ## 3) Posterior over slot values q(z_slot_r)
# Aggregate across candidates using current belief over subgraphs:
#   p_r(v) ∝ Σ_j  q(z_subgraph = S_j) * loc(r, v | S_j)
# Include UNKNOWN as a value. Normalize p_r to sum to 1.
# Store this as q(z_slot_r).

# %% [markdown]
# ## 4) Features per slot (how “ask-worthy” is r?)
# For each slot r, compute:
# - Entropy:              H_r = −Σ_v p_r(v) log p_r(v)             # higher = more uncertain
# - Unknown mass:         u_r = p_r(UNKNOWN)                        # higher = we lack data
# - Split power (cheap):  1 if top-2 subgraphs imply different fillers (or one UNKNOWN), else 0
#   (Slightly fancier: Jaccard distance between the two filler sets.)
# - Conflict flag:        1 if multiple incompatible fillers compete across candidates, else 0
# - User cost:            cost_r (small table, e.g., year=0.05, actor=0.10, email=0.40)
# - Past success rate:    psr_r ∈ [0,1] (moving avg of actual entropy drop when we asked this slot before)

# %% [markdown]
# ## 5) Score = approximate EIG − cost (greedy pick max)
# Heuristic EIG (works in practice):
#   eig_r ≈ a1*H_r + a2*split_power + a3*conflict + a4*u_r
# Adjust by past success:
#   adj_r = b1*psr_r
# Final score:
#   score_r = eig_r * adj_r − cost_r
# Defaults:
#   a1=1.0, a2=0.5, a3=0.3, a4=0.5, b1=1.0
# Pick r* = argmax_r score_r. That’s the slot we ASK.

# %% [markdown]
# ## 6) Generate the question (LLM does the wording, not the thinking)
# Provide a small, structured payload to a tiny LLM:
# {
#   "checklist": "MovieQuery",
#   "slot": "year",
#   "top_candidates": [
#     {"id":"S1","film":"Inception","year":"2010"},
#     {"id":"S2","film":"Interstellar","year":"2014"}
#   ],
#   "reason": "High entropy; S1 vs S2 differ on 'year'; UNKNOWN low.",
#   "style": "one short, neutral question"
# }
# Output examples:
# - "Is the release **year 2010 or 2014**?"
# - If UNKNOWN is high: "Do you know the **release year**?"

# %% [markdown]
# ### Guardrails (so we don’t ask dumb questions)
# - Only consider **required** slots of the **current** checklist.
# - Cap candidate fillers per slot per S_j to K=3 (you don’t need a census).
# - If all slots are certain, but q(z_subgraph) is still mushy → prefer **SEARCH** on a fact that splits S1 vs S2.
# - If many slots come up UNKNOWN across most candidates → you probably picked the wrong checklist → ask a **checklist disambiguation** question instead.
# %% [markdown]
# ### Tiny worked example
# Slot film:  p(Inception)=0.58, p(Interstellar)=0.35, p(UNKNOWN)=0.07  → H low → **don’t ask**
# Slot year:  p(2010)=0.46, p(2014)=0.44, p(UNKNOWN)=0.10             → H high + split=1 → **ask year**
# Slot country: p(UNKNOWN)=0.85                                        → high unknown but weak splitter → likely lower score than year

# C) Rerank and hyperparameters
# - Learn α, β, γ, and σ² per Checklist from data (minimize NLL or calibrate vs accuracy).
# - Learn retrieval knobs (K, hops, τ) with simple bandit tuning on offline logs.

# High-level rules:
# 
# 1) ANSWER if q_top1 ≥ 0.70 AND (q_top1 − q_top2) ≥ 0.20 AND all required-slot entropies H_r < 0.5.
# 
# 2) Else compute EIG_ASK for top‑k highest‑entropy slots and normalize by cost:
#       score_ask(r) = EIG_ASK(r) / cost_ask
# 
# 3) Compute EIG_SEARCH (targeted retrieval) when structural disagreement, weak provenance, or novelty/OOD is high.
# 
# 4) Choose action with maximum Utility; break ties by lowest latency.
# 
# Sensible defaults:
# - θ_conf = 0.70, margin = 0.20, H_r threshold = 0.5.
# - If top1−top2 margin < 0.20, favor ASK/SEARCH.
# - Core retrieval and likelihood knobs as in Defaults above.


# %% [markdown]
# # Detecting "known unknowns" vs "unknown unknowns"
# We should tell apart "I know what I’m missing" from "I don’t even know the neighborhood."
# 
# ## Known unknown (good news, just ask)
# - Posterior focused on one checklist/subgraph but one or two slots have high entropy.
# - Action: ASK for the highest EIG slot. The question is surgical.
# 
# ## Unknown unknown (probably search)
# - Posterior spread across unrelated checklists, or OOD signals are high:
#   - query far from training distribution in embedding space,
#   - low cosine to all anchors,
#   - high reconstruction error on structure sketch,
#   - repeated contradictions in top candidates.
# - Action: SEARCH with a broadening query, or restate to the user that scope may be outside knowledge.

# %% [markdown]
# # Crafting the ASK (targeted question design)
# We don't ask open-ended essays. We ask for the bit that collapses uncertainty fastest.
# 
# Steps:
# - Rank slots by entropy H_r and by impact on distinguishing top hypotheses.  [mutual information with h_t]
# - Form a concise question that names the slot and offers constrained choices if possible.
#   Example: "Which product is this about: A, B, or C?"                       [max EIG per token]
# - Enforce a budget: at most N questions before we must SEARCH or ANSWER.

# %% [markdown]
# # Crafting the SEARCH (targeted retrieval design)
# We don’t shotgun Google. We design the retrieval to split the posterior.
# 
# Steps:
# - Inspect where top candidates disagree: specific entity, date, or relation.
# - Build a retrieval query that would confirm/deny those diffs.               [discriminative evidence]
# - Prefer sources with known reliability profiles; weight by provenance.      [reduces risk]
# - Estimate EIG_SEARCH from past calibration: how often this retrieval narrows entropy.

# %% [markdown]
# # Calibration and metacognition (confidence that isn't delusional)
# Confidence scores must mean something.
# 
# - Train a small calibration head mapping features → predicted accuracy:
#   features = [q(top1), margin, H[q(h_t)], slot completeness, provenance stats].
# - Use reliability diagrams and ECE to tune thresholds.                       [expected calibration error]
# - If confidence is miscalibrated in a domain, increase ASK/SEARCH aggressiveness there.

# %% [markdown]
# # Noise and temperature controls
# - Gaussian noise σ² controls how forgiving we are to mismatches.            [aleatoric]
# - Sampling temperature τ controls how peaky the posterior q(h_t) is.        [exploration vs exploitation]
# - Increase σ or τ when text is short, ambiguous, or OOD indicators are high.
# - Decrease when inputs are long, specific, and matched to priors.

# %% [markdown]
# # Posterior predictive check (sanity pass before answering)
# Before committing to ANSWER:
# - Generate o' from the chosen hypothesis and check it explains key parts of o_t.
# - If large residual error remains on any critical feature, downgrade to ASK/SEARCH.
# - This prevents confident nonsense with a 1-line test.

# %% [markdown]
# # Learning from outcomes (close the loop)
# - After we ANSWER, track whether the user accepts/corrects it.               [supervision]
# - After ASK/SEARCH, measure realized entropy drop vs predicted EIG.          [calibrate EIG]
# - Update priors p(h_t), slot reliability, and provenance weights over time.  [online learning]
#

# %% [markdown]
# # Lifecycle of the posterior across turns
# We maintain a belief state b_t ≈ q_t(h_t) after each turn t.
# Next turn's priors p_{t+1}(h_{t+1}) are derived from b_t with temporal decay and context.
# Think: "Bayesian filtering for dialogue."
# 
# **MOMDP structure:**
# - **Observed (x):** checklist, step, filled_slots_this_session, high-provenance facts
# - **Hidden (y):** unfilled slot values, subgraph_id (top-M), novelty/OOD flag
# - We maintain belief over y conditioned on x
# 
# ## Notation
# - **b_t** = q_t(h_t): belief state (posterior) at end of turn t
# - **p_{t+1}(h_{t+1})**: prior distribution for next turn
# - **h_t** = {z_checklist, z_goal, z_subgraph, {z_slot_r}, z_dialogue_act, z_step, z_novelty}

# %% [markdown]
# ## 1) End of turn t: we have q_t(h_t) and we just acted (ANSWER/ASK/SEARCH)
# - **If ANSWER:** mark which subgraph + slot values we committed to
# - **If ASK:** we received user reply → treat as hard/soft evidence, update q_t
# - **If SEARCH:** new facts/provenance added to graph → update available candidates
# 
# **Step transitions (observed x):**
# - If all REQUIRES(current_step) ⊆ filled_slots → advance deterministically via NEXT_IF
# - Else: stay at current step
# - Filled slots persist across turns; unknowns remain until ASK/SEARCH resolves them

# %% [markdown]
# ## 2) What do we persist? (so next turn isn’t amnesic)
# Persist a compact **session state** object:
# - q_t(z_checklist) and q_t(z_goal)   [distributions, not just argmax]
# - q_t(z_subgraph) truncated to top-M  [store ids + weights]
# - Slot posteriors for required slots  [value dists + “unknown” mass]
# - OOD/novelty flag, residual error stats
# - Retrieval trace (anchors, params, top-M pre-inference scores) and posterior q(z_subgraph) summary; final selection if chosen
# - Conversation summary (1 paragraph), updated with the decision and any facts learned

# %% [markdown]
# ## 3) Turn t+1: build priors from yesterday’s beliefs
# High-level rule: next priors = tempered carryover × base priors × recency/context.
# For each latent:
# - p_{t+1}(z_checklist) ∝  (q_t(z_checklist))^ρ  × base_schema_prior × context_boost
# - p_{t+1}(z_goal)      ∝  (q_t(z_goal))^ρ       × intent_head(history) × Ψ[checklist,goal]
# - p_{t+1}(z_subgraph)  ∝  (q_t(z_subgraph))^ρ   × recency × simplicity × provenance
# - p_{t+1}(z_slot_r)    from previous slot posteriors, but **reset if checklist changes**
# - p_{t+1}(z_novelty)    decays toward baseline unless OOD signals persist
#
# Where ρ ∈ (0,1] is "inertia":
#   - ρ≈1  → sticky beliefs across turns (good for a focused task)
#   - ρ≈0.6→ forgetful (good when users ping-pong topics)

# %% [markdown]
# ## 4) Floors, guards, and resets (so you don’t get stuck)
# - Always keep a nonzero floor on "None" checklist and "null subgraph" (prevents forced fits).
# - If utterance similarity to prior context drops hard (topic shift), **soft reset**:
#   - lower ρ for checklist/goal/subgraph this turn
#   - re-run selective activation with broader seeds
# - If the user contradicts last turn’s answer, apply a **penalty memory** to that provenance/type
#   so it doesn’t win again without stronger evidence.

# %% [markdown]
# ## 5) Posterior predictive → anticipating the next turn
# We anticipate the next 1–3 utterances to plan efficiently (like humans do).
# 
# ### Short-horizon planning (unlocking branches)
# Procedural mode (when z_step is defined):
# - "Unlocking a branch" means asking for (or searching) the smallest missing precondition that enables the next Procedure Step
#   or collapses competing subgraph hypotheses. It moves us from the current node in the procedure graph to a decisive next node.
# - Maintain a tiny **plan** latent z_plan = ordered list of 1–3 next items that would unlock progress.
#   Treat this as the top of the session scratchpad (we persist it), and record why each item is chosen
#   (expected split, cost, and how it helps).
# - Compute a 2-step lookahead score:
#   - EIG_1: immediate entropy drop from ASK/SEARCH now.
#   - EIG_2: "Think two moves ahead." First, how much we learn if we act now (EIG_1).
#     Then, after a likely outcome, what's the best next move and how much more would we learn? We
#     average this over a few plausible outcomes.
# - Prefer actions that unlock steps (satisfy REQUIRES on z_step) and separate top-2 subgraphs.
# 
# Non‑procedural mode (when z_step=None):
# - Treat "branches" as topic alternatives or answer templates (e.g., identify vs recommend vs verify), or as top‑K subgraphs.
# - Unlocking = asking for a minimal fact that disambiguates top topics/entities or fills the highest‑entropy slot relevant to the goal.
# - Build z_plan from: (a) highest MI slots w.r.t. top‑2 subgraphs, (b) goal‑conditioned slot priors, (c) recency/novelty triggers.
# - Same EIG_1/EIG_2 computation; just drop step precondition checks.
# 
# Practical heuristics:
# - If procedural: rank missing REQUIRES by (entropy × step‑criticality).
# - If non‑procedural: rank by (entropy × goal relevance × discriminativeness across top‑2 subgraphs).
# - Add one discriminative fact between top‑2 subgraphs if margin is thin.
# - Keep horizon to 2 (occasionally 3) to control cost.

# %% [markdown]
# # Evaluation metrics 
# 1. Retrieval recall@10 for known entities
# 2. Uncertainty calibration (Expected Calibration Error on held-out questions not used in training)  
# 3. Action efficiency (# turns to correct answer vs baseline)
# 4. Component ablations (what happens if we remove z_dialogue_act and how does that affect system performance?)
