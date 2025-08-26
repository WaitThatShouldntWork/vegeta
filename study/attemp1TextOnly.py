# %% [markdown]
# # Bayesian Active Inference Decider
# Build a decider that chooses: answer vs ask vs search, using entropy. 
# Domain-agnostic but tested on movie knowledge.
#
# ## Theory Foundation: Bayesian Brain Model
# The Bayesian brain maintains beliefs about hidden causes `v` of sensory data `u`. 
# It generates predictions `u'` from beliefs via generative model `p(u|v)`, compares 
# to actual sensations, and updates beliefs to reduce prediction error.
#
# Exact Bayesian inference `p(v|u)` however, is intractable, so we use 
# approximate posterior `q_φ(v)` and minimize variational free energy:
# ```
# F(φ) = KL[q_φ(v) || p(v)] - E_q[log p(u|v)]
# ```
# This balances complexity (stay close to prior) vs accuracy (explain data).

# %% [markdown]
# ## Glossary of symbols (concise)
# - u: observed utterance features this turn (u_sem, u_terms, u_meta)
# - u': predicted features from hypothesis v (u'_sem, u'_struct, u'_terms)
# - S_j: candidate subgraph j
# - δ_sem, δ_struct, δ_terms: per-channel distances used in the likelihood
# - σ_sem², σ_struct², σ_terms²: noise variances per channel (aleatoric)
# - α, β, γ: channel weights inside the log-likelihood
# - τ_retrieval: temperature for retrieval/candidate prior softmax
# - τ_posterior: temperature when normalizing p(u|v)·p(v) into q(v)
# - H[·]: Shannon entropy; EIG: expected information gain
# - R: retrieval context (anchors, params, candidates, provenance)

# %% [markdown]
# ## Setup

# ### set up database & seed data

# use neo4j driver to Load seed into local graph (neo4j desktop)

# add embeddings to each node property (string, name, description, etc)
# Store in vector index
# add relation aware embeddings with GDS (fastRP, node2vec, graphSAGE, etc) into node property

# %% [markdown]
# # Process user utterance
# 1) **Extract entities/keywords (LLM-first JSON)**
#    - Call a small 4B LLM in JSON/function mode to return:
#      { canonical_terms (≤15), entities [{surface, normalized, type}], numbers, dates }.
#    - Validate against a strict JSON schema; on failure: retry with "JSON only" prompt; on second failure: fall back to a minimal rule-based extractor for this turn.
#    - Canonicalize: lowercase, lemmatize, dedupe; keep at most 15 canonical_terms.
#    - These canonical_terms feed δ_terms (likelihood). The linked entities (below) feed retrieval only.
# Outputs: {canonical_terms_set, entities[], numbers[], dates[]}
# %% [markdown]
# 2) **Make `u_terms`**:
#    - **Set form:** `u_terms_set` = canonical_terms_set.
#      δ_terms default = 1 − Jaccard(u_terms_set, expected_terms_set(S_j)).
#      Fallback (tiny sets, min size < 3): δ_terms = 0.5·(1−Jaccard) + 0.5·(1−cosine(avg(term_embeddings))).
#    - **Vector form (optional):** embed each term; average → `u_terms_vec` for the fallback cosine.
# Outputs: {u_terms_set, optional u_terms_vec}
# %% [markdown]
# 3) **Link to graph when possible** (symbolic lookup)
#    - For each extracted entity.normalized, query Neo4j full-text over :Entity(name, aliases).
#    - Keep top 1–3 entity ids per entity if scores pass a small threshold; else leave as raw string.
#
# Notes:
# - δ_terms is a likelihood component only. Retrieval scores must not be reused inside the likelihood.
# - Outputs feed two places:
#     (a) **Retrieval (R):** linked_entity_ids boost anchors alongside semantic kNN.
#     (b) **Likelihood (later):** u_terms_set (and optional u_terms_vec) drive the terms channel.

###############################################################################################################
# %% [markdown]
# # Selective activation
# ## Initial subgraph anchor candidates
# 1) Take the embedded user utterance -> q_sem (e.g., 768-dim using nomic-text-embed via ollama).
# 2) k-NN **against sem_emb** to get initial recall scores s_sem.            [index: idx_sem, metric: cosine]
# 3) For reranking, k-NN **against graph_emb** to get s_graph.                 [index: idx_graph, metric: cosine]
# 4) Normalize scores (z-score per list), then combine:
#       s_combined = 0.7 * z_sem + 0.3 * z_graph
#    Note: temperature τ is used later only when forming p(z_subgraph|anchors) via softmax; not here.
#    Clarification: τ_retrieval denotes the retrieval softmax temperature used for p(z_subgraph|anchors).
# 5) Rank by s_combined and take top-K anchor nodes.                           [K = 10]

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
# - δ_sem    = 1 - cosine(u_sem, u'_sem(S_j))
# - δ_struct = || log1p(u_struct_obs(S_j)) − log1p(u'_struct(S_j)) ||_2
# - δ_terms  = 1 − Jaccard(u_terms_set, expected_terms_set(S_j)); if min set size < 3, use 0.5·(1−Jaccard) + 0.5·(1−cosine(avg(term_embeds)))
# - penalties: +0.3 per missing required slot; hubbiness penalties (degree caps, detour penalties)
#
# Define the log-likelihood-based score used for ranking:
#   log p(u | v = S_j) = - [ α * δ_sem / σ_sem²
#                         + β * δ_struct / σ_struct²
#                         + γ * δ_terms / σ_terms²
#                         + penalties ]  + const
#
# Use this as the only scoring function for candidate ranking. 

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
# - Persist these candidates and their features to be used when computing the likelihood p(u|v) and the posterior q(z_subgraph).
# - Optional: lightweight reranking to improve the top-M ordering without collapsing.

# %% [markdown]
# ## Retrieval context R (definition)
# We denote the retrieval context as R:
# - R = {anchors, expansion params, candidate subgraphs S_j, vec_subgraph(S_j), u_struct_obs(S_j), provenance}
# - R is only used to build priors and per-candidate features, not part of observation u.
# - Used for: p(z_subgraph|anchors), slot priors, u_struct_obs(S_j), vec_subgraph(S_j).

# %% [markdown]
# ## Keep a retrieval log (provenance)
# Persist (JSON):
# - session_id, turn_id, timestamp, utterance_raw, u_terms_set
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
# - N_expected (cap for u'_terms) = 20
# - small_set_threshold = 3; small_set_blend = 0.5
# - λ_missing = 0.30; d_cap = 40; λ_hub = 0.02

###############################################################################################################
# %% [markdown]
# # Creating the prior (v)
# We keep a belief over hidden causes/states (v) that could explain the user's utterance.

# Hidden states: z_checklist, z_goal, z_subgraph, z_slot_r, z_dialogue_act, z_step, z_novelty. The “actual but unknown” situation this turn.

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
# ## Prior over checklist p(z_checklist)
# What kind of problem are we trying to solve? (pick the playbook):
# Examples: LoanApplication vs MovieQuery vs EligibilityCheck vs None.
# - **Frequency**: which checklists are common in our logs. (Counts → probabilities)
# - **Recency/context**: things discussed in the last few turns get a boost (with time decay).
# - **User profile**: tiny nudge if we know the user’s domain.
# - **Null option**: always keep “none-of-the-above” nonzero to avoid forcing a bad fit.

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
#   - max cosine(u_sem, anchors) < 0.35 → novelty high (Low max similarity to all anchors)
#   - Mahalanobis distance to global embedding mean/cov > 3 → novelty high (High distance from known cluster)
#   - poor checklist likelihood (utterance doesn’t fit any checklist language).
# - Effect: raise p(z_novelty), bump σ² across channels, bias policy toward SEARCH.
# - Mapping: novelty ∈ [0,1] → inflate = 1 + novelty; σ_*²_eff = inflate · σ_*² (cap at ×2).

#################################################################################

# %% [markdown]
# # Observation (u) (external)
# - u_sem: utterance embedding (what the user said, as a vector)
# - u_terms : utterance keywords/entities (bag of terms OR an embedding of that bag)
# - u_meta  : short history summary, time, user prefs (used to bias priors; not scored directly)
#
# This is the "sensory data" we compare against u' from the generative model.

# ## Per-candidate observed structure (measured AFTER retrieval)
# - u_struct_obs(S_j): what we actually see in candidate S_j
# -- (counts of node labels, edge-types, simple cardinalities in that subgraph)

# %% [markdown]
# ## Plug-in map (where each thing plugs in)
# - u_sem: utterance embedding → used in δ_sem vs u′_sem(S_j)
# - u_terms_set / u_terms_vec: utterance terms → used in δ_terms vs expected_terms(S_j)
# - u_meta: history/preferences/time → only biases priors, not likelihood
# - R (anchors, S_j): used to build p(z_subgraph|anchors), u_struct_obs(S_j), vec_subgraph(S_j), slot priors; R is not observation
# - u_struct_obs(S_j): observed counts inside S_j → used in δ_struct vs u′_struct(S_j)

#################################################################################

#################################################################################

# %% [markdown]
# # Prediction of u' = g(v) (what we *expect* to observe if v were true):
# Think of g(v) as a simple, deterministic *predictor* that spits out "expected observation features."
#    - Semantic: subgraph embedding (pooled node vecs ⊕ structure sketch), expected key terms.
#    -- Key terms are:
#    --- Names/aliases from nodes in the candidate subgraph (entities, types).
#    --- Relation names (RelationType) and slot names (“applicant,” “release year”).
#    --- Checklist-specific lexicon (“interest rate,” “eligibility,” “recommend,” “similar”).
#    --- Optional: top TF-IDF tokens from documents linked via HAS_SOURCE.
#    --- Other consideration: use LLM fed from this data to propose a set of possible key terms, and even better, questions.
#    -- We'll:
#    --- Create a small expected-term vector (bag of words or an embedding of those terms), capped to top N_expected terms.
#    --- Compare to the utterance terms/embedding as part of the likelihood

#    - Structure: which required slots/edges should be present (from the Checklist).
#    - Noise model: how forgiving we are (σ²), so small mismatches aren’t catastrophic.

# %% [markdown]
# Not a giant neural decoder. Just structured expectations you can check against reality.
#
# We'll output three channels:
#   u'_sem   = expected semantic vector for the candidate subgraph
#             - Weighted pooling of node embeddings in the candidate subgraph
#             - ⊕ concat with the small structure vector
#             - Normalize

#   u'_struct = expected structure sketch (counts/patterns of types/edges)
#             - Counts of node labels and edge types required by the Checklist
#             - Plus any cardinality expectations (ONE vs MANY)
#   u'_terms = expected key terms/phrases likely to appear in the utterance
#             - Bag of expected tokens from node names/aliases, slot/rel labels, checklist lexicon
#             - Cap size to N_expected (e.g., 20), dedupe, lowercase, lemmatize/stem
#             - Optionally embed this small list to compare with utterance embedding
#             - Optionally use LLM to propose a set of possible key terms, and even better, questions.
#
# Then the likelihood compares (u_sem, u_struct, u_terms) vs (u'_sem, u'_struct, u'_terms) with a noise model.

# Normalize channels before weighting:
# - δ_sem: clamp to [0,1] (cosine distance on L2-normalized vectors).
# - δ_struct: L2 on normalized log1p-count vectors; scale to [0,1] by a fixed norm cap.
# - δ_terms: Jaccard-based ∈ [0,1]; small-set fallback blend stays in [0,1].
# Calibrate α, β, γ so terms contribute similarly on a validation set.

# %% [markdown]
# # 2) g(v) — predicted channels we compare against u
# For a hypothesis v ≈ {z_checklist, z_subgraph[, z_slot_r, z_goal]}:
# - u'_sem(S_j)    : pooled node embeddings ⊕ structure vector (concat, then normalize)
# - u'_struct(S_j) : expected slot/edge pattern from the Checklist (what SHOULD exist)
# - u'_terms(S_j)  : expected key terms (names/aliases from nodes, relation words, checklist lexicon)
#
# Likelihood uses per-channel distances:
# - δ_sem    = 1 - cosine(u_sem, u'_sem)
# - δ_struct = distance(u_struct_obs(S_j), u'_struct)          # e.g., L2 on counts, penalties for required-missing
# - δ_terms  = 1 - JaccardOverlap OR 1 - cosine(term-embeds)
# Then combine with weights and noise σ².
# This likelihood is the single source of truth; the candidate scoring section uses this exact definition.


# For each channel, we'll apply our noise model σ²:
# Heuristics to set them:

# Start with fixed values (e.g., 0.3, 0.2, 0.2) and calibrate on a small validation set.
# Adapt online:
# Short, vague utterance → increase σ_sem² and σ_terms² (be more forgiving).
# High OOD/novelty → increase all σ² (don’t overcommit).
# Very long, specific utterance → decrease σ² (be stricter).
# Likelihood shape (conceptually):

# log p(u | v) = - [ α * δ_sem / σ_sem²
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
# # Posterior
# ## q(.) = your approximate posterior over the hidden states (our current belief after looking at the evidence)
# factorised q
#  q(v) = q(z_checklist)
#        · q(z_goal | z_checklist, history)
#        · q(z_subgraph | z_checklist, anchors)
#        · q(z_step | z_checklist, z_goal, history)    # optional; if None, drop this factor
#        · ∏_r q(z_slot_r | z_subgraph, z_step)        # condition on z_step when present; else on z_subgraph only
#        · q(z_dialogue_act | history, utterance)
#        · q(z_novelty | anchors, distances)

#################################################################################

# %% [markdown]
# # Variational update (free energy view)
# We keep an approximate posterior q(v) over checklists/subgraphs/slots.
# We minimize free energy to balance "fit the data" vs "don’t overfit":
# 
#   F = KL[q(v) || p(v)] - E_q[ log p(u | v) ]
# 
# Intuition:
# - If the data fits badly, E_q[log p(u|v)] is low → F goes up → we must change beliefs.
# - If we move too far from the prior without evidence, KL goes up → F goes up.
# - We descend F to update q(v).                                           [simple gradient or discrete reweighting]
#
# In practice: normalize p(u|v) * p(v) across candidates to get q(v).      [softmax with temperature τ]
# Clarification: use τ_posterior here; τ_retrieval is used only for retrieval prior.

# %% [markdown]
# # Quantifying uncertainty (what do we know, what do we not?)
# We need numbers that tell us "I don't know" vs "I kind of know."
# 
# ## Global uncertainty over hypotheses
# - Entropy H[q(v)] over candidate subgraphs/checklists.  [Shannon entropy]
#   High H => belief is spread thin => we’re uncertain.
# - Margin: q(top1) - q(top2).                            [cheap proxy]
#
# ## Slot-level uncertainty (known unknowns)
# - For each required slot r, keep a distribution q(value_r).
# - Slot entropy H_r: high means we don't know this slot. [ask target]
#
# ## Model uncertainty vs data noise
# - Use ensembles or dropout to get variance over scores.  [epistemic]
# - Use σ² in Gaussian noise for aleatoric uncertainty.    [data noise]
#
# These metrics drive the meta-decision: answer vs ask vs search.

# %% [markdown]
# # Expected Information Gain (EIG) for actions
# We estimate how much an action would reduce uncertainty.
# 
# ## Actions (kept simple)
# - ANSWER: say the best answer now. Utility ≈ confidence − risk − delay_cost.
# - ASK: ask the user a precise question to fill the most uncertain slot.
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

# A) Decision policy (core)
# - State features: [q_top1, margin, H(q_checklist), H(q_subgraph), max_slot_entropy,
#                    OOD flag, provenance score, expected cost(time/$), turn_count]
# - Actions: {ANSWER, ASK(slot r*), SEARCH(pattern p*)}
# - Reward: +1 if final answer accepted/correct; -λ_time per turn; -λ_$ per API call; large -C if wrong answer.
# - Learn with: contextual bandit or tiny actor-critic (2–3 layer MLP). No need to backprop through the graph.

# %% [markdown]
# # B) Question selection (which slot to ASK)
#
# Goal: pick **one** slot to ask that will shrink uncertainty the most (high EIG), without annoying the user.
# We do this in 6 tidy steps. Then we hand the winner to a tiny LLM to phrase the question.

# %% [markdown]
# ## 0) Inputs we already have
# - Active checklist (from z_checklist top-1).
# - Candidate subgraphs S_j (from retrieval R) with q(z_subgraph = S_j).
# - Slot specs (required slots only): allowed types/relations.
# - u_terms (utterance terms) for light name overlap checks.

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
#   term_match ∈ [0,1]     # quick overlap with u_terms if filler has a name
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
# We don’t ask open-ended essays. We ask for the bit that collapses uncertainty fastest.
# 
# Steps:
# - Rank slots by entropy H_r and by impact on distinguishing top hypotheses.  [mutual information with v]
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
#   features = [q(top1), margin, H[q(v)], slot completeness, provenance stats].
# - Use reliability diagrams and ECE to tune thresholds.                       [expected calibration error]
# - If confidence is miscalibrated in a domain, increase ASK/SEARCH aggressiveness there.

# %% [markdown]
# # Noise and temperature controls
# - Gaussian noise σ² controls how forgiving we are to mismatches.            [aleatoric]
# - Sampling temperature τ controls how peaky the posterior q(v) is.          [exploration vs exploitation]
# - Increase σ or τ when text is short, ambiguous, or OOD indicators are high.
# - Decrease when inputs are long, specific, and matched to priors.

# %% [markdown]
# # Posterior predictive check (sanity pass before answering)
# Before committing to ANSWER:
# - Generate u' from the chosen hypothesis and check it explains key parts of u.
# - If large residual error remains on any critical feature, downgrade to ASK/SEARCH.
# - This prevents confident nonsense with a 1-line test.

# %% [markdown]
# # Learning from outcomes (close the loop)
# - After we ANSWER, track whether the user accepts/corrects it.               [supervision]
# - After ASK/SEARCH, measure realized entropy drop vs predicted EIG.          [calibrate EIG]
# - Update priors p(v), slot reliability, and provenance weights over time.    [online learning]
#

# %% [markdown]
# # Lifecycle of the posterior across turns
# We keep a belief state b_t ≈ q_t(v) after each turn t. Next turn’s priors come from b_t,
# tempered by decay and context. Think "Bayesian filtering for dialogue."

# %% [markdown]
# ## 0) Notation
# - b_t: belief state at end of turn t (your posterior q_t(v))
# - p_{t+1}: priors for next turn
# - v = { z_checklist, z_goal, z_subgraph, {z_slot_r}, z_novelty }

# %% [markdown]
# ## 1) End of turn t: we have q_t(v) and we just acted (ANSWER/ASK/SEARCH)
# - If ANSWER: mark which subgraph + slots we committed to.
# - If ASK: we got a user reply → treat it as hard/soft evidence.
# - If SEARCH: we wrote new facts/provenance → graph changed (great).

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
#
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


