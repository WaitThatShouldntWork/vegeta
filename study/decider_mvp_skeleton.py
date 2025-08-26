# %% [markdown]
# VEGETA Decider — Executable Skeleton (MVP)
# Minimal, function-free cells. Variables flow top→down; safe to run as-is.

# %%
# 0. CONFIG / CONSTANTS
ALPHA = 1.0
BETA = 0.5
GAMMA = 0.3
SIGMA_SEM2 = 0.30
SIGMA_STRUCT2 = 0.20
SIGMA_TERMS2 = 0.20
TAU_RETRIEVAL = 0.7
TAU_POSTERIOR = 0.7
K_ANCHORS = 10
M_CANDIDATES = 20
HOPS = 2
N_TERMS_MAX = 15
N_EXPECTED = 20
SMALL_SET_THRESHOLD = 3
SMALL_SET_BLEND = 0.5
LAMBDA_MISSING = 0.30
DEGREE_CAP = 40
LAMBDA_HUB = 0.02

# %%
# 1. INPUTS (per turn)
utterance_raw = None
history_summary = None
u_meta = {"summary": history_summary, "time": None, "prefs": {}}

# %%
# 2. UTTERANCE PREPROCESSING → u_sem, u_terms_set[, u_terms_vec]
u_sem = None                     # vector (e.g., 768-dim)
u_terms_set = set()              # canonical terms (lowercased, lemmatized)
u_terms_vec = None               # optional pooled term embedding

# %%
# 3. RETRIEVAL (anchors) → R
anchors = []                     # [{"node_id": ..., "s_sem": ..., "s_graph": ..., "s_combined": ...}, ...]
R = {
    "anchors": anchors,
    "expansion_params": {"hops": HOPS},
    "candidates": [],
    "provenance": {},
}

# %%
# 4. CANDIDATE EXPANSION → candidates S_j
candidates = []                  # list of candidate ids or small dicts

# %%
# 5. PER-CANDIDATE FEATURES → vec_subgraph, u_struct_obs
vec_subgraph = {}                # {S_id: vector}
u_struct_obs = {}                # {S_id: structure_counts}

# %%
# 6. EXPECTED CHANNELS g(v) → u'_sem/struct/terms
u_prime_sem = {}                 # {S_id: vector}
u_prime_struct = {}              # {S_id: counts}
u_prime_terms = {}               # {S_id: set[str]}

# %%
# 7. LIKELIHOOD DISTANCES δ_* and penalties
delta_sem = {}                   # {S_id: float}
delta_struct = {}                # {S_id: float}
delta_terms = {}                 # {S_id: float}
penalties = {}                   # {S_id: float}
log_likelihood = {}              # {S_id: float}

# %%
# 8. PRIORS
p_checklist = {}                 # {"MovieQuery": 1.0, ...}
p_goal = {}                      # {"identify": 1.0, ...}
p_subgraph = {}                  # {S_id: prior_prob}
p_dialogue_act = {}              # {"clarify": ..., ...}
p_step = {}                      # {"None": 1.0}
p_novelty = 0.0                  # scalar in [0,1]

# %%
# 9. POSTERIOR q(z_subgraph)
q_subgraph = {}                  # {S_id: posterior_prob}

# %%
# 10. SLOT TABLE AND POSTERIORS q(z_slot_r)
slot_table = {}                  # {slot: [{ "S": S_id, "fillers": [v1, ...] }, ...]}
q_slots = {}                     # {slot: {value: prob}}

# %%
# 11. UNCERTAINTY METRICS
H_subgraph = 0.0
slot_entropies = {}              # {slot: float}
margin_top12 = 0.0

# %%
# 12. EIG APPROX + UTILITY + ACTION
eig_ask = {}                     # {slot: float}
eig_search = 0.0
action_scores = {"ANSWER": 0.0, "ASK": 0.0, "SEARCH_LOCAL": 0.0, "SEARCH_WEB": 0.0}
chosen_action = None
ask_slot = None

# %%
# 13. LOGGING (retrieval trace and decisions)
retrieval_log = {
    "utterance_raw": utterance_raw,
    "u_terms_set": list(u_terms_set),
    "anchors": anchors,
    "candidates": candidates,
    "posterior": q_subgraph,
    "slot_entropies": slot_entropies,
    "decision": {"action": chosen_action, "ask_slot": ask_slot},
}

# %%
# 14. SESSION STATE CARRYOVER (b_t)
session_state = {
    "q_checklist": p_checklist,
    "q_goal": p_goal,
    "q_subgraph": q_subgraph,
    "q_slots": q_slots,
    "novelty": p_novelty,
    "retrieval_trace": {"R": R},
    "summary": history_summary,
}


