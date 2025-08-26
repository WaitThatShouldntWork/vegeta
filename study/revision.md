Study + Drill + Revision Plan (domain-agnostic, EIG-ready)
Phase 0 — Pocket Fluency (the bare minimum math you need)

Goal: read outputs, spot nonsense, and steer.

Random variables, conditionals: p(u∣v), p(v∣u), marginal p(u)

Bayes’ rule, log-likelihoods, MAP vs posterior

Gaussian mean/variance; precision = σ⁻² ; why logs are your friend

Cosine similarity; von Mises–Fisher intuition for normalized vectors

Entropy H(p); KL divergence D_KL(q∥p)

Variational free energy (ELBO) as a computable bound on surprise

Drills

Convert three textual statements into Bayes form and back.

Given mean/variance, tell me precision and which of two sensors you’d trust.

Compute a tiny entropy: (0.7,0.2,0.1) vs (0.34,0.33,0.33).

Take two sentence vectors; compute cosine; explain what a 0.85 vs 0.45 score means for likelihood weighting.

Phase 1 — The Spine (v, u, u′) and your tiny ontology

Goal: lock the core variables that never change across domains.

Observation u: user utterance features (text + optional structured cues)

Hidden state v=(t,e,z): task, referent entity, slots

Generator g(v): predicts features you'd expect in the utterance if v were true

Drills

List 3 tasks you’ll actually support now: {lookup, identify_by_clue, recommend}.

For each, name 2–3 slots you might need (year, actor, genre). Mark each slot {known, unknown}.

Write one sentence per task: "If v=(t,e,z) were true, I'd expect words/features like …"

Phase 2 — Likelihoods you can swap into any domain

Goal: get two working likelihood families; compose them additively.

vMF/embedding: normalize embeddings, prototype u'v = norm(α·μt + β·μe + γ·μz); score ∝ κ·u⊤u'v


Naive Bayes/bag-of-words: token priors per task, entity, slot; sum log-probs

Structured checks (mini logistic): numbers, dates, booleans contribute small logits

Drills

Build a toy token table for two tasks and two films; hand-score a query.

Create 3 prototype vectors; compute blended u' for two different (α,β,γ).

Identify three structured cues you can parse reliably (year, known actor string, explicit “recommend”).

Phase 3 — Priors and Posterior

Goal: stop vibing; quantify belief.

Priors: recency (session history), popularity/centrality, user prefs

Posterior: p(v∣u) ∝ p(u∣v)p(v); normalize over candidate v

Drills

Start with uniform priors; inject a recency bump; watch the MAP flip with a vague query.

Plot posterior over top-5 candidates; compute entropy; decide “answer vs ask.”

Phase 4 — Decision policy and EIG

Goal: pick Answer / Ask / Search like an adult, not a heuristic.

Measure uncertainty: entropy over task, referent, critical slots

Ask(slot i): expected entropy drop from getting value of slot i

Search: expected change in likelihood mass from external retrieval

Answer: act if P(correct) clears threshold; include action costs

Drills

Given a posterior with two near-ties on referent, estimate which slot question cuts entropy most.

Assign simple costs: ask=1, search=2, answer=0; compute utility = EIG − cost; pick argmax.

Show one counterexample where high EIG isn’t worth the latency cost.

Phase 5 — Evaluation harness

Goal: keep yourself honest.

Gold mini-set: 30 queries with intended v*

Metrics: top-1 accuracy on t and e, calibration (reliability curve), action optimality vs oracle

Ablations: NB only, vMF only, combined; priors on/off

Drills

Write 5 ambiguous queries and label acceptable multiple v.

Log a confusion matrix for tasks; identify your degeneracies.

Stress test with OOD entity names; check kappa decay saves you from overconfidence.

Revision: “One Ontology,” slots, and reifying without turning into a whiteboard cult
The tiny upper ontology (reusable)

Nodes: Entity, Type, Attribute, Value, Document, Procedure, Step, User

Edges: INSTANCE_OF(Entity→Type), HAS_SLOT(Type→Attribute), HAS_VALUE(Entity→Value), ABOUT(Document→Entity), HAS_STEP(Procedure→Step), NEXT_IF(Step→Step), LIKES(User→Entity)

Keep it boring. If a pattern doesn’t help the decider today, it doesn’t go in.

What is a slot?

A named requirement a task needs to resolve the user’s intent.

Examples: for lookup(cast, film=e), slots might be {year?, actor?} where ? means unknown.

Represent in graph:

(:Type {name:"Film"})-[:HAS_SLOT]->(:Attribute {name:"year"})

Slot state lives in your belief over z: known, unknown, or a distribution over values.

Reification (only when you must)

Turn an edge or claim into a node so you can attach metadata like time, source, or uncertainty.

Example: a user rating you want to timestamp and trust-score:

(:User)-[:MADE]->(:Rating {score:8, ts:…, confidence:0.8})-[:OF]->(:Film)

Another: a textual mention of an entity with provenance:

(:Sentence)-[:MENTIONS]->(:Mention {offset:…, conf:0.7})-[:OF]->(:Entity)

If you’re not attaching metadata, don’t reify. Footnotes can’t save a broken graph.

What you’ll produce at each checkpoint

C0 Pocket sheet: one-page cheat sheet with Bayes, entropy, precision definitions.

C1 Likelihood mini-lib: two scorers (NB, vMF) with 12 demo entities.

C2 Posterior + entropy: returns top-k v and entropy numbers. No decisions yet.

C3 Policy: Answer/Ask/Search with a tiny EIG approximation and action costs.

C4 Eval: 30-query harness, ablations, and a plot or two. Then you’re allowed to say “active inference” in public.