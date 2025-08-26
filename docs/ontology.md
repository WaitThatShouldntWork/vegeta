# VEGETA Ontology (Editable Memory Graph, EMG)

This is a small, reusable schema used across domains (movies, cyber, etc.). It keeps the core vocabulary tiny, treats uncertainty as first‑class, and always ties facts to their sources.

## Purpose
- Represent real‑world things as `:Entity` with a lightweight `:Type`.
- Store simple attributes as slots (`:Entity-[:HAS_SLOT]->:SlotValue`) or reify claims as `:Fact` (`SUBJECT`/`PREDICATE`/`OBJECT`) when we need provenance/uncertainty.
- Track provenance via `:Document` and a lexical hierarchy (`:Section` → `:Paragraph` → `:Sentence`).
- Drive decisions by asking/searching to reduce uncertainty (EIG), then write back new/verified facts.

## Core Labels
- `:Entity` — concrete things (Film, Person, Company, CVE, Product)
- `:Type` — class/taxonomy nodes (Film, Person, Award, Country, CVE, Product)
- `:SlotValue` — compact values for MetaSlots (e.g., Country=Japan)
- `:Fact` — reified claims with confidence and source; linked via `SUBJECT`/`PREDICATE`/`OBJECT`
- `:RelationType` — registry of predicate names (e.g., WON_AWARD, RELATED_TO)
- `:Document` / `:Section` / `:Paragraph` / `:Sentence` — lexical hierarchy for sources
- `:Checklist` — typed requirement sets for answering specific questions (procedural templates)
- `:SlotSpec` — requirements within a checklist (what must be present/validated)
- `:Goal` — long-term user ambitions for personalization and performance improvement
- (Optional) `:Procedure`, `:Step`, `:Tool` — procedural memory layer (preconditions/effects)

## Relationship Primitives (≈15)
- `(:Entity)-[:INSTANCE_OF]->(:Type)`
- `(:Entity)-[:HAS_SLOT]->(:SlotValue {slot, value, confidence?, source?})`
- `(:Fact)-[:SUBJECT]->(:Entity)`
- `(:Fact)-[:PREDICATE]->(:RelationType)`
- `(:Fact)-[:OBJECT]->(:Entity | :SlotValue | :Type)`
- `(:Fact)-[:HAS_SOURCE {support}]->(:Document)`
- Lexical: `:Document-[:HAS_SECTION]->:Section-[:HAS_PARAGRAPH]->:Paragraph-[:HAS_SENTENCE]->:Sentence`
- Mentions: `(:Sentence)-[:MENTIONS {confidence, via}]->(:Entity)`
- Checklists: `(:Checklist)-[:REQUIRES]->(:SlotSpec)`, `(:SlotSpec)-[:VALIDATES]->(:Entity | :SlotValue)`
- Goals: `(:Goal)-[:GOAL_OF]->(:User)`, `(:Goal)-[:INFLUENCES]->(:Checklist)`
- (Optional) procedural: `:Procedure-[:HAS_STEP]->:Step`, `:Step-[:REQUIRES]->(:SlotValue or Condition)`

## Keys & Constraints (typical)
- Unique: `:Type(name)`, `:RelationType(name)`, `:Document(source_url)`, `(:SlotValue(slot, value))`, `:Checklist(name)`, `:Goal(name, user_id)`
- Lexical scoping: unique `(Section.doc_url, order)`, `(Paragraph.doc_url, order)`, `(Sentence.doc_url, order)`
- SlotSpec scoping: unique `(checklist_name, slot_name)`
- Use `elementId()` for reporting ids (avoid legacy `id()`).

## Implementation guidance (movies demo)
- Treat categories like Genre as SlotValues, not entities
  - Use `(:Entity)-[:HAS_SLOT]->(:SlotValue {slot:"Genre", value})` instead of `(:Genre)` nodes for compact filters.
  - Keep domain relations only when they connect two entities (e.g., `(:Person)-[:ACTED_IN]->(:Film)`).
- Property conventions (for retrieval and terms)
  - `:Entity {id, name, aliases?: string[], summary?}`
  - Optional embeddings (populated by ingestion): `sem_emb?`, `graph_emb?`.
  - `[:HAS_SLOT {slot, value, confidence?, source_url?, timestamp?}]`.
- Minimal registries and indexes
  - `:Type(name)` unique; `:RelationType(name)` unique.
  - `:Document(source_url)` unique; lexical scoping unique on `(doc_url, order)` for `:Section/:Paragraph/:Sentence`.
  - Composite unique on `(:SlotValue(slot, value))`.
  - Full‑text index on `:Entity(name, aliases)`; optional vector indexes on `sem_emb`, `graph_emb`.
- Checklists (first‑class)
  - `:Checklist {name, description}` and `:SlotSpec {name, expect_labels, rel?, required, cardinality}` with scoping unique per checklist.
  - Use checklists to define required slots (what should exist) rather than pre‑creating placeholders.

## SlotValues (MetaSlots)
A `:SlotValue` captures compact, high‑utility attributes we ask about frequently.

- Movies: `GenreCluster`, `Era`, `Country`, `Studio`, `AwardsSignal`
- Cyber: `actively_exploited`, `epss_high`, `severity_high`, `internet_exposed`
- Slot edges can carry confidence and source: `(:Entity)-[hs:HAS_SLOT {confidence, source}]->(:SlotValue)`

Why slots:
- Fast filtering: slots form a small “question vocabulary” for 20Q‑style narrowing
- Portable: each domain picks a small slot set with clear semantics

### AwardsSignal (Movies)
- Coarse slot reflecting awards prominence: `High | Medium | Low`
- Stored as `HAS_SLOT` for quick disambiguation and ranking
- Concrete awards should be reified as `:Fact` (e.g., WON_AWARD) with `HAS_SOURCE` to documents
- Today: seed assigns AwardsSignal; future: derive from parsed awards facts or reputable lists

## Reified Facts & Provenance
- Use `:Fact(kind, confidence, source, timestamp)` with `SUBJECT`/`PREDICATE`/`OBJECT`
- Attach at least one `HAS_SOURCE` to a `:Document` with lexical breadcrumbs (`:Sentence` etc.)
- Example (Movies): WON_AWARD fact linking a Film to an Award, with a HAS_SOURCE to the Wikipedia page

## How EIG picks Ask vs Search (current behavior)
We prioritize actions that reduce uncertainty (entropy) the most, minus a small cost.

Uncertainty:
- Movies: entropy over the candidate film set (near‑term: unknown‑first + graph filtering; next: true EIG split by slot)
- Cyber: entropy over a small latent state; we already compute Bayesian EIG for `actively_exploited` and `epss_high`

Ask (private info):
- If a slot is inherently private (Cyber: `internet_exposed`), Ask the user

Search (public info):
- If a slot/fact is public (Movies: awards; Cyber: KEV/EPSS/patch/workaround), Search first (Wikipedia/Wikidata/vendor)
- Write back a `:Fact` with `HAS_SOURCE`, then update uncertainty and continue

## Write‑back Policy (quality & de‑dup)
- Facts: `MERGE` by `(kind, subject, object)`; on match, only bump confidence; always accumulate `HAS_SOURCE`
- Slots: ensure `HAS_SLOT.confidence = max(old, new)`; record `source`
- Lexical: MERGE `:Document` by `source_url`; compute the next `Section.order` to avoid collisions; always attach `:Sentence` (no orphans)

## Text2Cypher (planned retriever)
- If the graph lacks a needed fact, generate a short Cypher template (via local LLM/Ollama) and execute it safely
- Fall back to Wikipedia/Wikidata parsing if Cypher cannot answer
- This avoids bespoke retrievers and generalizes across domains

## Checklists (Procedural Templates)
Checklists define typed requirement sets for answering specific questions. They serve as explicit procedural memory that can be introspected and improved over time.

Structure:
- `:Checklist {name, description}` — e.g., "IdentifyFilm", "AssessCyberRisk"
- `:SlotSpec {name, expect_labels, rel, required, cardinality}` — individual requirements

Example (IdentifyFilm):
```cypher
(:Checklist {name:"IdentifyFilm"})
(:SlotSpec {name:"film", expect_labels:["Film"], rel:"INSTANCE_OF", required:true, cardinality:"ONE"})
(:SlotSpec {name:"year", expect_labels:["Year","Date"], required:false, cardinality:"ONE"})
(:SlotSpec {name:"actor", expect_labels:["Person"], required:false, cardinality:"MANY"})
```

Benefits:
- Explicit "completeness" definition for each task type
- Introspectable reasoning (why did we ask X?)
- Foundation for learned policies that skip steps when confidence is high
- Clear separation from world model (no ghost edges in your entity graph)

## Goals (Long-term Personalization)
Goals capture user ambitions that influence system behavior over time.

Two types:
1. **Dialogue intent** (session goal): identify, recommend, verify, explore, act
   - Keep as categorical variable `z_goal` in decider state
   - Drives ASK vs SEARCH vs ANSWER decisions

2. **User ambitions** (long-term goals): "watch AFI Top 100", "reduce security risk", "save for a house"
   - Store as `:Goal {name, horizon, priority}` nodes
   - Link via `(:Goal)-[:GOAL_OF]->(:User)`
   - Influence checklist design: `(:Goal)-[:INFLUENCES]->(:Checklist)`

Benefits:
- Bias priors and ASK design based on user preferences
- Personalize questioning strategy over time
- Track progress toward user objectives
- Optional for core decider; valuable for product experience

## Movies quick map
- Entity: Film nodes (`:Entity`) linked to `:Type {name:'Film'}` via `INSTANCE_OF`
- Slots: `GenreCluster`, `Era`, `Country`, `Studio`, `AwardsSignal`
- Facts: awards (WON_AWARD) → supported by `:Document` with lexical nodes and `:MENTIONS`
- Checklists: IdentifyFilm, RecommendFilm → define what info is needed for each task
- Goals: user film preferences → influence which checklists to use and how to prioritize slots
- Identify flow: select checklist → filter by known slots → ask unknown with highest EIG → confirmatory search → write back
