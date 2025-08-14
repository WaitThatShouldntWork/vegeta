## TODO (live task list)

- product requiremnt doc/vision of the project in README.md and architecture.md


### Next
- [x] Schema/metamodel: `schema/metamodel.yaml`; `schema/lint_schema.py`
 - [x] Neo4j bootstrap: `schema/bootstrap.cypher` (CLI: `vegeta bootstrap`)
 - [x] Add basic tests (TTD): CLI and schema linter
  - [ ] Improve one-step eval accuracy (target >=0.2 immediate Answer rate)
    - [x] Lower answer_confidence_threshold to 0.65 in `decider/choose.py`
    - [x] Reduce Ask EIG and increase Ask cost in `decider/eig.py`
    - [x] Blend Bayesian EIG for `epss_high` alongside `actively_exploited`
    - [x] Reduce Search EIG baseline in `decider/eig.py`
    - [x] Make eval use risk-derived entropy and resolved slots like CLI
    - [x] Re-run eval and record metrics in `eval/results/simple.json` and `eval/results/ask.json`
    - [x] One-step metric: report `answer_rate` (not accuracy)
  - [ ] Ingest: `ingest/cyber_etl.py` (NVD, CPE, KEV, EPSS, ATT&CK)
  - [x] Learning loop
    - [x] `features/` adapters (base + cyber)
    - [x] `policy/` (contextual bandit + trainer)
    - [x] `env/interaction.py` (noisy user + Bayesian update)
    - [x] `agent/question_selector.py` (slot EIG)
    - [x] CLI: `vegeta train`, `vegeta simulate`
    - [x] Integrate learned policy into `decider/choose.py` when weights exist
  - [ ] Active loop (EMG + Wikipedia + LLM)
    - [x] `retrieval/wikipedia.py` basic retriever
    - [x] `llm/ollama_client.py` (local model stub) for question gen
    - [x] CLI: `vegeta loop` (uncertainty→question→search→write-back, JSONL logs)
    - [x] Graph write-back: reified `:Fact` + `HAS_SOURCE` + lexical nodes
    - [x] Seed dataset (EMG) and test episodes measuring steps/questions/searches
    - [ ] Optional: LangGraph agent (ReAct) wrapper for tool orchestration
    - [x] Slot diversity across steps (exclude already-asked)
    - [x] Replace Neo4j deprecated id() with elementId()
    - [ ] Existence checks: skip retrieval if slot/fact known with confidence≥τ
    - [ ] Smarter upsert: merge additional HAS_SOURCE on existing Fact; bump confidence
    - [ ] Lexical scoping: unique (doc_url, order), allocate next order; avoid collisions
    - [ ] Stop condition: end loop early when evidence threshold reached (entropy/EIG)
    - [ ] Fill templates with context vars (e.g., `{asset_name}`) when assets are provided
    - [ ] EMG loader: ingest `data/emg/seed_entities.jsonl` into graph with Types/Slots
    - [ ] CLI: `vegeta loop-emg --episodes data/emg/episodes.jsonl` to benchmark steps/asks/searches
 - [x] Retrieval stubs: `retrieval/base.py`, `retrieval/local_snapshot.py`
  - [x] Base interface and local stub + unit test
- [ ] Wire retrieval into decider when searching
  - [x] Flagged path with preview and tests
 - [x] Evaluation harness: `eval/run_eval.py`; `eval/scenarios_cyber.jsonl`
   - [x] Minimal runner and stub scenarios
 - [ ] Cyber domain consolidation (dataset + questions)
   - [x] README: cyber slice (CVE, Product, Vendor, Technique, Event; AFFECTS/MITIGATES/etc.)
  - [x] Expand `domain_packs/cyber/questions.yaml` to 12–15
   - [x] Add `domain_packs/cyber/mappings.py` (ETL mappers)
   - [x] Identify retrieval targets: vendor advisories, KEV updates, EPSS feed
    - [x] README: add CVSS/EPSS/KEV source and scoring explanation

- [ ] Cyber ETL (Aura‑ready)
  - [x] NVD CVE JSON v2: CVE nodes (id, dates, cvss), parse CPEs → `AFFECTS` edges
  - [x] CPE mapping: Vendor/Product nodes; `PRODUCES` Vendor→Product
  - [x] CISA KEV: `kev_flag=true` on CVE; add `:Event` and `APPLIES_TO`
  - [x] EPSS daily CSV: `epss`, `epss_percentile` on CVE
  - [ ] ATT&CK STIX: Technique/Mitigation nodes; `MITIGATES`, `RELATED_TO` (subset included via flag)
  - [x] KEV and EPSS enrichers wired to NVD fetch path
  - [x] Upsert + snapshot label `vYYYY_MM_DD`; ensure constraints (`vegeta bootstrap`)
  - [x] Use Aura creds from env (`AURA_URI`/`NEO4J_URI`, `AURA_USER`, `AURA_PASSWORD`)

### Done
- [x] Docs: explain simulated user, question source, and learning loop plan
- [x] Append EMG Ontology Brief to `docs/project_brief.md`
- [x] Add docs: `docs/policy.md`, `docs/domain_pack_cyber.md` (short placeholders)
- [x] Create `domain_packs/cyber/` with `config.yaml`, `questions.yaml` (skeleton)
- [x] CLI: accept `--assets` JSON and echo a couple fields in the decision output
- [x] Project metadata with CLI entrypoint (`pyproject.toml` → `vegeta`)
- [x] CLI stub `cli/decide.py` wired to `decider.choose.choose_action`
- [x] Windows setup script `scripts/setup.ps1`
- [x] Dev quick start `docs/dev_guide.md`
- [x] Package scaffolding: `decider/`, `schema/`, `ingest/`, `retrieval/`
- [x] `.gitignore`
- [x] Minimal tests: CLI assets echo, schema linter whitelist


