# VEGETA Detailed Process Flow

This diagram shows how the system runs end-to-end, including the active loop, what each function does, and how modules interact.

```mermaid
flowchart TD
  subgraph CLI
    A0["decide"] --> A1
    A2["loop"] --> A3
    A4["loop-emg"] --> A5
  end

  subgraph Core
    C1["decider/choose.choose_action"]
    C2["decider/confidence.compute_answer_confidence"]
    C3["decider/eig.score_* + expected_entropy_drop_bayes"]
    C4["decider/bayes.build_prior_from_signals"]
  end

  subgraph Graph
    G1["db/neo4j_client.get_driver"]
    G2["decider/graph_signals.get_cve_graph_signals"]
    G3["db/writeback.upsert_fact_with_source + upsert_document_with_sentence + link_sentence_mentions_entity + fact_exists / has_slot_value"]
    G4["emg/loader.load_seed_entities"]
    G5["emg/query.find_films_by_slots"]
  end

  subgraph Retrieval
    R1["retrieval/wikipedia.search_wikipedia"]
    R2["llm/ollama_client.generate (text2q / future text2cypher)"]
  end

  subgraph Questions
    Q1["agent/question_selector.pick_next_slot_by_eig"]
    Q2["agent/question_generator.render_from_domain_pack"]
  end

  %% decide (one-shot)
  A0 -->|"reads CVE signals"| G2 --> C4 --> C2 --> C3 --> C1 --> A1["render scores + decision"]

  %% loop (cyber)
  A2 -->|"init"| Q1 --> Q2 --> R1 -->|"docs"| A3
  A2 -->|"signals"| G2
  A3 -->|"write-back if new"| G3 -->|"update belief"| Q1

  %% loop-emg (movies)
  A4 -->|"seed optional"| G4 --> A5
  A5 -->|"filter by known"| G5 -->|"if <=1 match"| A5
  A5 -->|"ask slot when private"| Q1 --> Q2
  A5 -->|"search public fact"| R1 --> G3

  %% global deps
  G1 -.-> G2
  G1 -.-> G3
```

Legend:
- **decide**: one-shot scoring (no loop). Uses confidence + EIG to choose Answer/Ask/Search.
- **loop**: uncertainty → question → search → write-back; repeats until early stop.
- **loop-emg**: movie episodes (identify_film / retrieve_award). Filters by known slots, asks unknown, searches once, writes back provenance.
- **Private vs public slots**: private → Ask only; public → Search.
