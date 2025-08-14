## VEGETA Overview

VEGETA is an Expected Information Gain (EIG) decision engine that chooses among three actions at each step: Answer, Ask, or Search. It maintains a simple belief over the current task state and picks the action expected to reduce uncertainty the most after accounting for costs.

See the visual architecture in `architecture.md`.

### Agreed tooling and platform
- Python 3.11+ managed with `uv`; PowerShell scripts for tasks on Windows
- Neo4j Aura (16–32 GB RAM); no local Docker needed

### Initial defaults (v1)
- Working set caps: ≤ 25k nodes, ≤ 250k relationships (configurable)
- EIG sampling: K = 12 simulated outcomes per Ask action
- Risk model: `cvss_norm = CVSS/10.0`; `risk_score = 0.5*cvss_norm + 0.4*EPSS + 0.1*KEV_flag`
- Cost weights: `ask.user_annoyance = 1.0`, `search.token_cost = 1.0`, `search.latency_weight = 0.001`
 - AuraDS GDS: run PageRank and/or fastRP on the working set to produce priors for confidence calibration

### Notes for beginners
- Discrete hypothesis set: a small list of possible "world states" (e.g., which procedure step we’re in, or whether an asset is internet-exposed). We track probabilities over this list and update them as we Ask/Search.
- Predicate whitelist: the allowed relationship types in the graph (e.g., `AFFECTS`, `MITIGATES`). Restricting to a whitelist keeps the schema clean and queries fast. We include `AFFECTS` to match the ingest plan.
- Working set: the bounded subgraph around the current task (seed entities and a few hops). Smaller by default for responsiveness; can be enlarged later.
 - AuraDS/GDS priors: GDS algorithms (PageRank/fastRP) assign scores/embeddings to nodes. We use these as lightweight priors/features to initialize confidence before any questions are asked.


