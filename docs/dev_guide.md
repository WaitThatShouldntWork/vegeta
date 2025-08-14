## Dev quick start (Windows)

1) Install Python 3.11+ and ensure `python` is on PATH.
2) From repo root, run in PowerShell:

```powershell
scripts/setup.ps1 -Dev
```

3) Activate venv and run the demo CLI:

```powershell
.venv\Scripts\Activate.ps1
vegeta decide --domain cyber --cve CVE-2025-TEST

# Optional: force a search and preview docs
vegeta decide --domain cyber --cve CVE-2025-TEST --force-action search --retrieve --top-k 2
```

Notes:
- Uses `uv` for fast env + installs. The CLI is a stub to validate wiring.
- Next steps: add `schema/`, `decider/`, `ingest/`, and `retrieval/` per `README.md`.

### Optional: configure Neo4j Aura credentials
- Create a `.env` at repo root with:
  - `AURA_URI=neo4j+s://...`
  - `AURA_USER=...`
  - `AURA_PASSWORD=...`
- Or set env vars in your PowerShell session:
  ```powershell
  $env:AURA_URI = "neo4j+s://..."; $env:AURA_USER = "neo4j"; $env:AURA_PASSWORD = "..."
  ```
- These are read by `config/settings.py` when Aura-backed features are enabled.
- Optional connectivity check (if `neo4j` driver installed): add a small snippet:
  ```python
  from db.neo4j_client import check_connectivity
  print(check_connectivity())
  ```

### Eval quick run
- Minimal scenarios file: `eval/scenarios_cyber.jsonl` (JSONL)
- Run:
  ```powershell
  .\.venv\Scripts\python.exe eval\run_eval.py
  ```
- Output: JSON summary with counts of Answer/Ask/Search choices

### Mini cyber ingest
- Build a tiny cyber snapshot, print counts:
  ```powershell
  vegeta ingest-cyber --mini
  ```
- Upsert to Aura (requires `AURA_URI`, `AURA_USER`, `AURA_PASSWORD`):
  ```powershell
  scripts/ingest_cyber.ps1 -Upsert
  ```

### Triage quick check
```powershell
vegeta triage --cvss 8.1 --epss 0.5 --kev --asset-present --internet-exposed
```
Outputs a simple recommendation and risk score.


### Optional: local LLM via Ollama
- Install Ollama and pull a small model (e.g., `gemma2:4b` or `gemma3:12b`).
- We will route question generation and fact extraction to a local endpoint when available; otherwise we use template fallbacks.


