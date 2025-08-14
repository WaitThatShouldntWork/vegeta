from __future__ import annotations

from ingest.cyber_etl import load_cyber_snapshot
from ingest.working_set import build_working_set


def test_working_set_includes_neighbor() -> None:
    snap = load_cyber_snapshot("mini")
    ws = build_working_set(snap, seeds=["CVE-2025-TEST"]) 
    ids = {n["id"] for n in ws["nodes"]}
    assert "CVE-2025-TEST" in ids
    assert "product:demoapp" in ids


