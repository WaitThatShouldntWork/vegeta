from __future__ import annotations

from ingest.cyber_etl import load_cyber_snapshot


def test_load_cyber_snapshot_mini() -> None:
    snap = load_cyber_snapshot("mini")
    assert "nodes" in snap and "relationships" in snap and "meta" in snap
    assert any("CVE-2025-TEST" == n["id"] for n in snap["nodes"])
    assert any(r["type"] == "AFFECTS" for r in snap["relationships"])


def test_load_cyber_snapshot_empty() -> None:
    snap = load_cyber_snapshot("empty")
    assert snap["nodes"] == []
    assert snap["relationships"] == []


