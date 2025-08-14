from __future__ import annotations

from ingest.cyber_etl import build_snapshot_from_inputs, sample_inputs_mini


def test_build_snapshot_from_inputs_shapes() -> None:
    data = sample_inputs_mini()
    snap = build_snapshot_from_inputs(**data)
    ids = {n["id"] for n in snap["nodes"]}
    assert "CVE-2025-TEST" in ids
    assert any(r["type"] == "AFFECTS" for r in snap["relationships"])


