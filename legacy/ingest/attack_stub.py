from __future__ import annotations

from typing import Dict, Any, List


def build_attack_stub() -> Dict[str, List[Dict[str, Any]]]:
    """Return a tiny ATT&CK-like stub with one technique and one mitigation.

    Nodes: Technique T0000, Mitigation M0000
    Edge: (Mitigation)-[:MITIGATES]->(Technique)
    """
    nodes: List[Dict[str, Any]] = [
        {
            "id": "technique:T0000",
            "labels": ["Entity", "Technique"],
            "properties": {"name": "Stub Technique"},
        },
        {
            "id": "mitigation:M0000",
            "labels": ["Entity", "Mitigation"],
            "properties": {"name": "Stub Mitigation"},
        },
    ]
    relationships: List[Dict[str, Any]] = [
        {"type": "MITIGATES", "start": "mitigation:M0000", "end": "technique:T0000", "properties": {}},
    ]
    return {"nodes": nodes, "relationships": relationships}


