from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
import urllib.request


ATTACK_ENTERPRISE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"


def fetch_attack_stix(url: str = ATTACK_ENTERPRISE_URL) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as resp:  # nosec - controlled URL
        return json.load(resp)


def parse_attack(data: Dict[str, Any], limit: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse STIX bundle into Technique and Mitigation nodes and MITIGATES edges.

    Returns (nodes, edges). Limits total techniques to `limit` for responsiveness.
    """
    objects = data.get("objects", []) or []
    id_to_obj: Dict[str, Dict[str, Any]] = {}
    techniques: Dict[str, Dict[str, Any]] = {}
    mitigations: Dict[str, Dict[str, Any]] = {}
    nodes: List[Dict[str, Any]] = []
    rels: List[Dict[str, Any]] = []

    for obj in objects:
        stix_id = obj.get("id")
        if not stix_id:
            continue
        id_to_obj[stix_id] = obj
        if obj.get("type") == "attack-pattern":
            # external_id like T1234
            ext_id = _extract_external_id(obj)
            if ext_id and ext_id.startswith("T"):
                techniques[stix_id] = obj
        elif obj.get("type") == "course-of-action":
            ext_id = _extract_external_id(obj)
            if ext_id and (ext_id.startswith("M") or ext_id.startswith("COA")):
                mitigations[stix_id] = obj

    # Limit techniques for responsiveness
    allowed_tech_ids = set(list(techniques.keys())[: max(0, limit)])

    for stix_id, obj in techniques.items():
        if stix_id not in allowed_tech_ids:
            continue
        ext_id = _extract_external_id(obj) or obj.get("id")
        name = obj.get("name", "Technique")
        nodes.append({
            "id": f"technique:{ext_id}",
            "labels": ["Entity", "Technique"],
            "properties": {"name": name},
        })

    for stix_id, obj in mitigations.items():
        ext_id = _extract_external_id(obj) or obj.get("id")
        name = obj.get("name", "Mitigation")
        nodes.append({
            "id": f"mitigation:{ext_id}",
            "labels": ["Entity", "Mitigation"],
            "properties": {"name": name},
        })

    # Relationships of type mitigates
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "mitigates":
            continue
        src = obj.get("source_ref")  # typically course-of-action
        tgt = obj.get("target_ref")  # typically attack-pattern
        if src in mitigations and tgt in allowed_tech_ids:
            src_ext = _extract_external_id(mitigations[src]) or src
            tgt_ext = _extract_external_id(techniques[tgt]) or tgt
            rels.append({
                "type": "MITIGATES",
                "start": f"mitigation:{src_ext}",
                "end": f"technique:{tgt_ext}",
                "properties": {},
            })

    return nodes, rels


def _extract_external_id(obj: Dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []) or []:
        ext_id = ref.get("external_id")
        if isinstance(ext_id, str):
            return ext_id
    return None


