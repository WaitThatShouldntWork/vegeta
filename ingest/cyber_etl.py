from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict, Iterable, Any, Optional

from domain_packs.cyber.mappings import (
    map_nvd_cve_to_node,
    map_cpe_to_product,
    map_cve_affects_edge,
    map_vendor_node,
    map_vendor_produces_edge,
)
from db.neo4j_client import get_driver


class Node(TypedDict):
    id: str
    labels: List[str]
    properties: Dict[str, object]


class Relationship(TypedDict):
    type: str
    start: str
    end: str
    properties: Dict[str, object]


class Snapshot(TypedDict):
    nodes: List[Node]
    relationships: List[Relationship]
    meta: Dict[str, object]


def load_cyber_snapshot(sample: Literal["mini", "empty"] = "mini") -> Snapshot:
    """Return a tiny in-memory cyber snapshot for tests and demos.

    sample="mini" includes one CVE, one Product, and an AFFECTS relation.
    """
    if sample == "empty":
        return {"nodes": [], "relationships": [], "meta": {"sample": sample}}

    nodes: List[Node] = [
        {
            "id": "CVE-2025-TEST",
            "labels": ["Entity", "CVE"],
            "properties": {"cvss": 7.5, "epss": 0.2, "kev_flag": False},
        },
        {
            "id": "product:demoapp",
            "labels": ["Entity", "Product"],
            "properties": {"name": "DemoApp"},
        },
    ]
    relationships: List[Relationship] = [
        {
            "type": "AFFECTS",
            "start": "CVE-2025-TEST",
            "end": "product:demoapp",
            "properties": {},
        }
    ]
    meta: Dict[str, object] = {"sample": sample, "version": "0.1.0"}
    return {"nodes": nodes, "relationships": relationships, "meta": meta}


# ----- New minimal ETL helpers (data -> snapshot) -----

def build_snapshot_from_inputs(
    cves: Iterable[Dict[str, Any]],
    cpe_by_cve: Dict[str, List[str]],
    kev_cves: Iterable[str] | None = None,
    epss_by_cve: Dict[str, float] | None = None,
    epss_pct_by_cve: Dict[str, float] | None = None,
) -> Snapshot:
    kev_set = set(kev_cves or [])
    nodes: List[Node] = []
    rels: List[Relationship] = []

    product_ids: Dict[str, Node] = {}
    vendor_ids: Dict[str, Node] = {}

    for cve in cves:
        cve_node = map_nvd_cve_to_node(cve)
        if cve_node["id"] in kev_set:
            cve_node["properties"]["kev_flag"] = True
        if epss_by_cve and cve_node["id"] in epss_by_cve:
            cve_node["properties"]["epss"] = float(epss_by_cve[cve_node["id"]])
        if epss_pct_by_cve and cve_node["id"] in epss_pct_by_cve:
            cve_node["properties"]["epss_percentile"] = float(epss_pct_by_cve[cve_node["id"]])
        nodes.append(cve_node)

        for cpe23 in cpe_by_cve.get(cve_node["id"], []):
            prod_node = map_cpe_to_product(cpe23)
            if prod_node["id"] not in product_ids:
                product_ids[prod_node["id"]] = prod_node
            rels.append(map_cve_affects_edge(cve_node["id"], prod_node["id"]))
            # Vendor
            v_node = map_vendor_node(prod_node["properties"].get("vendor"))
            if v_node and v_node["id"] not in vendor_ids:
                vendor_ids[v_node["id"]] = v_node
            if v_node:
                rels.append(map_vendor_produces_edge(v_node["id"], prod_node["id"]))

    nodes.extend(product_ids.values())
    nodes.extend(vendor_ids.values())

    # Add Event nodes for KEV CVEs
    for cve_id in kev_set:
        event_id = f"event:kev:{cve_id}"
        nodes.append({
            "id": event_id,
            "labels": ["Event"],
            "properties": {"source": "CISA KEV"},
        })
        rels.append({
            "type": "APPLIES_TO",
            "start": event_id,
            "end": cve_id,
            "properties": {},
        })
    return {"nodes": nodes, "relationships": rels, "meta": {"source": "build_inputs"}}


def upsert_snapshot_to_neo4j(snapshot: Snapshot, snapshot_label: Optional[str] = None) -> Dict[str, int]:
    """Upsert nodes/rels into Neo4j if driver available. Returns counts.

    Labels: use base labels from node["labels"], plus optional snapshot label.
    """
    driver = get_driver()
    if driver is None:
        return {"nodes": 0, "relationships": 0}

    node_count = 0
    rel_count = 0
    lbl_suffix = f":{snapshot_label}" if snapshot_label else ""

    with driver.session() as session:
        # Constraints assumed from schema/bootstrap.cypher
        for n in snapshot["nodes"]:
            labels = ":".join(n.get("labels", ["Entity"]))
            q = f"MERGE (x:{labels}{lbl_suffix} {{id:$id}}) SET x += $props"
            session.run(q, id=n["id"], props=n.get("properties", {}))
            node_count += 1

        for r in snapshot["relationships"]:
            q = (
                f"MATCH (a {{id:$start}}), (b {{id:$end}}) "
                f"MERGE (a)-[e:{r['type']}]->(b) SET e += $props"
            )
            session.run(q, start=r["start"], end=r["end"], props=r.get("properties", {}))
            rel_count += 1

    return {"nodes": node_count, "relationships": rel_count}


def sample_inputs_mini() -> Dict[str, Any]:
    """Small input bundle for tests/demos."""
    cves = [
        {"id": "CVE-2025-TEST", "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]}},
        {"id": "CVE-2025-OTHER", "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 5.0}}]}},
    ]
    cpe_by_cve = {
        "CVE-2025-TEST": ["cpe:2.3:a:demo:demoapp:*:*:*:*:*:*:*:*"] ,
        "CVE-2025-OTHER": ["cpe:2.3:a:acme:widget:*:*:*:*:*:*:*:*"] ,
    }
    kev_cves = {"CVE-2025-TEST"}
    epss_by_cve = {"CVE-2025-TEST": 0.21, "CVE-2025-OTHER": 0.12}
    return {"cves": cves, "cpe_by_cve": cpe_by_cve, "kev_cves": kev_cves, "epss_by_cve": epss_by_cve}



