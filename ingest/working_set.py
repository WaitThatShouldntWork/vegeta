from __future__ import annotations

from typing import Dict, Iterable, List, Set, Any

from ingest.cyber_etl import Snapshot
from db.neo4j_client import get_driver


def build_working_set(snapshot: Snapshot, seeds: Iterable[str], max_nodes: int = 100) -> Snapshot:
    """Return a subgraph containing seeds and their 1-hop neighbors.

    Enforces a simple max_nodes cap; if exceeded, truncates deterministically.
    """
    seed_set: Set[str] = set(seeds)
    id_to_node: Dict[str, dict] = {n["id"]: n for n in snapshot["nodes"]}

    kept_nodes: Set[str] = set()
    for sid in seed_set:
        if sid in id_to_node:
            kept_nodes.add(sid)

    # 1-hop neighbors via relationships
    for rel in snapshot["relationships"]:
        if rel["start"] in seed_set:
            kept_nodes.add(rel["end"])
        if rel["end"] in seed_set:
            kept_nodes.add(rel["start"])

    # Enforce cap
    kept_list = sorted(kept_nodes)[:max_nodes]
    kept_set = set(kept_list)

    nodes = [id_to_node[nid] for nid in kept_list if nid in id_to_node]
    relationships = [
        r
        for r in snapshot["relationships"]
        if r["start"] in kept_set and r["end"] in kept_set
    ]

    return {"nodes": nodes, "relationships": relationships, "meta": {"source": "working_set"}}


def build_working_set_from_neo4j(cve_id: str, max_nodes: int = 100, max_rels: int = 500) -> Snapshot:
    """Pull a bounded 1-hop subgraph around a CVE from Neo4j.

    Whitelisted predicates are assumed by the data we ingest. Keeps counts under caps.
    """
    driver = get_driver()
    if driver is None:
        return {"nodes": [], "relationships": [], "meta": {"source": "working_set:aura:none"}}

    nodes: List[Dict[str, Any]] = []
    rels: List[Dict[str, Any]] = []
    with driver.session() as session:
        # Collect nodes (CVE and 1-hop neighbors)
        node_rows = session.run(
            """
            MATCH (c:CVE {id:$id})
            OPTIONAL MATCH (c)-[r]->(n)
            WITH collect(DISTINCT c) + collect(DISTINCT n) AS ns
            UNWIND ns AS n
            WITH DISTINCT n
            WHERE n IS NOT NULL
            RETURN n.id AS id, labels(n) AS labels, n AS props
            LIMIT $max_nodes
            """,
            id=cve_id,
            max_nodes=max_nodes,
        ).data()
        for row in node_rows:
            props = dict(row["props"])
            props.pop("id", None)
            nodes.append({"id": row["id"], "labels": row["labels"], "properties": props})

        node_ids = {n["id"] for n in nodes}

        # Collect relationships among those nodes
        rel_rows = session.run(
            """
            UNWIND $ids AS i
            MATCH (a {id:i})-[e]->(b)
            WHERE b.id IN $ids
            RETURN type(e) AS type, a.id AS start, b.id AS end, e AS props
            LIMIT $max_rels
            """,
            ids=list(node_ids),
            max_rels=max_rels,
        ).data()
        for row in rel_rows:
            rels.append({
                "type": row["type"],
                "start": row["start"],
                "end": row["end"],
                "properties": dict(row["props"]),
            })

    return {"nodes": nodes, "relationships": rels, "meta": {"source": "working_set:aura"}}


