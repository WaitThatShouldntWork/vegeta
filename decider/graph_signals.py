from __future__ import annotations

from typing import Any, Dict, Optional

from db.neo4j_client import get_driver
from ingest.working_set import build_working_set_from_neo4j


def get_cve_graph_signals(cve_id: str) -> Optional[Dict[str, Any]]:
    """Return signals for a CVE from Neo4j: cvss (float), epss (float), kev_flag (bool).

    Returns None if Neo4j is not configured.
    """
    driver = get_driver()
    if driver is None:
        return None
    with driver.session() as session:
        row = session.run(
            """
            MATCH (c:CVE {id:$id})
            RETURN c.cvss AS cvss, c.epss AS epss, coalesce(c.kev_flag,false) AS kev
            """,
            id=cve_id,
        ).single()
        if row is None:
            return {"cvss": None, "epss": None, "kev_flag": False}
        return {"cvss": row["cvss"], "epss": row["epss"], "kev_flag": bool(row["kev"])}


def get_centrality_prior(cve_id: str) -> Optional[float]:
    """Compute a lightweight centrality prior in [0,1] from the working set.

    Fallback approximation when GDS is not used: normalize degree among neighbors.
    Returns None if Neo4j is not configured or no neighbors.
    """
    ws = build_working_set_from_neo4j(cve_id=cve_id, max_nodes=500, max_rels=2000)
    nodes = ws.get("nodes", [])
    rels = ws.get("relationships", [])
    if not nodes:
        return None
    # Degree map
    deg: Dict[str, int] = {}
    for r in rels:
        deg[r["start"]] = deg.get(r["start"], 0) + 1
        deg[r["end"]] = deg.get(r["end"], 0) + 1
    if not deg:
        return 0.0
    # Consider neighbors of the CVE only
    neighbor_ids = {r["end"] for r in rels if r["start"] == cve_id} | {r["start"] for r in rels if r["end"] == cve_id}
    if not neighbor_ids:
        return 0.0
    peak = max(deg.get(nid, 0) for nid in neighbor_ids)
    # Normalize by max degree in working set to [0,1]
    max_deg = max(deg.values()) or 1
    prior = float(peak) / float(max_deg)
    return prior


def get_cve_graph_signals_batch(cve_ids: list[str]) -> Dict[str, Dict[str, Any]]:
    """Batch fetch cvss/epss/kev_flag for a list of CVE ids. Returns id->signals.

    Empty dict if Neo4j is not configured.
    """
    driver = get_driver()
    if driver is None or not cve_ids:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with driver.session() as session:
        rows = session.run(
            """
            UNWIND $ids AS id
            MATCH (c:CVE {id:id})
            RETURN c.id AS id, c.cvss AS cvss, c.epss AS epss, coalesce(c.kev_flag,false) AS kev
            """,
            ids=cve_ids,
        ).data()
        for r in rows:
            out[r["id"]] = {"cvss": r.get("cvss"), "epss": r.get("epss"), "kev_flag": bool(r.get("kev"))}
    # Fill missing ids with defaults
    for cid in cve_ids:
        if cid not in out:
            out[cid] = {"cvss": None, "epss": None, "kev_flag": False}
    return out


