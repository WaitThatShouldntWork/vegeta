from __future__ import annotations

from pathlib import Path
from typing import List
import json

from db.neo4j_client import get_driver


def gen_scenarios_from_neo4j(
    output: Path,
    kev_limit: int = 50,
    epss_limit: int = 50,
    min_epss: float = 0.6,
) -> int:
    """Generate scenarios JSONL by sampling KEV CVEs and high-EPSS CVEs from Neo4j.

    Returns number of scenarios written. No-op if Neo4j is not configured.
    """
    driver = get_driver()
    if driver is None:
        return 0

    rows: List[dict] = []
    with driver.session() as session:
        kev_rows = session.run(
            """
            MATCH (c:CVE)
            WHERE coalesce(c.kev_flag,false) = true
            RETURN c.id AS cve
            LIMIT $limit
            """,
            limit=kev_limit,
        ).data()
        rows.extend({"domain": "cyber", "cve": r["cve"], "seed": 42} for r in kev_rows)

        epss_rows = session.run(
            """
            MATCH (c:CVE)
            WHERE coalesce(c.epss,0.0) >= $min_epss
            RETURN c.id AS cve
            LIMIT $limit
            """,
            min_epss=min_epss,
            limit=epss_limit,
        ).data()
        rows.extend({"domain": "cyber", "cve": r["cve"], "seed": 7} for r in epss_rows)

        # Fallbacks if too few scenarios
        if len(rows) < 100:
            cvss_rows = session.run(
                """
                MATCH (c:CVE)
                WHERE coalesce(c.cvss,0.0) >= 7.0
                RETURN c.id AS cve
                LIMIT $limit
                """,
                limit=max(0, 300 - len(rows)),
            ).data()
            rows.extend({"domain": "cyber", "cve": r["cve"], "seed": 13} for r in cvss_rows)

        if len(rows) < 100:
            any_rows = session.run(
                """
                MATCH (c:CVE)
                RETURN c.id AS cve
                LIMIT $limit
                """,
                limit=max(0, 300 - len(rows)),
            ).data()
            rows.extend({"domain": "cyber", "cve": r["cve"], "seed": 21} for r in any_rows)

    # Write JSONL
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


