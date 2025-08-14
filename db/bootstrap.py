from __future__ import annotations

from pathlib import Path
from typing import List

from db.neo4j_client import get_driver


def _split_cypher(script: str) -> List[str]:
    # Naive split on semicolons; ignores lines starting with // and empty lines
    parts: List[str] = []
    current: List[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            parts.append("\n".join(current).rstrip(";\n\r "))
            current = []
    if current:
        parts.append("\n".join(current))
    return parts


def apply_bootstrap(cypher_file: Path = Path("schema/bootstrap.cypher")) -> int:
    """Apply bootstrap Cypher statements to Neo4j. Returns count of executed statements."""
    driver = get_driver()
    if driver is None:
        return 0
    script = cypher_file.read_text(encoding="utf-8")
    statements = _split_cypher(script)
    applied = 0
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)
            applied += 1
    return applied


