from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore

from config.settings import get_aura_settings


@dataclass
class Neo4jStatus:
    available: bool
    message: str


def get_driver():  # type: ignore[override]
    """Return a neo4j.Driver if credentials and dependency present; else None."""
    aura = get_aura_settings()
    if not (aura.uri and aura.user and aura.password):
        return None
    if GraphDatabase is None:
        return None
    return GraphDatabase.driver(aura.uri, auth=(aura.user, aura.password))


def check_connectivity() -> Neo4jStatus:
    drv = get_driver()
    if drv is None:
        return Neo4jStatus(False, "Neo4j not configured or driver missing (check AURA_URI/NEO4J_URI and driver install)")
    try:
        with drv.session() as session:
            result = session.run("RETURN 1 AS ok")
            _ = result.single()
        return Neo4jStatus(True, "Neo4j connectivity OK")
    except Exception as exc:
        return Neo4jStatus(False, f"Neo4j connectivity failed: {exc}")


