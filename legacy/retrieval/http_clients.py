from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Cost:
    token_cost: float = 1.0
    latency_weight: float = 0.001


def fetch_vendor_advisories(cve_id: str, max_results: int = 3) -> List[str]:
    """Stub: return an empty list; to be implemented with real HTTP clients.

    Include costs via the Cost dataclass when integrating with the policy.
    """
    _ = (cve_id, max_results)
    return []


