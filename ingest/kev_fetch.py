from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Set

import urllib.request


def fetch_kev_json(url: str = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json") -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:  # nosec - controlled URL
        return json.load(resp)


def extract_kev_cves(data: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for item in data.get("vulnerabilities", []) or []:
        cve_id = item.get("cveID") or item.get("cveId") or item.get("cve")
        if isinstance(cve_id, str):
            out.add(cve_id)
    return out


