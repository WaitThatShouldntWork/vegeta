from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, DefaultDict
from collections import defaultdict

import urllib.request


def fetch_nvd_recent(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch recent NVD CVEs (JSON 2.0) limited to `limit` records.

    Note: unauthenticated and simplistic; for production, use API key and pagination.
    """
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=%d" % limit
    with urllib.request.urlopen(url, timeout=30) as resp:  # nosec - controlled URL
        data = json.load(resp)
    return data.get("vulnerabilities", [])


def extract_cves(cves: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in cves:
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        metrics = cve.get("metrics", {})
        out.append({"id": cve_id, "metrics": metrics})
    return out


def _extract_cpe_uris_from_config(config: Any, uris: List[str]) -> None:
    """Collect cpe23Uri strings from configurations that may be dict or list.

    Handles structures like:
    - { nodes: [ { cpeMatch: [ { criteria: 'cpe:2.3:...' } ] } ] }
    - [ { nodes: [...] }, ... ]
    """
    if isinstance(config, list):
        for elem in config:
            _extract_cpe_uris_from_config(elem, uris)
        return

    if not isinstance(config, dict):
        return

    for node in config.get("nodes", []) or []:
        for match in node.get("cpeMatch", []) or []:
            uri = match.get("criteria") or match.get("cpe23Uri")
            if uri:
                uris.append(uri)
        # recurse if children exist
        children = node.get("children")
        if children:
            _extract_cpe_uris_from_config({"nodes": children}, uris)


def extract_cpe_map(cves: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Return mapping CVE_ID -> list of cpe23Uri strings.

    Attempts to read 'configurations' either at item root or under item['cve'].
    """
    mapping: DefaultDict[str, List[str]] = defaultdict(list)
    for item in cves:
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        if not cve_id:
            continue
        uris: List[str] = []
        configs = item.get("configurations") or cve.get("configurations") or {}
        _extract_cpe_uris_from_config(configs, uris)
        if uris:
            mapping[cve_id].extend(uris)
    return dict(mapping)


