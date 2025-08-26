from __future__ import annotations

from typing import Dict, Any, List


def map_nvd_cve_to_node(cve: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a minimal CVE node shape from NVD JSON."""
    cve_id = cve.get("id") or cve.get("cve", {}).get("id") or cve.get("cve", {}).get("CVE_data_meta", {}).get("ID")
    metrics = (
        cve.get("metrics")
        or cve.get("impact")
        or {}
    )
    cvss = None
    if "cvssMetricV31" in metrics:
        arr = metrics.get("cvssMetricV31") or []
        if arr:
            cvss = arr[0].get("cvssData", {}).get("baseScore")
    elif "baseMetricV3" in metrics:
        cvss = metrics.get("baseMetricV3", {}).get("cvssV3", {}).get("baseScore")
    return {
        "id": cve_id,
        "labels": ["Entity", "CVE"],
        "properties": {
            "cvss": cvss,
        },
    }


def map_cpe_to_product(cpe23: str) -> Dict[str, Any]:
    parts = cpe23.split(":")
    vendor = parts[3] if len(parts) > 3 else None
    product = parts[4] if len(parts) > 4 else None
    return {
        "id": f"product:{vendor}:{product}",
        "labels": ["Entity", "Product"],
        "properties": {"vendor": vendor, "product": product, "cpe23Uri": cpe23},
    }


def map_cve_affects_edge(cve_id: str, product_id: str) -> Dict[str, Any]:
    return {"type": "AFFECTS", "start": cve_id, "end": product_id, "properties": {}}


def map_vendor_node(vendor: str | None) -> Dict[str, Any] | None:
    if not vendor:
        return None
    return {
        "id": f"vendor:{vendor}",
        "labels": ["Entity", "Vendor"],
        "properties": {"name": vendor},
    }


def map_vendor_produces_edge(vendor_id: str, product_id: str) -> Dict[str, Any]:
    return {"type": "PRODUCES", "start": vendor_id, "end": product_id, "properties": {}}


