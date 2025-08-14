from __future__ import annotations

from domain_packs.cyber.mappings import map_cpe_to_product, map_vendor_node, map_vendor_produces_edge


def test_vendor_and_product_mapping_from_cpe() -> None:
    cpe = "cpe:2.3:a:acme:widget:*:*:*:*:*:*:*:*"
    prod = map_cpe_to_product(cpe)
    assert prod["labels"] == ["Entity", "Product"]
    assert prod["properties"]["vendor"] == "acme"
    assert prod["properties"]["product"] == "widget"

    v = map_vendor_node(prod["properties"]["vendor"])
    assert v is not None
    assert v["labels"] == ["Entity", "Vendor"]

    edge = map_vendor_produces_edge(v["id"], prod["id"])  # type: ignore[index]
    assert edge["type"] == "PRODUCES"


