from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Node:
    identifier: str
    attributes: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """A minimal directed graph with string node identifiers."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._adjacency: Dict[str, List[str]] = {}

    def add_node(self, node_id: str, **attributes: Any) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = Node(node_id, dict(attributes))
            self._adjacency.setdefault(node_id, [])
        else:
            # Merge/overwrite attributes if node exists
            self._nodes[node_id].attributes.update(attributes)

    def add_edge(self, source_id: str, target_id: str, **_attributes: Any) -> None:
        if source_id not in self._nodes:
            self.add_node(source_id)
        if target_id not in self._nodes:
            self.add_node(target_id)
        self._adjacency[source_id].append(target_id)

    def neighbors(self, node_id: str) -> List[str]:
        return list(self._adjacency.get(node_id, []))

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes


