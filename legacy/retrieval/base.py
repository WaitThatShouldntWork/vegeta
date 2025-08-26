from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol


@dataclass
class RetrievalDoc:
    text: str
    metadata: Dict[str, object]


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 3) -> List[RetrievalDoc]:
        ...


