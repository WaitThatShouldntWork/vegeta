from __future__ import annotations

from typing import List

from retrieval.base import RetrievalDoc, Retriever


class LocalSnapshotRetriever:
    """Very small stub retriever returning canned results.

    Later: hook to ingested snapshot; for now: predictable output.
    """

    def __init__(self, corpus_name: str = "cyber-snapshot") -> None:
        self.corpus_name = corpus_name

    def retrieve(self, query: str, k: int = 3) -> List[RetrievalDoc]:
        docs: List[RetrievalDoc] = []
        for i in range(k):
            docs.append(
                RetrievalDoc(
                    text=f"Stub doc {i+1} for query: {query}",
                    metadata={"source": self.corpus_name, "rank": i + 1},
                )
            )
        return docs


