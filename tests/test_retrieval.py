from __future__ import annotations

from retrieval.local_snapshot import LocalSnapshotRetriever


def test_local_snapshot_retriever_returns_k_docs() -> None:
    r = LocalSnapshotRetriever(corpus_name="test-corpus")
    docs = r.retrieve("cve query", k=2)
    assert len(docs) == 2
    assert all("Stub doc" in d.text for d in docs)
    assert all(d.metadata.get("source") == "test-corpus" for d in docs)


