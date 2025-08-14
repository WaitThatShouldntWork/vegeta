from __future__ import annotations

from retrieval.wikipedia import search_wikipedia


def test_wikipedia_search_smoke() -> None:
	docs = search_wikipedia("Spirited Away", k=1)
	assert isinstance(docs, list)
	# Network may be unavailable in CI; tolerate empty results
	assert docs == [] or (hasattr(docs[0], "text") and hasattr(docs[0], "metadata"))


