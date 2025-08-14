from __future__ import annotations

from decider.choose import choose_action


def test_search_includes_preview_when_enabled() -> None:
    state = choose_action(
        domain="cyber",
        cve="CVE-2025-TEST",
        assets_path=None,
        seed=42,
        force_action="search",
        do_retrieval=True,
        top_k=2,
    )
    assert state["choice"] == "search"
    assert "retrieval_preview" in state
    assert isinstance(state["retrieval_preview"], list)
    assert len(state["retrieval_preview"]) == 2
    assert all("text" in d and "metadata" in d for d in state["retrieval_preview"])


