from __future__ import annotations

from agent.question_generator import render_question_from_slot, llm_generate_question, render_from_domain_pack


def test_render_question_templates() -> None:
	q = render_question_from_slot("internet_exposed")
	assert "exposed" in q.lower()
	assert q.endswith("?")


def test_llm_generate_question_fallback_no_ollama() -> None:
	q = llm_generate_question("actively_exploited", context={}, use_ollama=False)
	assert isinstance(q, str)
	assert q.endswith("?")


def test_render_from_domain_pack_template_fill() -> None:
	q = render_from_domain_pack("internet_exposed", context={"asset_name": "edge-fw-01"}, domain="cyber")
	assert "edge-fw-01" in q
	assert q.endswith("?")


