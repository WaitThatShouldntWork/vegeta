from __future__ import annotations

from typing import Dict

from llm.ollama_client import generate as ollama_generate
import yaml
from pathlib import Path


def render_question_from_slot(slot: str, entity_hint: str | None = None) -> str:
	"""Heuristically humanize a slot to a natural question without domain lists."""
	name = str(slot).strip().replace("_", " ")
	# Split CamelCase to words
	parts = []
	buf = ""
	for ch in name:
		if buf and ch.isupper() and (not buf[-1].isupper()):
			parts.append(buf)
			buf = ch
		else:
			buf += ch
	if buf:
		parts.append(buf)
	label = " ".join(p.lower() for p in parts if p)
	label = label.strip() or "it"
	if entity_hint:
		return f"Can you share more about the {label} for {entity_hint}?"
	return f"Could you share a bit more about the {label}?"


def llm_generate_question(slot: str, context: Dict[str, str], use_ollama: bool = False) -> str:
	if not use_ollama:
		return render_question_from_slot(slot)
	# Generic, domain-agnostic compact prompt
	known_keys = ", ".join(context.get("known_keys", [])[:4])
	role = "You are a helpful assistant."
	style = "Ask a short, natural question a person would ask. Avoid internal field names like camelCase or underscores."
	prompt = (
		f"{role} {style} Focus on the information named '{slot}'. "
		f"If it helps, related info already known: {known_keys if known_keys else 'none'}. "
		"Ask exactly one question."
	)
	try:
		resp = ollama_generate(prompt=prompt, model=context.get("llm_model", "gemma:2b"), temperature=0.2)
		line = (resp or "").strip().splitlines()[0]
		return line if line else render_question_from_slot(slot)
	except Exception:
		return render_question_from_slot(slot)


def render_from_domain_pack(slot: str, context: Dict[str, str], domain: str = "cyber") -> str:
	"""Schema-driven rendering using domain pack questions.yaml.

	Tries to load a template for the slot and fill simple placeholders from context.
	Falls back to LLM or default renderer.
	"""
	try:
		path = Path(f"domain_packs/{domain}/questions.yaml")
		data = yaml.safe_load(path.read_text(encoding="utf-8"))
		for item in data:
			if str(item.get("slot")) == slot:
				tmpl = item.get("template") or item.get("text")
				if not tmpl:
					break
				# naive placeholder replacement
				q = tmpl
				for k, v in context.items():
					q = q.replace("{" + k + "}", str(v))
				return q
	except Exception:
		pass
	# fallback: LLM then hardcoded
	q = llm_generate_question(slot, context=context, use_ollama=context.get("use_ollama", False))
	return q or render_question_from_slot(slot)


