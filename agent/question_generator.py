from __future__ import annotations

from typing import Dict

from llm.ollama_client import generate as ollama_generate
import yaml
from pathlib import Path


def render_question_from_slot(slot: str, entity_hint: str | None = None) -> str:
	if slot == "internet_exposed":
		return "Is the affected asset exposed to the public Internet?"
	if slot == "actively_exploited":
		return "Is this vulnerability currently being exploited in the wild in our environment?"
	if slot == "epss_high":
		return "Is the probability of exploitation (EPSS) high for this CVE in our context?"
	# fallback
	return f"Can you provide more detail about {slot}?"


def llm_generate_question(slot: str, context: Dict[str, str], use_ollama: bool = False) -> str:
	if not use_ollama:
		return render_question_from_slot(slot)
	# Very small, structured prompt to encourage a short, grounded question
	prompt = (
		"You are a concise security analyst. Generate a single, short yes/no question that helps resolve the slot '"
		+ slot
		+ "' for the current case. Do not add explanations."
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


