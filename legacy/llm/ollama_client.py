from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional


def generate(
	prompt: str,
	model: str = "gemma:2b",
	base_url: Optional[str] = None,
	temperature: float = 0.2,
) -> str:
	"""Call a local Ollama server to generate text.

	Relies on the standard /api/generate endpoint. Returns the response text or raises on HTTP error.
	"""
	base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
	url = f"{base.rstrip('/')}/api/generate"
	payload = {
		"model": model,
		"prompt": prompt,
		"stream": False,
		"options": {"temperature": float(temperature)},
	}
	req = urllib.request.Request(
		url,
		data=json.dumps(payload).encode("utf-8"),
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urllib.request.urlopen(req, timeout=60) as resp:
		data = json.loads(resp.read().decode("utf-8"))
	return data.get("response", "")


