from __future__ import annotations

import json
from pathlib import Path


def test_emg_seed_files_exist() -> None:
	root = Path("data/emg")
	assert (root / "seed_entities.jsonl").exists()
	assert (root / "episodes.jsonl").exists()


def test_emg_seed_entities_parse() -> None:
	path = Path("data/emg/seed_entities.jsonl")
	lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
	items = [json.loads(ln) for ln in lines]
	assert any(it.get("name") == "Spirited Away" for it in items)
	assert all("slots" in it for it in items)


