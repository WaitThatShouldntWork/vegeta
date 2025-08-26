from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterable

from db.writeback import upsert_entity_with_type, upsert_slot_value


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_seed_entities(path: Path = Path("data/emg/seed_entities.jsonl")) -> int:
    """Load EMG seed entities (Entities with Type and SlotValues). Returns count loaded."""
    count = 0
    for item in _iter_jsonl(path):
        name = str(item.get("name"))
        type_name = str(item.get("type") or "Entity")
        upsert_entity_with_type(entity_name=name, type_name=type_name)
        slots = item.get("slots") or {}
        if isinstance(slots, dict):
            for slot, value in slots.items():
                upsert_slot_value(entity_name=name, slot=str(slot), value=str(value), confidence=0.9, source="seed")
        count += 1
    return count


