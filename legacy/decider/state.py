from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class State:
    domain: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)

    def set_slot(self, name: str, value: Any) -> None:
        self.slots[name] = value


