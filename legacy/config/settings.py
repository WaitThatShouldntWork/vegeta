from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class AuraSettings:
    uri: Optional[str]
    user: Optional[str]
    password: Optional[str]


def get_aura_settings() -> AuraSettings:
    # Load from .env if present at repo root
    _load_dotenv_if_present(Path(".env"))
    # Support both AURA_* and NEO4J_* env names
    uri = os.getenv("AURA_URI") or os.getenv("NEO4J_URI")
    user = os.getenv("AURA_USER") or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("AURA_PASSWORD") or os.getenv("NEO4J_PASSWORD")
    return AuraSettings(uri=uri, user=user, password=password)


