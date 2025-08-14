from __future__ import annotations

import os
from config.settings import get_aura_settings


def test_env_loading_from_os_environ(monkeypatch) -> None:
    monkeypatch.setenv("AURA_URI", "neo4j+s://example.databases.neo4j.io")
    monkeypatch.setenv("AURA_USER", "neo4j")
    monkeypatch.setenv("AURA_PASSWORD", "secret")
    s = get_aura_settings()
    assert s.uri and s.user and s.password


