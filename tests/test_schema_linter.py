from __future__ import annotations

from typer.testing import CliRunner
from schema.lint_schema import app


def test_whitelist_ok() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--seen", "AFFECTS,MITIGATES"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "whitelisted" in result.stdout.lower()


def test_unknown_rels_fail() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--seen", "AFFECTS,FOO"], catch_exceptions=False)
    assert result.exit_code != 0
    assert "unknown relationship types" in result.stdout.lower()


