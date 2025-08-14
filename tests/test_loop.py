from __future__ import annotations

from typer.testing import CliRunner

from cli.decide import app


def test_active_loop_runs_without_writeback() -> None:
	runner = CliRunner()
	result = runner.invoke(app, ["loop", "--cve", "CVE-2025-TEST", "--steps", "1"])
	assert result.exit_code == 0
	assert '"log"' in result.stdout
