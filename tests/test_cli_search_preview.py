from __future__ import annotations

from typer.testing import CliRunner

from cli.decide import app


def test_cli_search_preview_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "decide",
            "--domain",
            "cyber",
            "--cve",
            "CVE-2025-TEST",
            "--force-action",
            "search",
            "--retrieve",
            "--top-k",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert '"retrieval_preview"' in result.stdout
    assert '"text"' in result.stdout

