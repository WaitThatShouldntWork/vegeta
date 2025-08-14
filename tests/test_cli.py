from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.decide import app


def test_decide_assets(tmp_path: Path) -> None:
    assets_path = tmp_path / "assets.json"
    assets_path.write_text(
        '{"assets": [{"id": 1}, {"id": 2}, {"id": 3}]}', encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "decide",
            "--domain",
            "cyber",
            "--cve",
            "CVE-2025-TEST",
            "--assets",
            str(assets_path),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "VEGETA Action Scores" in result.stdout
    assert '"assets_count": 3' in result.stdout
    assert '"choice": "ask"' in result.stdout


