from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Set, List

import typer
import yaml  # type: ignore[import-not-found]


app = typer.Typer(add_completion=False, no_args_is_help=False)


def load_relationship_whitelist(metamodel_path: Path) -> Set[str]:
    with metamodel_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rels = data.get("relationship_types", [])
    return {str(r) for r in rels}


def parse_seen(seen: Iterable[str]) -> Set[str]:
    parsed: Set[str] = set()
    for item in seen:
        for token in item.split(","):
            token = token.strip()
            if token:
                parsed.add(token)
    return parsed


@app.command()
def main(
    seen: List[str] = typer.Option(
        [], "--seen", help="Comma-separated relationship types observed in data/code"
    ),
    metamodel: Path = typer.Option(
        Path("schema/metamodel.yaml"), "--metamodel", help="Path to metamodel.yaml"
    ),
) -> None:
    """Fail if any seen relationship types are outside the whitelist.

    If --seen is omitted, prints the whitelist and exits 0.
    """
    whitelist = load_relationship_whitelist(metamodel)
    seen_set = parse_seen(seen)

    if not seen_set:
        typer.echo("Whitelist:")
        for r in sorted(whitelist):
            typer.echo(f"- {r}")
        raise typer.Exit(code=0)

    unknown = sorted(s for s in seen_set if s not in whitelist)
    if unknown:
        typer.echo("Unknown relationship types detected:")
        for r in unknown:
            typer.echo(f"- {r}")
        raise typer.Exit(code=1)

    typer.echo("All relationship types are whitelisted.")
    raise typer.Exit(code=0)


if __name__ == "__main__":  # pragma: no cover
    # Allow running as a module or script
    try:
        app()
    except SystemExit as exc:  # ensure proper exit code when embedded
        sys.exit(exc.code)


