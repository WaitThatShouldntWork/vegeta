from __future__ import annotations

import typer

from vegeta.core.utility import expected_value


app = typer.Typer(help="VEGETA minimal CLI for foundational value demos.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Core demos and utilities."""


@app.command()
def example() -> None:
    """Show a tiny v, u, u' example (no EIG)."""

    actions = ["A", "B"]

    def u(action: str, state: str) -> float:
        return 1.0 if action == state else 0.0

    v_prior = expected_value(actions, {"A": 0.6, "B": 0.4}, u)
    v_post = expected_value(actions, {"A": 0.3, "B": 0.7}, u)

    typer.echo(f"Prior best value v = {v_prior}")
    typer.echo(f"Posterior best value v' = {v_post}")


if __name__ == "__main__":
    app()


