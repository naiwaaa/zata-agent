from __future__ import annotations

import typer

from zata.cli import scrape


app = typer.Typer(
    short_help="-h",
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command()(scrape.scrape)
