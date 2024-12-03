from __future__ import annotations

from typing import Annotated
from pathlib import Path

import typer
import polars as pl

from zata.data.scrapers import Site
from zata.data.scrapers.x import XScraper
from zata.envs.credentials import XCredentials


app = typer.Typer(
    short_help="-h",
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def scrape(
    site: Annotated[Site, typer.Option()],
    username: Annotated[str, typer.Option()],
    output: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
            help="Save to CSV file",
        ),
    ],
) -> None:
    """Scrape influencer's data."""
    match site:
        case Site.X:
            credentials = XCredentials()
            scraper = XScraper(credentials)
        case Site.THREADS:
            raise NotImplementedError

    posts = scraper.scrape_posts(username)

    pl.DataFrame(posts).write_csv(output)


@app.command()
def train() -> None:
    """Fine-tune LLM model."""
