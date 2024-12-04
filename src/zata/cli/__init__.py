from __future__ import annotations

import tomllib
from typing import Annotated
from pathlib import Path

import typer
import gradio as gr
import polars as pl

from zata.data.scrapers import Site
from zata.data.pipelines import preprocess_data
from zata.data.scrapers.x import XScraper
from zata.model.constants import DEFAULT_MODEL_NAME
from zata.envs.credentials import XCredentials


app = typer.Typer(
    short_help="-h",
    add_completion=False,
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
            help="Save to Parquet file",
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

    pl.DataFrame(posts).write_parquet(output, compression="zstd", compression_level=22)


@app.command()
def data_prep(
    raw: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Input raw data",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
            help="Save to Parquet file",
        ),
    ],
) -> None:
    """Preprocess raw data."""
    data = pl.read_parquet(raw)
    processed_data = preprocess_data(data)
    processed_data.write_parquet(output, compression="zstd", compression_level=22)


@app.command()
def train(
    data: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Save fine-tuned model to this directory",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Fine-tuning arguments TOML file",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
            help="Save fine-tuned model to this directory",
        ),
    ],
    model_name: Annotated[str, typer.Option()] = DEFAULT_MODEL_NAME,
) -> None:
    """Fine-tune LLM model."""
    from zata.model import trainer  # noqa: PLC0415
    from zata.model.args import FinetuningArguments  # noqa: PLC0415

    trainer.train(
        model_name=model_name,
        data_path=data,
        finetuning_args=FinetuningArguments.model_validate(
            tomllib.loads(config.read_text())
        ),
        save_to_dir=output,
    )


@app.command()
def serve(
    model: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Fine-tuned model directory",
        ),
    ],
) -> None:
    """Open UI."""
    from zata.model.inference import generate_response_wrapper  # noqa: PLC0415

    generate_response = generate_response_wrapper(finetuned_model=str(model))

    gr.ChatInterface(
        fn=generate_response,
        type="messages",
        title="Zata",
    ).queue().launch()
