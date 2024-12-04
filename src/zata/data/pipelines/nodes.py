from __future__ import annotations

import polars as pl


PATTERN_MENTION = r"@[a-zA-Z0-9_-]+"
PATTERN_HASHTAG = r"#[a-zA-Z0-9_-]+"
PATTERN_URL = r"(https?://)\S+|www\S+"


def remove_cols(data: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return data.drop(cols)


def remove_retweets(
    data: pl.DataFrame,
    col: str,
) -> pl.DataFrame:
    return data.filter(~pl.col(col).str.starts_with("RT @"))


def remove_short_tweets(
    data: pl.DataFrame,
    col: str,
    min_length: int = 10,
) -> pl.DataFrame:
    return data.filter(pl.col(col).str.len_chars() > min_length)


def replace_newlines(
    data: pl.DataFrame,
    col: str,
    new_col: str | None = None,
) -> pl.DataFrame:
    return data.with_columns(pl.col(col).str.replace_all("\n", " ").alias(new_col or col))


def replace_mentions(
    data: pl.DataFrame,
    col: str,
    new_col: str | None = None,
) -> pl.DataFrame:
    return data.with_columns(
        pl.col(col).str.replace_all(PATTERN_MENTION, "MENTION").alias(new_col or col)
    )


def replace_urls(
    data: pl.DataFrame,
    col: str,
    new_col: str | None = None,
) -> pl.DataFrame:
    return data.with_columns(
        pl.col(col).str.replace_all(PATTERN_URL, "URL").alias(new_col or col)
    )


def replace_hashtags(
    data: pl.DataFrame,
    col: str,
    new_col: str | None = None,
) -> pl.DataFrame:
    return data.with_columns(pl.col(col).str.replace_all("#", "").alias(new_col or col))
