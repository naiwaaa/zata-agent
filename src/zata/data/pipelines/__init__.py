from __future__ import annotations

from typing import TYPE_CHECKING

from zata.data.pipelines import nodes


if TYPE_CHECKING:
    from polars import DataFrame


def preprocess_data(data: DataFrame) -> DataFrame:
    """Clean influencer's posts."""
    return (
        data.pipe(nodes.remove_cols, ["id"])
        .pipe(nodes.remove_retweets, "text")
        .pipe(nodes.remove_non_english_characters, "text")
        .pipe(nodes.replace_newlines, "text")
        .pipe(nodes.replace_mentions, "text")
        .pipe(nodes.replace_urls, "text")
        .pipe(nodes.replace_hashtags, "text")
        .pipe(nodes.remove_blank_spaces, "text")
        .pipe(nodes.replace_amp, "text")
        .pipe(nodes.remove_short_tweets, "text")
    )
