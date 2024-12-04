from __future__ import annotations

from typing import TYPE_CHECKING

from zata.data.pipelines import nodes


if TYPE_CHECKING:
    from polars import DataFrame


def preprocess_data(data: DataFrame) -> DataFrame:
    """Clean influencer's posts."""
    return (
        data.pipe(nodes.remove_cols, ["id"])
        .pipe(nodes.remove_retweets, "content")
        .pipe(nodes.replace_newlines, "content")
        .pipe(nodes.replace_mentions, "content")
        .pipe(nodes.replace_urls, "content")
        .pipe(nodes.replace_hashtags, "content")
        .pipe(nodes.remove_short_tweets, "content")
    )
