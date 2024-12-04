from __future__ import annotations

import warnings
import contextlib
from typing import TYPE_CHECKING
from http.client import RemoteDisconnected

from zata.data.posts import InfluencerPost
from zata.data.scrapers import Scraper


if TYPE_CHECKING:
    from tweepy.user import User

    from zata.envs.credentials import XCredentials

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import tweepy


class XScraper(Scraper):
    def __init__(self, credentials: XCredentials) -> None:
        self.client = tweepy.Client(credentials.x_bearer_token)

    def scrape_posts(self, username: str) -> list[InfluencerPost]:
        user: User = self.client.get_user(username=username).data
        paginator = tweepy.Paginator(
            self.client.get_users_tweets, user.id, max_results=100
        )

        scraped_posts: list[InfluencerPost] = []
        with contextlib.suppress(
            tweepy.TweepyException,
            ConnectionError,
            RemoteDisconnected,
        ):
            scraped_posts.extend(
                InfluencerPost(id=str(tweet.id), text=tweet.text)
                for tweet in paginator.flatten(limit=1000)
            )

        return scraped_posts
