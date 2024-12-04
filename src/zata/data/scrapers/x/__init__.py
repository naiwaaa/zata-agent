from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

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
        self.client = tweepy.Client(credentials.bearer_token, wait_on_rate_limit=True)

    def scrape_posts(self, username: str) -> list[InfluencerPost]:
        user: User = self.client.get_user(username=username).data
        paginator = tweepy.Paginator(self.client.get_users_tweets, user.id, max_results=5)

        return [
            InfluencerPost(id=str(tweet.id), text=tweet.text)
            for tweet in paginator.flatten(limit=10)
        ]
