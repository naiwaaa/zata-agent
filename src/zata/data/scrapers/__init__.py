from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from zata.data.posts import InfluencerPost


class Site(StrEnum):
    X = "X"
    THREADS = "threads"


class Scraper(ABC):
    @abstractmethod
    def scrape_posts(self, username: str) -> list[InfluencerPost]:
        pass
