from __future__ import annotations

from pydantic import BaseModel


class InfluencerPost(BaseModel):
    id: str
    url: str
    content: str
    is_quote_post: bool
