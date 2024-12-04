from __future__ import annotations

from pydantic import BaseModel


class InfluencerPost(BaseModel):
    id: str
    text: str
