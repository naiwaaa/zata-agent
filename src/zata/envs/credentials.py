from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class XCredentials(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    x_bearer_token: str
