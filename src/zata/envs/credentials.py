from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class XCredentials(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="X_",
        extra="ignore",
    )

    bearer_token: str
