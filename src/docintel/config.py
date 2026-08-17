from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+psycopg://docintel:docintel@localhost:5432/docintel",
        validation_alias="DATABASE_URL",
    )
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_embedding_model: Literal["text-embedding-3-small"] = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
