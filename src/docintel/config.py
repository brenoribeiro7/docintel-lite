from functools import lru_cache

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
