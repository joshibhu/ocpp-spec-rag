"""Typed configuration, read once at import."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql://ocpp_owner:dev_only_not_a_secret@localhost:5435/ocppqa"
    )

    openai_api_key: str = ""
    # Which model turns text into coordinates. 1536 dimensions, and the
    # `vector(1536)` column is sized to match — changing this means
    # re-embedding everything, so it is a schema decision, not a setting.
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chat_model: str = "gpt-5-nano"


settings = Settings()
