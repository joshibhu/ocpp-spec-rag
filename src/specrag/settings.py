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

    # The allow-list of documents whose text may ground a normative claim.
    #
    # A COMMA-SEPARATED STRING AND NOT A list[str], which is a pydantic-settings
    # quirk worth knowing: a complex field type read from the environment is
    # parsed as JSON, so NORMATIVE_SOURCES=a.pdf,b.pdf would raise a validation
    # error and only ["a.pdf","b.pdf"] would work. A string plus one split is
    # the idiomatic way out. (Spring Boot binds a comma-separated env var to a
    # List<String> directly; Python does not.)
    #
    # These six are the OCA publications ingested on Day 11. Everything else in
    # the corpus - vendor manuals, community errata, field runbooks - is
    # unverified by default, including documents added after this line was
    # written. That default is the mitigation; the list is just its input.
    # A ceiling on completion tokens per answer, reasoning included. 0 means
    # no ceiling, which is the default and matches every run recorded so far.
    # See the comment in answer.py for why a cost defence cannot live in the
    # prompt and why this one is off until a caller turns it on.
    max_answer_tokens: int = 0

    normative_sources: str = (
        "ocpp-1.6-edition-2.pdf,"
        "ocpp-1.6-errata-sheet.pdf,"
        "ocpp-j-1.6-specification.pdf,"
        "ocpp-j-1.6-errata-sheet.pdf,"
        "ocpp-s-1.6-specification.pdf,"
        "ocpp-s-1.6-errata-sheet.pdf"
    )


settings = Settings()
