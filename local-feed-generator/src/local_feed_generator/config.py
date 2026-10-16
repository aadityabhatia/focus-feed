"""Configuration management for local-feed-generator.

Uses pydantic-settings to load configuration from environment variables
and optional .env file. Provides a singleton accessor for global config.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Required fields must be provided via environment variables or .env file.
    Optional fields have defaults as documented in the class body.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required fields ---

    base_url: str
    data_dir: str
    llm_endpoint: str
    llm_api_key: str
    llm_model: str
    llm_system_prompt: str
    summary_prompt: str = (
        "Summarize key takeaways from the given content. Output just the summary concisely."
    )
    title_prompt: str = (
        "Generate a concise, descriptive title (5-10 words) for the given content. "
        "Do not include any formatting, quotes, or extra text."
    )

    # --- Optional fields with defaults ---

    feed_format: str = "rss"

    summary_reflection_prompt: str | None = None

    transcription_model: str = "large-v3-turbo"
    transcription_device: str = "cuda"
    transcription_compute_type: str = "float16"
    transcription_endpoint: str | None = None
    transcription_api_key: str | None = None
    transcription_remote_model: str | None = None

    # --- Validators ---

    @field_validator(
        "base_url",
        "data_dir",
        "llm_endpoint",
        "llm_api_key",
        "llm_model",
        "llm_system_prompt",
    )
    @classmethod
    def validate_required_non_empty(cls, v: str) -> str:
        """Ensure required fields are not empty strings."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v


# --- Singleton pattern ---

_config_instance: Settings | None = None


def get_config() -> Settings:
    """Return the singleton Settings instance.

    On first call, loads configuration from environment variables and
    .env file. Subsequent calls return the same cached instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Settings()  # type: ignore[call-arg]
    return _config_instance


def reload_config() -> Settings:
    """Force re-creation of the singleton Settings instance.

    Useful in tests or when environment variables change at runtime.
    """
    global _config_instance
    _config_instance = Settings()  # type: ignore[call-arg]
    return _config_instance
