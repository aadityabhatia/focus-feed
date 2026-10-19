"""Configuration management for local-feed-generator.

Uses pydantic-settings to load configuration from environment variables
and optional .env file. Provides a singleton accessor for global config.
"""

import logging
import os
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_logger = logging.getLogger("local_feed_generator")

# Required configuration fields that must be non-empty.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "BASE_URL",
    "LLM_ENDPOINT",
    "LLM_API_KEY",
    "LLM_MODEL",
    "PROMPT_ARTICLE",
    "PROMPT_PODCAST",
    "PROMPT_TITLE",
)


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
    llm_endpoint: str
    llm_api_key: str
    llm_model: str
    prompt_article: str
    prompt_article_reflection: str | None = None
    prompt_podcast: str
    prompt_podcast_reflection: str | None = None
    prompt_title: str

    # --- Optional fields with defaults ---

    transcription_model: str = "large-v3-turbo"
    transcription_device: str = "cuda"
    transcription_compute_type: str = "float16"
    transcription_endpoint: str | None = None
    transcription_api_key: str | None = None
    transcription_remote_model: str | None = None

    # --- Validators ---

    @field_validator(
        "base_url",
        "llm_endpoint",
        "llm_api_key",
        "llm_model",
        "prompt_article",
        "prompt_podcast",
        "prompt_title",
    )
    @classmethod
    def validate_required_non_empty(cls, v: str) -> str:
        """Ensure required fields are not empty strings."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fallback_old_env_vars(cls, data: Any) -> Any:
        """Fall back to legacy environment variables if new ones are unset.

        Maps old env var names to new field names so that existing
        deployments continue to work after the rename.  New env vars
        always take priority.
        """
        if isinstance(data, dict):
            _fallback_map = {
                "prompt_article": "SUMMARY_PROMPT",
                "prompt_article_reflection": "SUMMARY_REFLECTION_PROMPT",
                "prompt_podcast": "SUMMARY_PROMPT",
                "prompt_podcast_reflection": "SUMMARY_REFLECTION_PROMPT",
                "prompt_title": "TITLE_PROMPT",
            }
            for field_name, old_env_var in _fallback_map.items():
                if not data.get(field_name):
                    old_value = os.environ.get(old_env_var)
                    if old_value:
                        data[field_name] = old_value
        return data


# --- Singleton pattern ---

_config_instance: Settings | None = None


def get_config() -> Settings:
    """Return the singleton Settings instance.

    On first call, loads configuration from environment variables and
    .env file. Subsequent calls return the same cached instance.
    """
    global _config_instance
    if _config_instance is None:
        _check_required_env_vars()
        _config_instance = Settings()  # type: ignore[call-arg]
        # Warn about important optional fields that are not set.
        _optional_fields: tuple[str, ...] = (
            "prompt_article_reflection",
            "prompt_podcast_reflection",
        )
        for field_name in _optional_fields:
            if getattr(_config_instance, field_name) is None:
                _logger.warning("Optional config field %s is not set", field_name)
    return _config_instance


def reload_config() -> Settings:
    """Force re-creation of the singleton Settings instance.

    Useful in tests or when environment variables change at runtime.
    """
    global _config_instance
    _check_required_env_vars()
    _config_instance = Settings()  # type: ignore[call-arg]
    return _config_instance


def _check_required_env_vars() -> None:
    """Log a CRITICAL message if any required configuration is missing.

    Checks the environment variables for all required fields before the
    pydantic Settings model is instantiated.  This ensures a clear,
    actionable CRITICAL-level log message is emitted (to stderr) before
    the :class:`ValidationError` is raised, so the user sees exactly which
    fields are missing.
    """
    missing = [field_name for field_name in _REQUIRED_FIELDS if not os.environ.get(field_name)]
    if missing:
        _logger.critical(
            "Missing required configuration environment variables: %s. "
            "Please set these before running local-feed-generator.",
            ", ".join(missing),
        )
