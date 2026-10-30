"""Remote-only adapter for the configured audio transcription API."""

from __future__ import annotations

import os

from pydantic import ValidationError

from focus_feed.config import get_config
from focus_feed.errors import ConfigurationError
from focus_feed.services.remote_transcriber import RemoteTranscriber


class Transcriber:
    """Transcribe audio exclusively through the configured remote API."""

    def __init__(self) -> None:
        """Initialize the remote transcriber from application settings."""
        try:
            config = get_config()
        except ValidationError as exc:
            missing = [
                field
                for field in ("TRANSCRIPTION_ENDPOINT", "TRANSCRIPTION_API_KEY")
                if not os.environ.get(field)
            ]
            if missing:
                raise ConfigurationError(
                    f"Remote transcription requires {', '.join(missing)}. "
                    "Set the required transcription environment variables."
                ) from exc
            raise ConfigurationError(f"Invalid application configuration: {exc}") from exc
        if not config.transcription_endpoint or not config.transcription_api_key:
            missing = []
            if not config.transcription_endpoint:
                missing.append("TRANSCRIPTION_ENDPOINT")
            if not config.transcription_api_key:
                missing.append("TRANSCRIPTION_API_KEY")
            raise ConfigurationError(
                f"Remote transcription requires {', '.join(missing)}. "
                "Set the required transcription environment variables."
            )
        self._remote_transcriber = RemoteTranscriber(config=config)

    def transcribe(self, filepath: str | os.PathLike[str]) -> str:
        """Transcribe a local audio file using the configured remote API."""
        return self._remote_transcriber.transcribe(filepath)

    def cleanup(self) -> None:
        """Keep the same lifecycle API as the underlying service."""
