"""Remote transcription service for audio files.

Uses an OpenAI-compatible API endpoint to transcribe audio files.
The OpenAI client is initialized once and reused for all transcription
calls via a singleton pattern (same as the local Transcriber).

Configuration is read from environment variables via the config module:
- TRANSCRIPTION_ENDPOINT (str | None) - OpenAI-compatible API base URL
- TRANSCRIPTION_API_KEY (str | None) - API key for authentication
- TRANSCRIPTION_REMOTE_MODEL (str | None) - Model name passed to the API
- TRANSCRIPTION_MODEL (str) - Fallback model name when TRANSCRIPTION_REMOTE_MODEL is not set

If ``transcription_remote_model`` is configured, it is passed as the ``model``
parameter to the API. Otherwise, ``transcription_model`` is used as the
``model`` parameter. The OpenAI SDK requires a model parameter, so a value
is always passed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from focus_feed.config import Settings, get_config
from focus_feed.errors import TranscriptionError, log_error

_logger = logging.getLogger("focus_feed")

# ---------------------------------------------------------------------------
# Retry configuration constants
# ---------------------------------------------------------------------------

#: Maximum number of retry attempts for transient API failures.
DEFAULT_MAX_RETRIES = 3

#: Base delay in seconds for the first retry; doubles with each subsequent attempt.
DEFAULT_BASE_DELAY = 1.0

#: Maximum jitter factor applied on top of the exponential delay (0 to 1).
DEFAULT_JITTER = 0.1

# ---------------------------------------------------------------------------
# Retryable vs. non-retryable exception mapping
# ---------------------------------------------------------------------------

#: Exceptions that indicate a transient failure and should be retried.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

#: Exceptions that indicate a permanent failure and should not be retried.
_NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
)


class RemoteTranscriber:
    """Transcribes audio files to markdown text using a remote API.

    Uses an OpenAI-compatible client pointed at the configured transcription
    endpoint. The client is initialized once and reused across all
    :meth:`transcribe` calls.

    This class follows the same interface as :class:`Transcriber` and can be
    used as a drop-in replacement.

    Attributes:
        _config: The application settings instance.
        _client: The OpenAI-compatible client instance, or ``None`` if not
            yet initialized.
        _max_retries: Maximum retry attempts for transient API failures.
        _base_delay: Base delay (seconds) for exponential backoff.
        _jitter: Maximum jitter fraction (0.0-1.0) added to backoff delay.
    """

    def __init__(
        self,
        config: Settings | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        jitter: float = DEFAULT_JITTER,
    ) -> None:
        """Initialize the RemoteTranscriber with configuration and client.

        Args:
            config: Settings instance. When ``None``, ``get_config()`` is
                called to obtain the singleton configuration.
            max_retries: Maximum number of retry attempts for transient
                API failures. Defaults to 3.
            base_delay: Base delay in seconds for exponential backoff.
                Defaults to 1.0.
            jitter: Maximum jitter fraction (0.0-1.0) applied on top of
                the computed delay. Defaults to 0.1.

        Raises:
            TranscriptionError: If the transcription endpoint or API key is
                not configured, or if the OpenAI client cannot be initialized.
        """
        self._config = config if config is not None else get_config()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._jitter = jitter

        if not self._config.transcription_endpoint:
            raise TranscriptionError(
                "TRANSCRIPTION_ENDPOINT is not configured. "
                "Set the TRANSCRIPTION_ENDPOINT environment variable."
            )
        if not self._config.transcription_api_key:
            raise TranscriptionError(
                "TRANSCRIPTION_API_KEY is not configured. "
                "Set the TRANSCRIPTION_API_KEY environment variable."
            )

        try:
            self._client = OpenAI(
                base_url=self._config.transcription_endpoint,
                api_key=self._config.transcription_api_key,
            )
        except Exception as exc:
            raise TranscriptionError(f"Failed to initialize OpenAI client: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, filepath: str | os.PathLike[str]) -> str:
        """Transcribe an audio file to markdown text without timestamps.

        The file must exist. Sends the audio file to the configured
        transcription endpoint and returns the clean transcript text.

        Args:
            filepath: Path to the audio file to transcribe.

        Returns:
            A markdown string containing the full transcript without
            any timestamp annotations.

        Raises:
            TranscriptionError: If the file does not exist, the API
                request fails, or transcription fails for any reason.
        """
        path = Path(filepath)

        if not path.exists():
            raise TranscriptionError(
                f"File not found: {filepath}",
                filepath=str(filepath),
            )

        # Get audio file info for user-facing output.
        file_size = self._get_file_size(path)
        audio_length = self._get_audio_duration(path)

        if audio_length is not None and file_size is not None:
            _logger.info(
                "Transcribing %s: %.1fs, %s bytes",
                path.name,
                audio_length,
                f"{file_size:,}",
            )
        elif file_size is not None:
            _logger.info("Transcribing %s: %s bytes", path.name, f"{file_size:,}")
        else:
            _logger.info("Transcribing %s", path.name)

        start_time = time.time()

        try:
            transcript = self._create_transcription(path, str(filepath))

            elapsed = time.time() - start_time
            word_count = len(transcript.split())
            _logger.info(
                "Completed transcription for %s in %.2fs (%d words)",
                path.name,
                elapsed,
                word_count,
            )

            return transcript

        except TranscriptionError:
            raise
        except Exception as exc:
            error = TranscriptionError(
                f"Transcription failed for '{filepath}': {exc}",
                filepath=str(filepath),
            )
            log_error(error, filepath=str(filepath), logger=_logger, stage="transcription")
            raise error from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_transcription(self, path: Path, filepath: str) -> str:
        """Send the audio file to the transcription API and return the text.

        Implements retry logic with exponential backoff for transient
        failures (connection errors, timeouts, rate limits, 5xx errors).

        If ``transcription_remote_model`` is configured, it is passed as the
        ``model`` parameter to the API. Otherwise, no model is specified and
        the remote endpoint determines which model to use.

        Args:
            path: Path object for the audio file.
            filepath: String representation of the filepath for error reporting.

        Returns:
            Clean transcript text without timestamps.

        Raises:
            TranscriptionError: If the API request fails after all retries,
                or if a non-retryable error occurs.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                # Pass the Path object directly; the OpenAI SDK handles
                # reading the file contents.
                kwargs: dict[str, Any] = {
                    "file": path,
                    "response_format": "text",
                }
                # The OpenAI SDK requires a model parameter. Use
                # transcription_remote_model if set, otherwise fall back to
                # transcription_model (which is always set via config defaults).
                if self._config.transcription_remote_model:
                    kwargs["model"] = self._config.transcription_remote_model
                else:
                    kwargs["model"] = self._config.transcription_model
                response = self._client.audio.transcriptions.create(**kwargs)
                # response_format="text" should return a plain string, but some
                # OpenAI-compatible endpoints (e.g. vLLM) return a JSON-encoded
                # string or an object regardless of response_format. Handle all
                # cases to ensure we return plain text.
                if isinstance(response, str):
                    # Check if the string is actually JSON (vLLM returns JSON
                    # as a string even with response_format="text")
                    if response.startswith("{"):
                        try:
                            parsed = json.loads(response)
                            if isinstance(parsed, dict) and "text" in parsed:
                                return str(parsed["text"]).strip()
                        except json.JSONDecodeError:
                            pass
                    return response.strip()
                else:
                    # Response is an object with a .text attribute
                    return str(response.text).strip()

            except _NON_RETRYABLE_ERRORS as exc:
                # Permanent failures — do not retry.
                error = TranscriptionError(
                    f"Transcription API request failed (non-retryable): {exc}",
                    filepath=filepath,
                )
                log_error(exc, filepath=filepath, logger=_logger, stage="transcription")
                raise error from exc

            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._calculate_backoff(attempt)
                    _logger.debug(
                        "Transcription API request failed (attempt %d/%d), retrying in %.2fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                else:
                    retry_info = f"attempt={attempt + 1}/{self._max_retries + 1}"
                    log_error(
                        exc,
                        filepath=filepath,
                        logger=_logger,
                        stage="transcription",
                        retry_info=retry_info,
                    )

            except Exception as exc:
                # Unexpected errors — do not retry.
                error = TranscriptionError(
                    f"Unexpected error during transcription API request: {exc}",
                    filepath=filepath,
                )
                log_error(exc, filepath=filepath, logger=_logger, stage="transcription")
                raise error from exc

        # All retries exhausted.
        raise TranscriptionError(
            f"Transcription API request failed after {self._max_retries + 1} attempts: {last_exc}",
            filepath=filepath,
        ) from last_exc

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter.

        Uses the formula: ``base_delay * (2 ** attempt) + jitter`` where
        jitter is a random value in ``[-jitter_factor, +jitter_fraction]``.

        Args:
            attempt: Zero-based attempt index.

        Returns:
            Delay in seconds before the next retry.
        """
        delay = self._base_delay * (2**attempt)
        jitter_amount = delay * self._jitter
        jitter: float = random.uniform(-jitter_amount, jitter_amount)
        return float(max(0.0, delay + jitter))

    @staticmethod
    def _get_file_size(path: Path) -> int | None:
        """Return the file size in bytes, or ``None`` if it cannot be determined."""
        try:
            return path.stat().st_size
        except OSError:
            return None

    @staticmethod
    def _get_audio_duration(path: Path) -> float | None:
        """Return the audio duration in seconds, or ``None`` if it cannot be determined.

        Uses the ``av`` library (already a dependency via faster-whisper) to
        read the container's audio stream metadata without decoding the full
        audio into memory.  All exceptions are caught so that transcription
        can proceed even when metadata cannot be retrieved.
        """
        try:
            import av

            with av.open(str(path), mode="r") as container:
                if container.streams.audio:
                    # duration_seconds is a runtime property on av streams;
                    # use getattr because av stubs may not include it.
                    duration = getattr(container.streams.audio[0], "duration_seconds", None)
                    if duration is not None:
                        return float(duration)
        except Exception:
            pass
        return None
