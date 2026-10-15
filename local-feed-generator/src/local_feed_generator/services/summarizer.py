"""Summarization service for local-feed-generator.

Provides LLM-based summarization of audio transcripts and title generation
for content that lacks source metadata. Uses an OpenAI-compatible client
pointed at a local LLM endpoint (as configured via environment variables).

Configuration (from config.py / design doc section 3.2):
    LLM_ENDPOINT        - OpenAI-compatible API base URL
    LLM_API_KEY         - API key for authentication
    LLM_MODEL           - Model identifier (optional; falls back to endpoint default)
    LLM_SYSTEM_PROMPT   - System prompt for summarization (default from design doc)
"""

import logging
import random
import time
from typing import NamedTuple

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
    UnprocessableEntityError,
)

from local_feed_generator.config import Settings, get_config
from local_feed_generator.errors import SummarizationError

_logger = logging.getLogger(__name__)


class LLmResponse(NamedTuple):
    """Container for an LLM chat completion response.

    Attributes:
        content: The generated text content from the LLM.
        prompt_tokens: Number of tokens in the prompt (input).
        completion_tokens: Number of tokens in the completion (output).
    """

    content: str
    prompt_tokens: int
    completion_tokens: int

# ---------------------------------------------------------------------------
# Retry configuration constants
# ---------------------------------------------------------------------------

#: Maximum number of retry attempts for transient API failures.
DEFAULT_MAX_RETRIES = 3

#: Base delay in seconds for the first retry; doubles with each subsequent attempt.
DEFAULT_BASE_DELAY = 1.0

#: Maximum jitter factor applied on top of the exponential delay (0 to 1).
DEFAULT_JITTER = 0.1

#: Title-generation system prompt — focused and concise.
TITLE_SYSTEM_PROMPT = (
    "Generate a concise, descriptive title (5-10 words) for the given content. "
    "The title should capture the main topic or theme. "
    "Do not include any formatting, quotes, or extra text — just the title."
)

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
    UnprocessableEntityError,
)


class Summarizer:
    """LLM-based summarization service.

    Initializes an OpenAI-compatible client using configuration from
    environment variables (via ``config.get_config()``) and provides methods
    for summarizing transcripts and generating titles.

    Attributes:
        _config: The application settings singleton.
        _client: The OpenAI-compatible client instance.
        _max_retries: Maximum retry attempts for transient errors.
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
        """Initialize the Summarizer with configuration and client.

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
            SummarizationError: If the OpenAI client cannot be initialized.
        """
        self._config = config if config is not None else get_config()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._jitter = jitter

        try:
            self._client = OpenAI(
                base_url=self._config.llm_endpoint,
                api_key=self._config.llm_api_key,
            )
        except Exception as exc:
            raise SummarizationError(f"Failed to initialize OpenAI client: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, transcript: str) -> tuple[str, int, int]:
        """Generate key takeaways from a transcript.

        Sends the transcript to the configured LLM with the system prompt
        from configuration (``LLM_SYSTEM_PROMPT``) and returns the generated
        summary along with token usage data.

        Args:
            transcript: The transcript text to summarize.

        Returns:
            A tuple of ``(summary, prompt_tokens, completion_tokens)``.

        Raises:
            SummarizationError: If the LLM request fails after all retries,
                or if the transcript is empty.
        """
        if not transcript or not transcript.strip():
            raise SummarizationError("Transcript text must not be empty")

        result = self._chat_completion(
            system_prompt=self._config.llm_system_prompt,
            user_content=transcript,
        )
        return result.content, result.prompt_tokens, result.completion_tokens

    def generate_title(self, content: str) -> tuple[str, int, int]:
        """Generate a concise title from content.

        Uses a dedicated title-generation system prompt to produce a short,
        descriptive title. This is used as a fallback when source metadata
        (e.g., PDF title, HTML ``<title>``) is unavailable.

        Args:
            content: The content text to generate a title from.

        Returns:
            A tuple of ``(title, prompt_tokens, completion_tokens)``.

        Raises:
            SummarizationError: If the LLM request fails after all retries,
                or if the content is empty.
        """
        if not content or not content.strip():
            raise SummarizationError("Content text must not be empty")

        result = self._chat_completion(
            system_prompt=TITLE_SYSTEM_PROMPT,
            user_content=content,
        )
        return result.content, result.prompt_tokens, result.completion_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_completion(
        self,
        system_prompt: str,
        user_content: str,
    ) -> LLmResponse:
        """Execute a chat completion with retry logic.

        Sends a two-message conversation (system + user) to the LLM and
        returns the content of the first choice along with token usage
        data. Implements exponential backoff with jitter for transient
        failures.

        Args:
            system_prompt: The system message content.
            user_content: The user message content (transcript or other text).

        Returns:
            An :class:`LLmResponse` with content, prompt_tokens, and
            completion_tokens.

        Raises:
            SummarizationError: If all retries are exhausted or a non-
                retryable error occurs.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.llm_model,  # type: ignore[arg-type]
                    messages=messages,  # type: ignore[arg-type]
                )
                content = response.choices[0].message.content
                if content is None:
                    raise SummarizationError("LLM returned an empty response")

                # Extract token usage; default to 0 if not provided.
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage is not None else 0
                completion_tokens = usage.completion_tokens if usage is not None else 0

                return LLmResponse(
                    content=str(content).strip(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            except _NON_RETRYABLE_ERRORS as exc:
                # Permanent failures — do not retry.
                raise SummarizationError(f"LLM request failed (non-retryable): {exc}") from exc

            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._calculate_backoff(attempt)
                    _logger.warning(
                        "LLM request failed (attempt %d/%d), retrying in %.2fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                else:
                    _logger.error(
                        "LLM request failed after %d attempts: %s",
                        self._max_retries + 1,
                        exc,
                    )

            except Exception as exc:
                # Unexpected errors — do not retry.
                raise SummarizationError(f"Unexpected error during LLM request: {exc}") from exc

        # All retries exhausted.
        raise SummarizationError(
            f"LLM request failed after {self._max_retries + 1} attempts: {last_exc}"
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
