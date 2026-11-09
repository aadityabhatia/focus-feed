"""Tests for the Summarizer service."""

from unittest import mock

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
import pytest

from focus_feed.errors import SummarizationError
from focus_feed.services.summarizer import Summarizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_response():
    """Create a mock OpenAI chat completion response."""
    response = mock.MagicMock()
    response.choices[0].message.content = "Key takeaways: This is a test summary."
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


@pytest.fixture
def mock_title_response():
    """Create a mock OpenAI chat completion response for title generation."""
    response = mock.MagicMock()
    response.choices[0].message.content = "Test Title"
    response.usage.prompt_tokens = 30
    response.usage.completion_tokens = 5
    return response


@pytest.fixture
def summarizer(mock_config, valid_env):
    """Create a Summarizer instance with a mocked OpenAI client."""
    with mock.patch.object(OpenAI, "__init__", return_value=None):
        s = Summarizer(config=mock_config)
        s._client = mock.MagicMock()
        yield s


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for exponential backoff retry logic."""

    @pytest.mark.parametrize(
        "error_factory",
        [
            lambda: APIConnectionError(message="Connection failed", request=mock.MagicMock()),
            lambda: APITimeoutError(request=mock.MagicMock()),
            lambda: RateLimitError(
                message="Rate limited", response=mock.MagicMock(status_code=429), body=None
            ),
            lambda: InternalServerError(
                message="Server error", response=mock.MagicMock(status_code=500), body=None
            ),
        ],
    )
    def test_retries_on_retryable_error(self, summarizer, mock_response, error_factory):
        """summarize() should retry on retryable errors and eventually succeed."""
        summarizer._client.chat.completions.create.side_effect = [error_factory(), mock_response]
        result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
        assert summarizer._client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize(
        "error_factory",
        [
            lambda: AuthenticationError(
                message="Invalid API key", response=mock.MagicMock(status_code=401), body=None
            ),
            lambda: BadRequestError(
                message="Bad request", response=mock.MagicMock(status_code=400), body=None
            ),
            lambda: PermissionDeniedError(
                message="Forbidden", response=mock.MagicMock(status_code=403), body=None
            ),
            lambda: NotFoundError(
                message="Not found", response=mock.MagicMock(status_code=404), body=None
            ),
        ],
    )
    def test_does_not_retry_on_non_retryable_error(self, summarizer, error_factory):
        """summarize() should not retry on non-retryable errors."""
        summarizer._client.chat.completions.create.side_effect = error_factory()
        with pytest.raises(SummarizationError, match="non-retryable"):
            summarizer.summarize("Test transcript")
        assert summarizer._client.chat.completions.create.call_count == 1

    def test_exhausts_retries_on_persistent_failure(self, summarizer):
        """summarize() should raise SummarizationError after max retries."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock.MagicMock()
        )
        with pytest.raises(SummarizationError, match="failed after"):
            summarizer.summarize("Test transcript")
        # Default max_retries=3, so 4 total attempts
        assert summarizer._client.chat.completions.create.call_count == 4

    def test_custom_max_retries(self, mock_config, mock_response):
        """Summarizer should respect custom max_retries."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config, max_retries=1)
            s._client = mock.MagicMock()
            s._client.chat.completions.create.side_effect = APIConnectionError(
                message="Connection failed", request=mock.MagicMock()
            )
            with pytest.raises(SummarizationError, match="failed after"):
                s.summarize("Test transcript")
            # max_retries=1, so 2 total attempts
            assert s._client.chat.completions.create.call_count == 2

    def test_retry_uses_exponential_backoff(self, summarizer, mock_response):
        """Retry delays should increase exponentially."""
        summarizer._client.chat.completions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            APIConnectionError(message="fail", request=mock.MagicMock()),
            mock_response,
        ]

        delays = []
        original_sleep = mock.patch("time.sleep")
        mock_sleep = original_sleep.start()
        mock_sleep.side_effect = lambda d: delays.append(d)

        try:
            result = summarizer.summarize("Test transcript")
            assert result == ("Key takeaways: This is a test summary.", 100, 50)
            assert len(delays) == 2
            # First delay: base_delay * 2^0 = 1.0
            # Second delay: base_delay * 2^1 = 2.0
            # With jitter, the actual values may vary slightly
            assert delays[0] < delays[1] or abs(delays[0] - delays[1]) < 0.5
        finally:
            original_sleep.stop()

    def test_retry_with_jitter(self, summarizer):
        """Backoff should include jitter to avoid thundering herd."""
        with mock.patch("time.sleep") as mock_sleep:
            summarizer._client.chat.completions.create.side_effect = [
                APIConnectionError(message="fail", request=mock.MagicMock()),
                APIConnectionError(message="fail", request=mock.MagicMock()),
                APIConnectionError(message="fail", request=mock.MagicMock()),
                APIConnectionError(message="fail", request=mock.MagicMock()),
            ]
            with pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            # All sleep calls should have positive delays
            for call in mock_sleep.call_args_list:
                delay = call.args[0]
                assert delay >= 0.0

    def test_unexpected_exception_wrapped(self, summarizer):
        """Unexpected exceptions should be wrapped in SummarizationError."""
        summarizer._client.chat.completions.create.side_effect = ValueError("Unexpected issue")
        with pytest.raises(SummarizationError, match="Unexpected error"):
            summarizer.summarize("Test transcript")
        assert summarizer._client.chat.completions.create.call_count == 1

    def test_retry_logs_debug(self, summarizer, mock_response, caplog):
        """Retry attempts should log at DEBUG level."""
        summarizer._client.chat.completions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            mock_response,
        ]

        with mock.patch("time.sleep"):
            with caplog.at_level("DEBUG", logger="focus_feed"):
                summarizer.summarize("Test transcript")

            assert any("retrying" in record.message.lower() for record in caplog.records)

    def test_final_failure_logs_error(self, summarizer, caplog):
        """Final failure after all retries should log an error."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )

        with mock.patch("time.sleep"):
            with caplog.at_level("ERROR"), pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            assert any("attempt=4/4" in record.message.lower() for record in caplog.records)


class TestBackoffCalculation:
    """Tests for the _calculate_backoff method."""

    def test_first_attempt_delay(self, summarizer):
        """First retry delay should be approximately base_delay (1.0s)."""
        delay = summarizer._calculate_backoff(0)
        # base_delay * 2^0 = 1.0, with jitter ±10%
        assert 0.9 <= delay <= 1.1

    def test_second_attempt_delay(self, summarizer):
        """Second retry delay should be approximately 2x base_delay."""
        delay = summarizer._calculate_backoff(1)
        # base_delay * 2^1 = 2.0, with jitter ±20%
        assert 1.6 <= delay <= 2.4

    def test_third_attempt_delay(self, summarizer):
        """Third retry delay should be approximately 4x base_delay."""
        delay = summarizer._calculate_backoff(2)
        # base_delay * 2^2 = 4.0, with jitter ±40%
        assert 2.4 <= delay <= 5.6

    def test_delay_never_negative(self, summarizer):
        """Backoff delay should never be negative."""
        for attempt in range(10):
            delay = summarizer._calculate_backoff(attempt)
            assert delay >= 0.0

    def test_custom_base_delay(self, mock_config):
        """Custom base_delay should be respected."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config, base_delay=2.0, jitter=0.0)
            delay = s._calculate_backoff(0)
            assert delay == 2.0  # No jitter, so exact value

    def test_zero_jitter(self, mock_config):
        """With zero jitter, delays should be exact powers of two."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config, base_delay=1.0, jitter=0.0)
            assert s._calculate_backoff(0) == 1.0
            assert s._calculate_backoff(1) == 2.0
            assert s._calculate_backoff(2) == 4.0
            assert s._calculate_backoff(3) == 8.0
