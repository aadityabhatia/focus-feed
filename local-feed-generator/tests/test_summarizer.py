"""Tests for the Summarizer service."""

import os
from unittest import mock

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
import pytest

from local_feed_generator.config import Settings
from local_feed_generator.errors import SummarizationError
from local_feed_generator.services.summarizer import Summarizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Provide a Settings instance with test values."""
    return Settings(
        base_url="https://example.com",
        data_dir="./data",
        llm_endpoint="https://llm.example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_system_prompt="Summarize key takeaways from this document.",
    )


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
def summarizer(mock_config):
    """Create a Summarizer instance with a mocked OpenAI client."""
    with mock.patch.object(OpenAI, "__init__", return_value=None):
        s = Summarizer(config=mock_config)
        s._client = mock.MagicMock()
        yield s


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestSummarizerInit:
    """Tests for Summarizer initialization."""

    def test_initializes_with_config(self, mock_config):
        """Summarizer should accept a config parameter."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config)
            assert s._config is mock_config

    def test_uses_get_config_when_no_config(self):
        """Summarizer should use get_config() when no config is provided."""
        env = {
            "BASE_URL": "https://example.com",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_SYSTEM_PROMPT": "Test prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            # Reset the singleton
            import local_feed_generator.config as config_module

            config_module._config_instance = None

            with mock.patch.object(OpenAI, "__init__", return_value=None):
                s = Summarizer()
                assert s._config is not None
                assert s._config.llm_endpoint == "https://llm.example.com/v1"

            # Restore singleton for other tests
            config_module._config_instance = None

    def test_client_uses_endpoint_from_config(self, mock_config):
        """OpenAI client should be initialized with the configured endpoint."""
        with mock.patch.object(OpenAI, "__init__", return_value=None) as mock_init:
            Summarizer(config=mock_config)
            mock_init.assert_called_once_with(
                base_url="https://llm.example.com/v1",
                api_key="test-key",
            )

    def test_client_init_failure_raises_summarization_error(self, mock_config):
        """Summarizer should raise SummarizationError if client init fails."""
        with (
            mock.patch.object(OpenAI, "__init__", side_effect=RuntimeError("connection refused")),
            pytest.raises(SummarizationError, match="Failed to initialize"),
        ):
            Summarizer(config=mock_config)


# ---------------------------------------------------------------------------
# summarize() tests
# ---------------------------------------------------------------------------


class TestSummarize:
    """Tests for the summarize() method."""

    def test_returns_summary_string(self, summarizer, mock_response):
        """summarize() should return a tuple of (summary, prompt_tokens, completion_tokens)."""
        summarizer._client.chat.completions.create.return_value = mock_response
        result = summarizer.summarize("This is a transcript about ILE courses.")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)

    def test_uses_system_prompt_from_config(self, summarizer, mock_response):
        """summarize() should use the system prompt from configuration."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test transcript")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "Summarize key takeaways from this document."

    def test_uses_llm_model_from_config(self, summarizer, mock_response):
        """summarize() should pass the configured model name."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test transcript")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_passes_transcript_as_user_content(self, summarizer, mock_response):
        """summarize() should pass the transcript as user content."""
        summarizer._client.chat.completions.create.return_value = mock_response
        transcript = "This is a long transcript about military education."
        summarizer.summarize(transcript)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == transcript

    def test_strips_whitespace_from_response(self, summarizer):
        """summarize() should strip leading/trailing whitespace from response."""
        response = mock.MagicMock()
        response.choices[0].message.content = "  Summary with spaces  "
        response.usage.prompt_tokens = 20
        response.usage.completion_tokens = 10
        summarizer._client.chat.completions.create.return_value = response
        result = summarizer.summarize("Test transcript")
        assert result == ("Summary with spaces", 20, 10)

    def test_empty_transcript_raises_error(self, summarizer):
        """summarize() should raise SummarizationError for empty transcript."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.summarize("")

    def test_whitespace_only_transcript_raises_error(self, summarizer):
        """summarize() should raise SummarizationError for whitespace-only transcript."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.summarize("   \n\t  ")

    def test_none_content_raises_error(self, summarizer):
        """summarize() should raise SummarizationError when LLM returns None content."""
        response = mock.MagicMock()
        response.choices[0].message.content = None
        summarizer._client.chat.completions.create.return_value = response
        with pytest.raises(SummarizationError, match="empty response"):
            summarizer.summarize("Test transcript")


# ---------------------------------------------------------------------------
# generate_title() tests
# ---------------------------------------------------------------------------


class TestGenerateTitle:
    """Tests for the generate_title() method."""

    def test_returns_title_string(self, summarizer, mock_title_response):
        """generate_title() should return a tuple of (title, prompt_tokens, completion_tokens)."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        result = summarizer.generate_title("Some content to generate a title from.")
        assert result == ("Test Title", 30, 5)

    def test_uses_title_system_prompt(self, summarizer, mock_title_response):
        """generate_title() should use a title-focused system prompt."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title("Some content")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "title" in system_msg["content"].lower()

    def test_passes_content_as_user_content(self, summarizer, mock_title_response):
        """generate_title() should pass the content as user content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        content = "Content about military strategy and tactics."
        summarizer.generate_title(content)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == content

    def test_empty_content_raises_error(self, summarizer):
        """generate_title() should raise SummarizationError for empty content."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.generate_title("")

    def test_whitespace_only_content_raises_error(self, summarizer):
        """generate_title() should raise SummarizationError for whitespace-only content."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.generate_title("   \n\t  ")


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for exponential backoff retry logic."""

    def test_retries_on_api_connection_error(self, summarizer, mock_response):
        """summarize() should retry on APIConnectionError and eventually succeed."""
        summarizer._client.chat.completions.create.side_effect = [
            APIConnectionError(message="Connection failed", request=mock.MagicMock()),
            mock_response,
        ]
        result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
        assert summarizer._client.chat.completions.create.call_count == 2

    def test_retries_on_timeout_error(self, summarizer, mock_response):
        """summarize() should retry on APITimeoutError and eventually succeed."""
        summarizer._client.chat.completions.create.side_effect = [
            APITimeoutError(request=mock.MagicMock()),
            mock_response,
        ]
        result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
        assert summarizer._client.chat.completions.create.call_count == 2

    def test_retries_on_rate_limit_error(self, summarizer, mock_response):
        """summarize() should retry on RateLimitError and eventually succeed."""
        summarizer._client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limited",
                response=mock.MagicMock(status_code=429),
                body=None,
            ),
            mock_response,
        ]
        result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
        assert summarizer._client.chat.completions.create.call_count == 2

    def test_retries_on_internal_server_error(self, summarizer, mock_response):
        """summarize() should retry on InternalServerError and eventually succeed."""
        summarizer._client.chat.completions.create.side_effect = [
            InternalServerError(
                message="Server error",
                response=mock.MagicMock(status_code=500),
                body=None,
            ),
            mock_response,
        ]
        result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
        assert summarizer._client.chat.completions.create.call_count == 2

    def test_does_not_retry_on_authentication_error(self, summarizer):
        """summarize() should not retry on AuthenticationError."""
        summarizer._client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=mock.MagicMock(status_code=401),
            body=None,
        )
        with pytest.raises(SummarizationError, match="non-retryable"):
            summarizer.summarize("Test transcript")
        assert summarizer._client.chat.completions.create.call_count == 1

    def test_does_not_retry_on_bad_request_error(self, summarizer):
        """summarize() should not retry on BadRequestError."""
        summarizer._client.chat.completions.create.side_effect = BadRequestError(
            message="Bad request",
            response=mock.MagicMock(status_code=400),
            body=None,
        )
        with pytest.raises(SummarizationError, match="non-retryable"):
            summarizer.summarize("Test transcript")
        assert summarizer._client.chat.completions.create.call_count == 1

    def test_does_not_retry_on_permission_denied_error(self, summarizer):
        """summarize() should not retry on PermissionDeniedError."""
        summarizer._client.chat.completions.create.side_effect = PermissionDeniedError(
            message="Forbidden",
            response=mock.MagicMock(status_code=403),
            body=None,
        )
        with pytest.raises(SummarizationError, match="non-retryable"):
            summarizer.summarize("Test transcript")
        assert summarizer._client.chat.completions.create.call_count == 1

    def test_does_not_retry_on_not_found_error(self, summarizer):
        """summarize() should not retry on NotFoundError."""
        from openai import NotFoundError

        summarizer._client.chat.completions.create.side_effect = NotFoundError(
            message="Not found",
            response=mock.MagicMock(status_code=404),
            body=None,
        )
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

    def test_retry_logs_warning(self, summarizer, mock_response, caplog):
        """Retry attempts should log a warning."""
        summarizer._client.chat.completions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            mock_response,
        ]

        with mock.patch("time.sleep"):
            with caplog.at_level("WARNING"):
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

            assert any("failed after" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# Backoff calculation tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model parameter tests
# ---------------------------------------------------------------------------


class TestModelParameter:
    """Tests for LLM model parameter handling."""

    def test_passes_model_from_config(self, summarizer, mock_response):
        """When llm_model is set, it should be passed to the API."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test")
        call_kwargs = summarizer._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_passes_none_model_when_not_configured(self, mock_config):
        """When llm_model is None, None should be passed (endpoint default)."""
        mock_config.llm_model = None
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config)
            s._client = mock.MagicMock()
            mock_resp = mock.MagicMock()
            mock_resp.choices[0].message.content = "Summary"
            mock_resp.usage.prompt_tokens = 10
            mock_resp.usage.completion_tokens = 5
            s._client.chat.completions.create.return_value = mock_resp
            s.summarize("Test")
            call_kwargs = s._client.chat.completions.create.call_args
            assert call_kwargs.kwargs["model"] is None


# ---------------------------------------------------------------------------
# Token tracking tests
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Tests for LLM token usage tracking."""

    def test_summarize_returns_prompt_tokens(self, summarizer, mock_response):
        """summarize() should return prompt_tokens from the API response."""
        summarizer._client.chat.completions.create.return_value = mock_response
        result = summarizer.summarize("Test transcript")
        assert result[1] == 100

    def test_summarize_returns_completion_tokens(self, summarizer, mock_response):
        """summarize() should return completion_tokens from the API response."""
        summarizer._client.chat.completions.create.return_value = mock_response
        result = summarizer.summarize("Test transcript")
        assert result[2] == 50

    def test_generate_title_returns_prompt_tokens(self, summarizer, mock_title_response):
        """generate_title() should return prompt_tokens from the API response."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        result = summarizer.generate_title("Some content")
        assert result[1] == 30

    def test_generate_title_returns_completion_tokens(self, summarizer, mock_title_response):
        """generate_title() should return completion_tokens from the API response."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        result = summarizer.generate_title("Some content")
        assert result[2] == 5

    def test_summarize_returns_zero_tokens_when_usage_is_none(self, summarizer):
        """summarize() should return 0 tokens when usage is None."""
        response = mock.MagicMock()
        response.choices[0].message.content = "Summary text"
        response.usage = None
        summarizer._client.chat.completions.create.return_value = response
        result = summarizer.summarize("Test transcript")
        assert result == ("Summary text", 0, 0)

    def test_chat_completion_returns_llm_response(self, summarizer, mock_response):
        """_chat_completion() should return an LLmResponse NamedTuple."""
        from local_feed_generator.services.summarizer import LLmResponse

        summarizer._client.chat.completions.create.return_value = mock_response
        result = summarizer._chat_completion(
            system_prompt="Test prompt",
            user_content="Test content",
        )
        assert isinstance(result, LLmResponse)
        assert result.content == "Key takeaways: This is a test summary."
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    def test_token_counts_returned_on_retry(self, summarizer, mock_response):
        """Token counts should come from the successful (retry) response."""
        summarizer._client.chat.completions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            mock_response,
        ]
        with mock.patch("time.sleep"):
            result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
