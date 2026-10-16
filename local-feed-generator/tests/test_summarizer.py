"""Tests for the Summarizer service."""

import os
from unittest import mock

from local_feed_generator.config import Settings
from local_feed_generator.errors import SummarizationError
from local_feed_generator.services.summarizer import Summarizer
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
        summary_prompt="Summarize key takeaways from this document.",
        title_prompt="Generate a title for the given content.",
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
            "SUMMARY_PROMPT": "Test summary prompt",
            "TITLE_PROMPT": "Test title prompt",
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

    def test_api_call_parameters(self, summarizer, mock_response):
        """summarize() should pass correct system prompt, model, and user content."""
        summarizer._client.chat.completions.create.return_value = mock_response
        transcript = "This is a long transcript about military education."
        summarizer.summarize(transcript)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "Summarize key takeaways from this document."
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

    @pytest.mark.parametrize("transcript", ["", "   \n\t  "])
    def test_empty_or_whitespace_transcript_raises_error(self, summarizer, transcript):
        """summarize() should raise SummarizationError for empty or whitespace-only transcript."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.summarize(transcript)

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

    def test_api_call_parameters(self, summarizer, mock_title_response):
        """generate_title() should use title-focused system prompt and pass content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        content = "Content about military strategy and tactics."
        summarizer.generate_title(content)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "title" in system_msg["content"].lower()
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert content in user_msg["content"]

    def test_truncates_content_to_4000_chars(self, summarizer, mock_title_response):
        """generate_title() should truncate content to ~4000 characters."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        long_content = "x" * 10000
        summarizer.generate_title(long_content)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        # The content portion should be truncated to 4000 chars
        assert "x" * 4000 in user_msg["content"]
        assert "x" * 4001 not in user_msg["content"]

    @pytest.mark.parametrize("filename,extracted_title,should_include", [
        ("lecture notes", "Lecture 1", True),
        ("notes", None, False),
        ("notes", "", False),
    ])
    def test_filename_and_extracted_title_in_content(
        self, summarizer, mock_title_response, filename, extracted_title, should_include
    ):
        """generate_title() should include filename and extracted_title when provided."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title("Some content", filename=filename, extracted_title=extracted_title)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        user_msg = call_kwargs.kwargs["messages"][1]
        assert f"Filename: {filename}" in user_msg["content"]
        if should_include:
            assert "Extracted Title: Lecture 1" in user_msg["content"]
        else:
            assert "Extracted Title" not in user_msg["content"]

    def test_includes_filename_and_extracted_title_before_content(
        self, summarizer, mock_title_response
    ):
        """generate_title() should include both filename and extracted title before content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title(
            "Some content here", filename="lecture notes", extracted_title="Lecture 1"
        )

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        user_content = user_msg["content"]
        # Filename and extracted title should appear before the content
        assert "Filename: lecture notes" in user_content
        assert "Extracted Title: Lecture 1" in user_content
        assert user_content.index("Filename:") < user_content.index("Some content here")
        assert user_content.index("Extracted Title:") < user_content.index("Some content here")

    def test_omits_content_label(self, summarizer, mock_title_response):
        """generate_title() should not include a 'Content:' label before the content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title("Some content here", filename="notes")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        assert "Content:\n" not in user_msg["content"]

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

    @pytest.mark.parametrize("error_factory", [
        lambda: APIConnectionError(message="Connection failed", request=mock.MagicMock()),
        lambda: APITimeoutError(request=mock.MagicMock()),
        lambda: RateLimitError(message="Rate limited", response=mock.MagicMock(status_code=429), body=None),
        lambda: InternalServerError(message="Server error", response=mock.MagicMock(status_code=500), body=None),
    ])
    def test_retries_on_retryable_error(self, summarizer, mock_response, error_factory):
        """summarize() should retry on retryable errors and eventually succeed."""
        summarizer._client.chat.completions.create.side_effect = [error_factory(), mock_response]
        result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
        assert summarizer._client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize("error_factory", [
        lambda: AuthenticationError(message="Invalid API key", response=mock.MagicMock(status_code=401), body=None),
        lambda: BadRequestError(message="Bad request", response=mock.MagicMock(status_code=400), body=None),
        lambda: PermissionDeniedError(message="Forbidden", response=mock.MagicMock(status_code=403), body=None),
        lambda: NotFoundError(message="Not found", response=mock.MagicMock(status_code=404), body=None),
    ])
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

    @pytest.mark.parametrize("method,response_fixture,expected_tokens", [
        ("summarize", "mock_response", (100, 50)),
        ("generate_title", "mock_title_response", (30, 5)),
    ])
    def test_returns_token_counts(self, request, summarizer, method, response_fixture, expected_tokens):
        """Methods should return (text, prompt_tokens, completion_tokens) tuples."""
        response = request.getfixturevalue(response_fixture)
        summarizer._client.chat.completions.create.return_value = response
        result = getattr(summarizer, method)("Test content")
        assert result[1] == expected_tokens[0]
        assert result[2] == expected_tokens[1]

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


# ---------------------------------------------------------------------------
# Stdout printing tests for connection errors
# ---------------------------------------------------------------------------


class TestConnectionErrorStdout:
    """Tests that connection errors are printed to stdout."""

    def test_connection_error_prints_to_stdout(self, summarizer, capsys):
        """APIConnectionError should print detailed info to stdout."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="Connection refused", request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        # stdout should contain connection error details
        assert "Error:" in captured.out
        assert "APIConnectionError" in captured.out
        assert "Connection refused" in captured.out
        assert "attempt=" in captured.out
        assert "https://llm.example.com/v1" in captured.out
        assert "test-model" in captured.out

    def test_timeout_error_prints_to_stdout(self, summarizer, capsys):
        """APITimeoutError should print detailed info to stdout."""
        summarizer._client.chat.completions.create.side_effect = APITimeoutError(
            request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "APITimeoutError" in captured.out

    def test_non_retryable_error_does_not_print_to_stdout(self, summarizer, capsys):
        """AuthenticationError should NOT print to stdout (not a connection error)."""
        summarizer._client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=mock.MagicMock(status_code=401),
            body=None,
        )
        with pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "Error:" not in captured.out

    def test_rate_limit_error_does_not_print_to_stdout(self, summarizer, capsys):
        """RateLimitError should NOT print to stdout (not a connection error)."""
        summarizer._client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limited",
            response=mock.MagicMock(status_code=429),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "Error:" not in captured.out

    def test_stdout_includes_retry_info(self, summarizer, capsys):
        """stdout output should include retry attempt information."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "attempt=" in captured.out

    def test_stdout_includes_endpoint(self, summarizer, capsys):
        """stdout output should include the LLM endpoint URL."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "https://llm.example.com/v1" in captured.out

    def test_stdout_includes_model(self, summarizer, capsys):
        """stdout output should include the model name."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "test-model" in captured.out

    def test_stdout_includes_stage_for_summarize(self, summarizer, capsys):
        """stdout output should include stage='summarization'."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        assert "summarization" in captured.out

    def test_stdout_includes_stage_for_title_generation(self, summarizer, capsys):
        """stdout output should include stage='title_generation'."""
        summarizer._client.chat.completions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.generate_title("Test content")

        captured = capsys.readouterr()
        assert "title_generation" in captured.out


# ---------------------------------------------------------------------------
# log_error integration tests for summarizer
# ---------------------------------------------------------------------------


class TestSummarizerLogError:
    """Tests that log_error is called with the correct parameters."""

    def test_log_error_called_on_non_retryable_error(self, summarizer):
        """log_error should be called for non-retryable errors."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=mock.MagicMock(status_code=401),
                body=None,
            )
            with pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            mock_log_error.assert_called_once()
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["stage"] == "summarization"
            assert call_kwargs.kwargs["endpoint"] == "https://llm.example.com/v1"
            assert call_kwargs.kwargs["model"] == "test-model"
            assert call_kwargs.kwargs["print_stdout"] is False

    def test_log_error_called_on_final_failure(self, summarizer):
        """log_error should be called when all retries are exhausted."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = APIConnectionError(
                message="Connection refused", request=mock.MagicMock()
            )
            with mock.patch("time.sleep"), pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            # Should be called for each retry attempt (3 retries + 1 final = 4)
            assert mock_log_error.call_count == 4
            # Check the last call has the final retry info
            last_call = mock_log_error.call_args_list[-1]
            assert last_call.kwargs["stage"] == "summarization"
            assert last_call.kwargs["retry_info"] == "attempt=4/4"
            assert last_call.kwargs["endpoint"] == "https://llm.example.com/v1"
            assert last_call.kwargs["model"] == "test-model"
            assert last_call.kwargs["print_stdout"] is True

    def test_log_error_called_for_retryable_error_during_retry(self, summarizer):
        """log_error should be called for connection errors during retries."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            # Set up the mock response for successful retry
            response = mock.MagicMock()
            response.choices[0].message.content = "Summary"
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 5
            summarizer._client.chat.completions.create.side_effect = [
                APIConnectionError(message="fail", request=mock.MagicMock()),
                response,
            ]
            with mock.patch("time.sleep"):
                summarizer.summarize("Test transcript")

            # log_error should be called once for the retry (connection error)
            assert mock_log_error.call_count == 1
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["stage"] == "summarization"
            assert call_kwargs.kwargs["retry_info"] == "attempt=1/4"
            assert call_kwargs.kwargs["print_stdout"] is True

    def test_log_error_not_called_for_rate_limit_during_retry(self, summarizer):
        """log_error should NOT be called for rate limit errors during retries."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            # Set up the mock response for successful retry
            response = mock.MagicMock()
            response.choices[0].message.content = "Summary"
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 5
            summarizer._client.chat.completions.create.side_effect = [
                RateLimitError(
                    message="Rate limited",
                    response=mock.MagicMock(status_code=429),
                    body=None,
                ),
                response,
            ]
            with mock.patch("time.sleep"):
                summarizer.summarize("Test transcript")

            # log_error should NOT be called for rate limit (not a connection error)
            assert mock_log_error.call_count == 0

    def test_log_error_called_for_unexpected_error(self, summarizer):
        """log_error should be called for unexpected errors."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = ValueError("Unexpected")
            with pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            mock_log_error.assert_called_once()
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["stage"] == "summarization"
            assert call_kwargs.kwargs["endpoint"] == "https://llm.example.com/v1"
            assert call_kwargs.kwargs["model"] == "test-model"
            assert call_kwargs.kwargs["print_stdout"] is False

    def test_log_error_stage_is_title_generation(self, summarizer):
        """log_error should use stage='title_generation' for generate_title."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=mock.MagicMock(status_code=401),
                body=None,
            )
            with pytest.raises(SummarizationError):
                summarizer.generate_title("Test content")

            mock_log_error.assert_called_once()
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["stage"] == "title_generation"

    def test_log_error_called_with_filepath(self, summarizer):
        """log_error should include filepath when passed to summarize()."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=mock.MagicMock(status_code=401),
                body=None,
            )
            with pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript", filepath="videos/lecture.mp4")

            mock_log_error.assert_called_once()
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["filepath"] == "videos/lecture.mp4"

    def test_log_error_called_with_filepath_for_generate_title(self, summarizer):
        """log_error should include filepath when passed to generate_title()."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=mock.MagicMock(status_code=401),
                body=None,
            )
            with pytest.raises(SummarizationError):
                summarizer.generate_title("Test content", filepath="docs/report.pdf")

            mock_log_error.assert_called_once()
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["filepath"] == "docs/report.pdf"


# ---------------------------------------------------------------------------
# Final failure tests for specific error types
# ---------------------------------------------------------------------------


class TestFinalFailureErrorTypes:
    """Tests for log_error behavior on final failure for different error types.

    Verifies that:
    - APITimeoutError prints to stdout (print_stdout=True)
    - RateLimitError does NOT print to stdout (print_stdout=False), errors logged to error.log
    - InternalServerError does NOT print to stdout (print_stdout=False), errors logged to error.log
    """

    def test_timeout_error_final_failure_prints_stdout(self, summarizer, capsys, tmp_path):
        """APITimeoutError on final failure should print to stdout."""
        # Configure error.log to write to a temp directory
        from local_feed_generator.errors import _configure_error_log
        _configure_error_log(data_dir=tmp_path, force=True)

        summarizer._client.chat.completions.create.side_effect = APITimeoutError(
            request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        # APITimeoutError is a connection error, so should print to stdout
        assert "Error:" in captured.out
        assert "APITimeoutError" in captured.out
        assert "attempt=" in captured.out

    def test_timeout_error_final_failure_logged_to_error_log(self, summarizer, tmp_path):
        """APITimeoutError on final failure should be logged to error.log."""
        from local_feed_generator.errors import _configure_error_log
        _configure_error_log(data_dir=tmp_path, force=True)

        summarizer._client.chat.completions.create.side_effect = APITimeoutError(
            request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        error_log_path = tmp_path / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "APITimeoutError" in content
        assert "ERROR" in content
        assert "summarization" in content

    def test_rate_limit_error_final_failure_no_stdout(self, summarizer, capsys, tmp_path):
        """RateLimitError on final failure should NOT print to stdout."""
        from local_feed_generator.errors import _configure_error_log
        _configure_error_log(data_dir=tmp_path, force=True)

        summarizer._client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limited",
            response=mock.MagicMock(status_code=429),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        # RateLimitError is NOT a connection error, so should NOT print to stdout
        assert "Error:" not in captured.out
        assert captured.out == ""

    def test_rate_limit_error_final_failure_logged_to_error_log(self, summarizer, tmp_path):
        """RateLimitError on final failure should be logged to error.log."""
        from local_feed_generator.errors import _configure_error_log
        _configure_error_log(data_dir=tmp_path, force=True)

        summarizer._client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limited",
            response=mock.MagicMock(status_code=429),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        error_log_path = tmp_path / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "RateLimitError" in content
        assert "Rate limited" in content
        assert "ERROR" in content
        assert "summarization" in content

    def test_internal_server_error_final_failure_no_stdout(self, summarizer, capsys, tmp_path):
        """InternalServerError on final failure should NOT print to stdout."""
        from local_feed_generator.errors import _configure_error_log
        _configure_error_log(data_dir=tmp_path, force=True)

        summarizer._client.chat.completions.create.side_effect = InternalServerError(
            message="Internal server error",
            response=mock.MagicMock(status_code=500),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        captured = capsys.readouterr()
        # InternalServerError is NOT a connection error, so should NOT print to stdout
        assert "Error:" not in captured.out
        assert captured.out == ""

    def test_internal_server_error_final_failure_logged_to_error_log(self, summarizer, tmp_path):
        """InternalServerError on final failure should be logged to error.log."""
        from local_feed_generator.errors import _configure_error_log
        _configure_error_log(data_dir=tmp_path, force=True)

        summarizer._client.chat.completions.create.side_effect = InternalServerError(
            message="Internal server error",
            response=mock.MagicMock(status_code=500),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        error_log_path = tmp_path / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "InternalServerError" in content
        assert "Internal server error" in content
        assert "ERROR" in content
        assert "summarization" in content

    def test_rate_limit_error_print_stdout_is_false(self, summarizer):
        """RateLimitError on final failure should call log_error with print_stdout=False."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limited",
                response=mock.MagicMock(status_code=429),
                body=None,
            )
            with mock.patch("time.sleep"), pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            # log_error is only called on final failure for non-connection
            # retryable errors (RateLimitError, InternalServerError).
            # During retries, log_error is only called for connection errors.
            assert mock_log_error.call_count == 1
            # Check the call has print_stdout=False
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["print_stdout"] is False
            assert call_kwargs.kwargs["stage"] == "summarization"

    def test_internal_server_error_print_stdout_is_false(self, summarizer):
        """InternalServerError on final failure should call log_error with print_stdout=False."""
        with mock.patch(
            "local_feed_generator.services.summarizer.log_error"
        ) as mock_log_error:
            summarizer._client.chat.completions.create.side_effect = InternalServerError(
                message="Internal server error",
                response=mock.MagicMock(status_code=500),
                body=None,
            )
            with mock.patch("time.sleep"), pytest.raises(SummarizationError):
                summarizer.summarize("Test transcript")

            # log_error is only called on final failure for non-connection
            # retryable errors (RateLimitError, InternalServerError).
            # During retries, log_error is only called for connection errors.
            assert mock_log_error.call_count == 1
            # Check the call has print_stdout=False
            call_kwargs = mock_log_error.call_args
            assert call_kwargs.kwargs["print_stdout"] is False
            assert call_kwargs.kwargs["stage"] == "summarization"
