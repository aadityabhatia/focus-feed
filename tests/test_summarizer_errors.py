"""Tests for the Summarizer service."""

from unittest import mock

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
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


class TestSummarizerLogError:
    """Tests that log_error is called with the correct parameters."""

    def test_log_error_called_on_non_retryable_error(self, summarizer):
        """log_error should be called for non-retryable errors."""
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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


class TestFinalFailureErrorTypes:
    """Tests for log_error behavior on final failure for different error types.

    Verifies that:
    - APITimeoutError prints to stdout (print_stdout=True)
    - RateLimitError does NOT print to stdout (print_stdout=False), errors logged to error.log
    - InternalServerError does NOT print to stdout (print_stdout=False), errors logged to error.log
    """

    def test_timeout_error_final_failure_prints_stdout(self, summarizer, capsys, tmp_path):
        """APITimeoutError on final failure should print to stdout."""
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

    def test_timeout_error_final_failure_logged_to_error_log(self, summarizer, isolate_error_log):
        """APITimeoutError on final failure should be logged to error.log."""
        summarizer._client.chat.completions.create.side_effect = APITimeoutError(
            request=mock.MagicMock()
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        error_log_path = isolate_error_log / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "APITimeoutError" in content
        assert "ERROR" in content
        assert "summarization" in content

    def test_rate_limit_error_final_failure_no_stdout(self, summarizer, capsys, tmp_path):
        """RateLimitError on final failure should NOT print to stdout."""
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

    def test_rate_limit_error_final_failure_logged_to_error_log(
        self, summarizer, isolate_error_log
    ):
        """RateLimitError on final failure should be logged to error.log."""
        summarizer._client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limited",
            response=mock.MagicMock(status_code=429),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        error_log_path = isolate_error_log / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "RateLimitError" in content
        assert "Rate limited" in content
        assert "ERROR" in content
        assert "summarization" in content

    def test_internal_server_error_final_failure_no_stdout(self, summarizer, capsys, tmp_path):
        """InternalServerError on final failure should NOT print to stdout."""
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

    def test_internal_server_error_final_failure_logged_to_error_log(
        self, summarizer, isolate_error_log
    ):
        """InternalServerError on final failure should be logged to error.log."""
        summarizer._client.chat.completions.create.side_effect = InternalServerError(
            message="Internal server error",
            response=mock.MagicMock(status_code=500),
            body=None,
        )
        with mock.patch("time.sleep"), pytest.raises(SummarizationError):
            summarizer.summarize("Test transcript")

        error_log_path = isolate_error_log / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "InternalServerError" in content
        assert "Internal server error" in content
        assert "ERROR" in content
        assert "summarization" in content

    def test_rate_limit_error_print_stdout_is_false(self, summarizer):
        """RateLimitError on final failure should call log_error with print_stdout=False."""
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
        with mock.patch("focus_feed.services.summarizer.log_error") as mock_log_error:
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
