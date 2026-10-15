"""Tests for the RemoteTranscriber service.

Tests cover:
- Initialization with config (endpoint, api_key, model)
- Configuration validation (missing endpoint, missing api_key)
- transcribe() method: file existence check, transcript formatting (no timestamps),
  error handling, progress output
- Retry logic for transient failures (connection, timeout, rate limit, 5xx)
- Non-retryable errors (auth, bad request, permission denied, not found)
- _calculate_backoff() method
- _get_file_size() and _get_audio_duration() helper methods
"""

import json
import os
from pathlib import Path
from unittest import mock

from local_feed_generator.config import Settings
from local_feed_generator.errors import ProcessingError, TranscriptionError
from local_feed_generator.services.remote_transcriber import RemoteTranscriber
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
    """Provide a Settings instance with test values for remote transcription."""
    return Settings(
        base_url="https://example.com/feeds",
        data_dir="./data",
        llm_endpoint="https://llm.example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_system_prompt="Test system prompt",
        transcription_endpoint="https://transcribe.example.com/v1",
        transcription_api_key="transcribe-key",
        transcription_model="whisper-1",
    )


@pytest.fixture
def remote_transcriber(mock_config):
    """Create a RemoteTranscriber instance with a mocked OpenAI client."""
    with mock.patch.object(OpenAI, "__init__", return_value=None):
        t = RemoteTranscriber(config=mock_config)
        t._client = mock.MagicMock()
        yield t


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestRemoteTranscriberInit:
    """Tests for RemoteTranscriber initialization."""

    def test_initializes_with_config(self, mock_config):
        """RemoteTranscriber should accept a config parameter."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config)
            assert t._config is mock_config

    def test_client_uses_endpoint_from_config(self, mock_config):
        """OpenAI client should be initialized with the configured endpoint."""
        with mock.patch.object(OpenAI, "__init__", return_value=None) as mock_init:
            RemoteTranscriber(config=mock_config)
            mock_init.assert_called_once_with(
                base_url="https://transcribe.example.com/v1",
                api_key="transcribe-key",
            )

    def test_missing_endpoint_raises_error(self, mock_config):
        """Should raise TranscriptionError when endpoint is not configured."""
        mock_config.transcription_endpoint = None
        with (
            mock.patch.object(OpenAI, "__init__", return_value=None),
            pytest.raises(TranscriptionError, match="TRANSCRIPTION_ENDPOINT is not configured"),
        ):
            RemoteTranscriber(config=mock_config)

    def test_missing_api_key_raises_error(self, mock_config):
        """Should raise TranscriptionError when API key is not configured."""
        mock_config.transcription_api_key = None
        with (
            mock.patch.object(OpenAI, "__init__", return_value=None),
            pytest.raises(TranscriptionError, match="TRANSCRIPTION_API_KEY is not configured"),
        ):
            RemoteTranscriber(config=mock_config)

    def test_client_init_failure_raises_transcription_error(self, mock_config):
        """Should raise TranscriptionError if OpenAI client init fails."""
        with (
            mock.patch.object(OpenAI, "__init__", side_effect=RuntimeError("connection refused")),
            pytest.raises(TranscriptionError, match="Failed to initialize OpenAI client"),
        ):
            RemoteTranscriber(config=mock_config)

    def test_uses_get_config_when_no_config(self):
        """RemoteTranscriber should use get_config() when no config is provided."""
        env = {
            "BASE_URL": "https://example.com",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_SYSTEM_PROMPT": "Test prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-key",
            "TRANSCRIPTION_MODEL": "whisper-1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            import local_feed_generator.config as config_module

            config_module._config_instance = None

            with mock.patch.object(OpenAI, "__init__", return_value=None):
                t = RemoteTranscriber()
                assert t._config is not None
                assert t._config.transcription_endpoint == "https://transcribe.example.com/v1"

            config_module._config_instance = None


# ---------------------------------------------------------------------------
# transcribe() — file existence
# ---------------------------------------------------------------------------


class TestTranscribeFileExistence:
    """Tests for file existence validation in transcribe()."""

    def test_transcribe_nonexistent_file_raises_error(self, remote_transcriber):
        """transcribe() should raise TranscriptionError for non-existent files."""
        with mock.patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("nonexistent.mp4")

            assert "File not found" in str(exc_info.value)
            assert "nonexistent.mp4" in str(exc_info.value)

    def test_transcribe_nonexistent_file_sets_filepath(self, remote_transcriber):
        """TranscriptionError should include the filepath attribute."""
        with mock.patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("/path/to/nonexistent.mp4")

            assert exc_info.value.filepath == "/path/to/nonexistent.mp4"


# ---------------------------------------------------------------------------
# transcribe() — output and progress
# ---------------------------------------------------------------------------


class TestTranscribeOutput:
    """Tests for transcript output and progress output in transcribe()."""

    def test_transcribe_returns_text_from_api(self, remote_transcriber):
        """transcribe() should return the text returned by the API."""
        remote_transcriber._client.audio.transcriptions.create.return_value = (
            "This is the transcript text."
        )

        with mock.patch.object(Path, "exists", return_value=True):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "This is the transcript text."

    def test_transcribe_handles_json_string_response(self, remote_transcriber):
        """transcribe() should extract text from JSON string response (vLLM behavior).

        Some OpenAI-compatible endpoints (e.g. vLLM) return a JSON-encoded string
        even when response_format='text' is specified. The transcriber should
        parse the JSON and return just the text field.
        """
        json_response = json.dumps(
            {"text": "This is the transcript text.", "usage": {"type": "duration", "seconds": 100}}
        )
        remote_transcriber._client.audio.transcriptions.create.return_value = json_response

        with mock.patch.object(Path, "exists", return_value=True):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "This is the transcript text."

    def test_transcribe_handles_json_string_with_whitespace(self, remote_transcriber):
        """transcribe() should strip whitespace from text extracted from JSON response."""
        json_response = json.dumps(
            {
                "text": "  This is the transcript text.  ",
                "usage": {"type": "duration", "seconds": 100},
            }
        )
        remote_transcriber._client.audio.transcriptions.create.return_value = json_response

        with mock.patch.object(Path, "exists", return_value=True):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "This is the transcript text."

    def test_transcribe_strips_whitespace_from_response(self, remote_transcriber):
        """transcribe() should strip leading/trailing whitespace from the API response."""
        remote_transcriber._client.audio.transcriptions.create.return_value = (
            "  This is the transcript text.  "
        )

        with mock.patch.object(Path, "exists", return_value=True):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "This is the transcript text."

    def test_transcribe_falls_back_to_transcription_model(self, remote_transcriber):
        """transcribe() should pass transcription_model as model when remote_model is not set.

        The OpenAI SDK requires a model parameter, so transcription_model is
        used as a fallback when transcription_remote_model is not configured.
        """
        remote_transcriber._client.audio.transcriptions.create.return_value = "text"

        with mock.patch.object(Path, "exists", return_value=True):
            remote_transcriber.transcribe("fake.mp4")

        call_kwargs = remote_transcriber._client.audio.transcriptions.create.call_args
        # mock_config has transcription_model="whisper-1" and no transcription_remote_model
        assert call_kwargs.kwargs["model"] == "whisper-1"

    def test_transcribe_passes_model_when_configured(self, mock_config):
        """transcribe() should pass model parameter when transcription_remote_model is set."""
        mock_config.transcription_remote_model = "whisper-1"
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config)
            t._client = mock.MagicMock()
            t._client.audio.transcriptions.create.return_value = "text"

            with mock.patch.object(Path, "exists", return_value=True):
                t.transcribe("fake.mp4")

        call_kwargs = t._client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-1"

    def test_transcribe_remote_model_takes_precedence_over_transcription_model(self, mock_config):
        """transcribe() should use transcription_remote_model over transcription_model."""
        mock_config.transcription_model = "large-v3-turbo"
        mock_config.transcription_remote_model = "whisper-1"
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config)
            t._client = mock.MagicMock()
            t._client.audio.transcriptions.create.return_value = "text"

            with mock.patch.object(Path, "exists", return_value=True):
                t.transcribe("fake.mp4")

        call_kwargs = t._client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-1"

    def test_transcribe_uses_response_format_text(self, remote_transcriber):
        """transcribe() should request response_format='text' for clean text."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "text"

        with mock.patch.object(Path, "exists", return_value=True):
            remote_transcriber.transcribe("fake.mp4")

        call_kwargs = remote_transcriber._client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["response_format"] == "text"

    def test_transcribe_passes_file_path(self, remote_transcriber):
        """transcribe() should pass the file path to the API."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "text"

        with mock.patch.object(Path, "exists", return_value=True):
            remote_transcriber.transcribe("my_audio.mp4")

        call_kwargs = remote_transcriber._client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["file"] == Path("my_audio.mp4")


class TestTranscribeProgressOutput:
    """Tests for progress output in transcribe()."""

    def test_transcribe_prints_start_metadata_with_duration(self, remote_transcriber, capsys):
        """transcribe() should print file name, duration, and size at start."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "Hello world"

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch.object(remote_transcriber, "_get_audio_duration", return_value=120.5),
        ):
            mock_stat.return_value.st_size = 12345
            remote_transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing lecture.mp4: 120.5s, 12,345 bytes" in captured.out

    def test_transcribe_prints_start_metadata_without_duration(self, remote_transcriber, capsys):
        """transcribe() should print file name and size when duration unavailable."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "Hello world"

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch.object(remote_transcriber, "_get_audio_duration", return_value=None),
        ):
            mock_stat.return_value.st_size = 12345
            remote_transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing lecture.mp4: 12,345 bytes" in captured.out

    def test_transcribe_prints_start_metadata_no_size(self, remote_transcriber, capsys):
        """transcribe() should print just filename when size is unavailable."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "Hello world"

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat", side_effect=OSError("stat failed")),
            mock.patch.object(remote_transcriber, "_get_audio_duration", return_value=None),
        ):
            remote_transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing lecture.mp4" in captured.out

    def test_transcribe_prints_completion_time(self, remote_transcriber, capsys):
        """transcribe() should print elapsed time after completion."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "Hello world"

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch.object(remote_transcriber, "_get_audio_duration", return_value=60.0),
        ):
            mock_stat.return_value.st_size = 1000
            remote_transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Completed in" in captured.out
        assert "s" in captured.out

    def test_transcribe_output_goes_to_stdout(self, remote_transcriber, capsys):
        """transcribe() output should go to stdout, not stderr."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "Hello world"

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch.object(remote_transcriber, "_get_audio_duration", return_value=30.0),
        ):
            mock_stat.return_value.st_size = 5000
            remote_transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing" in captured.out
        assert "Completed in" in captured.out
        assert captured.err == ""

    def test_transcribe_does_not_print_completion_on_error(self, remote_transcriber, capsys):
        """transcribe() should not print completion time on transcription error."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock.MagicMock()
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch.object(remote_transcriber, "_get_audio_duration", return_value=30.0),
            pytest.raises(TranscriptionError),
        ):
            mock_stat.return_value.st_size = 5000
            remote_transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Completed in" not in captured.out


# ---------------------------------------------------------------------------
# transcribe() — error handling
# ---------------------------------------------------------------------------


class TestTranscribeErrorHandling:
    """Tests for error handling in transcribe()."""

    def test_api_connection_error_wrapped(self, remote_transcriber):
        """APIConnectionError should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock.MagicMock()
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
        ):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("fake.mp4")

            assert "failed after" in str(exc_info.value)
            assert exc_info.value.filepath == "fake.mp4"

    def test_authentication_error_wrapped(self, remote_transcriber):
        """AuthenticationError should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=mock.MagicMock(status_code=401),
            body=None,
        )

        with mock.patch.object(Path, "exists", return_value=True):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("fake.mp4")

            assert "non-retryable" in str(exc_info.value)
            assert exc_info.value.filepath == "fake.mp4"

    def test_rate_limit_error_wrapped(self, remote_transcriber):
        """RateLimitError should be wrapped in TranscriptionError after retries."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = RateLimitError(
            message="Rate limited",
            response=mock.MagicMock(status_code=429),
            body=None,
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError),
        ):
            remote_transcriber.transcribe("fake.mp4")

    def test_bad_request_error_wrapped(self, remote_transcriber):
        """BadRequestError should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = BadRequestError(
            message="Bad request",
            response=mock.MagicMock(status_code=400),
            body=None,
        )

        with mock.patch.object(Path, "exists", return_value=True):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("fake.mp4")

            assert "non-retryable" in str(exc_info.value)
            assert exc_info.value.filepath == "fake.mp4"

    def test_unexpected_error_wrapped(self, remote_transcriber):
        """Unexpected exceptions should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = ValueError(
            "Unexpected issue"
        )

        with mock.patch.object(Path, "exists", return_value=True):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("fake.mp4")

            assert "Unexpected error" in str(exc_info.value)
            assert exc_info.value.filepath == "fake.mp4"

    def test_transcription_error_is_subclass_of_processing_error(self, remote_transcriber):
        """TranscriptionError should be a ProcessingError subclass."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock.MagicMock()
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(ProcessingError),
        ):
            remote_transcriber.transcribe("fake.mp4")


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for exponential backoff retry logic in _create_transcription()."""

    def test_retries_on_api_connection_error(self, remote_transcriber):
        """Should retry on APIConnectionError and eventually succeed."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = [
            APIConnectionError(message="Connection failed", request=mock.MagicMock()),
            "Success",
        ]

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
        ):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "Success"
        assert remote_transcriber._client.audio.transcriptions.create.call_count == 2

    def test_retries_on_timeout_error(self, remote_transcriber):
        """Should retry on APITimeoutError and eventually succeed."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = [
            APITimeoutError(request=mock.MagicMock()),
            "Success",
        ]

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
        ):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "Success"
        assert remote_transcriber._client.audio.transcriptions.create.call_count == 2

    def test_retries_on_rate_limit_error(self, remote_transcriber):
        """Should retry on RateLimitError and eventually succeed."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = [
            RateLimitError(
                message="Rate limited",
                response=mock.MagicMock(status_code=429),
                body=None,
            ),
            "Success",
        ]

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
        ):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "Success"
        assert remote_transcriber._client.audio.transcriptions.create.call_count == 2

    def test_retries_on_internal_server_error(self, remote_transcriber):
        """Should retry on InternalServerError and eventually succeed."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = [
            InternalServerError(
                message="Server error",
                response=mock.MagicMock(status_code=500),
                body=None,
            ),
            "Success",
        ]

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
        ):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == "Success"
        assert remote_transcriber._client.audio.transcriptions.create.call_count == 2

    def test_does_not_retry_on_authentication_error(self, remote_transcriber):
        """Should not retry on AuthenticationError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=mock.MagicMock(status_code=401),
            body=None,
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError, match="non-retryable"),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert remote_transcriber._client.audio.transcriptions.create.call_count == 1

    def test_does_not_retry_on_bad_request_error(self, remote_transcriber):
        """Should not retry on BadRequestError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = BadRequestError(
            message="Bad request",
            response=mock.MagicMock(status_code=400),
            body=None,
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError, match="non-retryable"),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert remote_transcriber._client.audio.transcriptions.create.call_count == 1

    def test_does_not_retry_on_permission_denied_error(self, remote_transcriber):
        """Should not retry on PermissionDeniedError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = PermissionDeniedError(
            message="Forbidden",
            response=mock.MagicMock(status_code=403),
            body=None,
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError, match="non-retryable"),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert remote_transcriber._client.audio.transcriptions.create.call_count == 1

    def test_does_not_retry_on_not_found_error(self, remote_transcriber):
        """Should not retry on NotFoundError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = NotFoundError(
            message="Not found",
            response=mock.MagicMock(status_code=404),
            body=None,
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError, match="non-retryable"),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert remote_transcriber._client.audio.transcriptions.create.call_count == 1

    def test_does_not_retry_on_unexpected_error(self, remote_transcriber):
        """Should not retry on unexpected (non-OpenAI) exceptions."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = ValueError(
            "Unexpected issue"
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert remote_transcriber._client.audio.transcriptions.create.call_count == 1

    def test_exhausts_retries_on_persistent_failure(self, remote_transcriber):
        """Should raise TranscriptionError after max retries."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock.MagicMock()
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            pytest.raises(TranscriptionError, match="failed after"),
        ):
            remote_transcriber.transcribe("fake.mp4")

        # Default max_retries=3, so 4 total attempts
        assert remote_transcriber._client.audio.transcriptions.create.call_count == 4

    def test_custom_max_retries(self, mock_config):
        """Should respect custom max_retries."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config, max_retries=1)
            t._client = mock.MagicMock()
            t._client.audio.transcriptions.create.side_effect = APIConnectionError(
                message="Connection failed", request=mock.MagicMock()
            )

            with (
                mock.patch.object(Path, "exists", return_value=True),
                mock.patch("time.sleep"),
                pytest.raises(TranscriptionError, match="failed after"),
            ):
                t.transcribe("fake.mp4")

            # max_retries=1, so 2 total attempts
            assert t._client.audio.transcriptions.create.call_count == 2

    def test_retry_uses_exponential_backoff(self, remote_transcriber):
        """Retry delays should increase exponentially."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            APIConnectionError(message="fail", request=mock.MagicMock()),
            "Success",
        ]

        delays = []
        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep") as mock_sleep,
        ):
            mock_sleep.side_effect = lambda d: delays.append(d)
            remote_transcriber.transcribe("fake.mp4")

        assert len(delays) == 2
        # First delay: base_delay * 2^0 = 1.0
        # Second delay: base_delay * 2^1 = 2.0
        # With jitter, the actual values may vary slightly
        assert delays[0] < delays[1] or abs(delays[0] - delays[1]) < 0.5

    def test_retry_logs_warning(self, remote_transcriber, caplog):
        """Retry attempts should log a warning."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            "Success",
        ]

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            caplog.at_level("WARNING"),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert any("retrying" in record.message.lower() for record in caplog.records)

    def test_final_failure_logs_error(self, remote_transcriber, caplog):
        """Final failure after all retries should log an error."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = APIConnectionError(
            message="fail", request=mock.MagicMock()
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("time.sleep"),
            caplog.at_level("ERROR"),
            pytest.raises(TranscriptionError),
        ):
            remote_transcriber.transcribe("fake.mp4")

        assert any("failed after" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# Backoff calculation tests
# ---------------------------------------------------------------------------


class TestBackoffCalculation:
    """Tests for the _calculate_backoff method."""

    def test_first_attempt_delay(self, remote_transcriber):
        """First retry delay should be approximately base_delay (1.0s)."""
        delay = remote_transcriber._calculate_backoff(0)
        # base_delay * 2^0 = 1.0, with jitter ±10%
        assert 0.9 <= delay <= 1.1

    def test_second_attempt_delay(self, remote_transcriber):
        """Second retry delay should be approximately 2x base_delay."""
        delay = remote_transcriber._calculate_backoff(1)
        # base_delay * 2^1 = 2.0, with jitter ±20%
        assert 1.6 <= delay <= 2.4

    def test_third_attempt_delay(self, remote_transcriber):
        """Third retry delay should be approximately 4x base_delay."""
        delay = remote_transcriber._calculate_backoff(2)
        # base_delay * 2^2 = 4.0, with jitter ±40%
        assert 2.4 <= delay <= 5.6

    def test_delay_never_negative(self, remote_transcriber):
        """Backoff delay should never be negative."""
        for attempt in range(10):
            delay = remote_transcriber._calculate_backoff(attempt)
            assert delay >= 0.0

    def test_custom_base_delay(self, mock_config):
        """Custom base_delay should be respected."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config, base_delay=2.0, jitter=0.0)
            delay = t._calculate_backoff(0)
            assert delay == 2.0  # No jitter, so exact value

    def test_zero_jitter(self, mock_config):
        """With zero jitter, delays should be exact powers of two."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config, base_delay=1.0, jitter=0.0)
            assert t._calculate_backoff(0) == 1.0
            assert t._calculate_backoff(1) == 2.0
            assert t._calculate_backoff(2) == 4.0
            assert t._calculate_backoff(3) == 8.0


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestGetFileSize:
    """Tests for the _get_file_size helper method."""

    def test_returns_file_size(self, remote_transcriber, tmp_path):
        """_get_file_size should return the file size in bytes."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"x" * 1024)

        result = RemoteTranscriber._get_file_size(audio_file)
        assert result == 1024

    def test_returns_none_on_os_error(self, remote_transcriber):
        """_get_file_size should return None when stat fails."""
        audio_file = Path("/nonexistent/path/test.mp4")

        result = RemoteTranscriber._get_file_size(audio_file)
        assert result is None


class TestGetAudioDuration:
    """Tests for the _get_audio_duration helper method."""

    def test_returns_duration_from_av(self, remote_transcriber, tmp_path):
        """_get_audio_duration should return duration from av container."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        mock_stream = mock.MagicMock()
        mock_stream.duration_seconds = 42.5

        mock_container = mock.MagicMock()
        mock_container.streams.audio = [mock_stream]

        mock_av_module = mock.MagicMock()
        mock_av_module.open.return_value.__enter__.return_value = mock_container

        with mock.patch.dict("sys.modules", {"av": mock_av_module}):
            result = RemoteTranscriber._get_audio_duration(audio_file)

        assert result == 42.5

    def test_returns_none_when_no_audio_streams(self, remote_transcriber, tmp_path):
        """_get_audio_duration should return None when no audio streams exist."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        mock_container = mock.MagicMock()
        mock_container.streams.audio = []

        mock_av_module = mock.MagicMock()
        mock_av_module.open.return_value.__enter__.return_value = mock_container

        with mock.patch.dict("sys.modules", {"av": mock_av_module}):
            result = RemoteTranscriber._get_audio_duration(audio_file)

        assert result is None

    def test_returns_none_on_av_error(self, remote_transcriber, tmp_path):
        """_get_audio_duration should return None when av.open raises."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        mock_av_module = mock.MagicMock()
        mock_av_module.open.side_effect = RuntimeError("av error")

        with mock.patch.dict("sys.modules", {"av": mock_av_module}):
            result = RemoteTranscriber._get_audio_duration(audio_file)

        assert result is None

    def test_returns_none_when_av_not_available(self, remote_transcriber, tmp_path):
        """_get_audio_duration should return None when av import fails."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        with mock.patch.dict("sys.modules", {"av": None}):
            result = RemoteTranscriber._get_audio_duration(audio_file)

        assert result is None
