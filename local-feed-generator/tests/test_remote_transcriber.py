"""Tests for the RemoteTranscriber service.

Tests cover:
- Initialization with config (endpoint, api_key, model)
- Configuration validation (missing endpoint, missing api_key)
- transcribe() method: transcript formatting (no timestamps), error handling
- Non-retryable errors (auth, bad request, permission denied, not found)
- _get_audio_duration() helper method

NOTE: Retry logic and backoff calculation tests are consolidated in
test_summarizer.py since both Summarizer and RemoteTranscriber share
identical retry implementations.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from unittest import mock

from local_feed_generator.config import Settings
from local_feed_generator.errors import TranscriptionError
from local_feed_generator.services.remote_transcriber import RemoteTranscriber
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Settings:
    """Provide a Settings instance with test values for remote transcription."""
    return Settings(  # type: ignore[call-arg]
        base_url="https://example.com/feeds",
        data_dir="./data",
        llm_endpoint="https://llm.example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        prompt_article="Summarize key takeaways.",
        prompt_article_reflection="Summarize key takeaways with reflection.",
        prompt_podcast="Summarize key takeaways from podcast.",
        prompt_podcast_reflection="Summarize key takeaways from podcast with reflection.",
        prompt_title="Generate a concise title.",
        transcription_endpoint="https://transcribe.example.com/v1",
        transcription_api_key="transcribe-key",
        transcription_model="whisper-1",
    )


@pytest.fixture
def remote_transcriber(mock_config: Settings) -> Iterator[RemoteTranscriber]:
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

    def test_initializes_with_config(self, mock_config: Settings) -> None:
        """RemoteTranscriber should accept a config parameter."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config)
            assert t._config is mock_config

    def test_client_uses_endpoint_from_config(self, mock_config: Settings) -> None:
        """OpenAI client should be initialized with the configured endpoint."""
        with mock.patch.object(OpenAI, "__init__", return_value=None) as mock_init:
            RemoteTranscriber(config=mock_config)
            mock_init.assert_called_once_with(
                base_url="https://transcribe.example.com/v1",
                api_key="transcribe-key",
            )

    @pytest.mark.parametrize(
        "unset_attr,error_msg",
        [
            ("transcription_endpoint", "TRANSCRIPTION_ENDPOINT is not configured"),
            ("transcription_api_key", "TRANSCRIPTION_API_KEY is not configured"),
        ],
    )
    def test_missing_config_raises_error(
        self, mock_config: Settings, unset_attr: str, error_msg: str
    ) -> None:
        """Should raise TranscriptionError when required config is missing."""
        setattr(mock_config, unset_attr, None)
        with (
            mock.patch.object(OpenAI, "__init__", return_value=None),
            pytest.raises(TranscriptionError, match=error_msg),
        ):
            RemoteTranscriber(config=mock_config)

    def test_client_init_failure_raises_transcription_error(self, mock_config: Settings) -> None:
        """Should raise TranscriptionError if OpenAI client init fails."""
        with (
            mock.patch.object(OpenAI, "__init__", side_effect=RuntimeError("connection refused")),
            pytest.raises(TranscriptionError, match="Failed to initialize OpenAI client"),
        ):
            RemoteTranscriber(config=mock_config)


# ---------------------------------------------------------------------------
# transcribe() — file existence
# ---------------------------------------------------------------------------


class TestTranscribeFileExistence:
    """Tests for file existence validation in transcribe().

    NOTE: File existence validation tests are covered in test_transcriber.py
    (TestTranscribeFileExistence) since Transcriber and RemoteTranscriber
    share identical file-existence checking behavior.
    """


# ---------------------------------------------------------------------------
# transcribe() — output and progress
# ---------------------------------------------------------------------------


class TestTranscribeOutput:
    """Tests for transcript output and API call parameters in transcribe()."""

    @pytest.mark.parametrize(
        "api_response,expected_result",
        [
            ("This is the transcript text.", "This is the transcript text."),
            ("  This is the transcript text.  ", "This is the transcript text."),
            (
                json.dumps(
                    {
                        "text": "This is the transcript text.",
                        "usage": {"type": "duration", "seconds": 100},
                    }
                ),
                "This is the transcript text.",
            ),
            (
                json.dumps(
                    {
                        "text": "  This is the transcript text.  ",
                        "usage": {"type": "duration", "seconds": 100},
                    }
                ),
                "This is the transcript text.",
            ),
        ],
    )
    def test_transcribe_output_handling(
        self, remote_transcriber: RemoteTranscriber, api_response: str, expected_result: str
    ) -> None:
        """transcribe() should handle various API response formats and strip whitespace."""
        remote_transcriber._client.audio.transcriptions.create.return_value = api_response  # type: ignore[attr-defined]

        with mock.patch.object(Path, "exists", return_value=True):
            result = remote_transcriber.transcribe("fake.mp4")

        assert result == expected_result

    @pytest.mark.parametrize(
        "remote_model,transcription_model,expected_model",
        [
            (None, "whisper-1", "whisper-1"),  # Falls back to transcription_model
            ("whisper-1", "large-v3-turbo", "whisper-1"),  # Remote model takes precedence
        ],
    )
    def test_transcribe_model_selection(
        self,
        mock_config: Settings,
        remote_model: str | None,
        transcription_model: str,
        expected_model: str,
    ) -> None:
        """transcribe() should select the correct model based on config."""
        mock_config.transcription_model = transcription_model
        mock_config.transcription_remote_model = remote_model
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            t = RemoteTranscriber(config=mock_config)
            t._client = mock.MagicMock()
            t._client.audio.transcriptions.create.return_value = "text"

            with mock.patch.object(Path, "exists", return_value=True):
                t.transcribe("fake.mp4")

        call_kwargs = t._client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == expected_model

    def test_transcribe_api_call_params(self, remote_transcriber: RemoteTranscriber) -> None:
        """transcribe() should pass correct parameters to the API."""
        remote_transcriber._client.audio.transcriptions.create.return_value = "text"  # type: ignore[attr-defined]

        with mock.patch.object(Path, "exists", return_value=True):
            remote_transcriber.transcribe("my_audio.mp4")

        call_kwargs = remote_transcriber._client.audio.transcriptions.create.call_args  # type: ignore[attr-defined]
        assert call_kwargs.kwargs["response_format"] == "text"
        assert call_kwargs.kwargs["file"] == Path("my_audio.mp4")


class TestTranscribeProgressOutput:
    """Tests for progress output in transcribe().

    NOTE: Progress output tests are consolidated in test_transcriber.py
    (TestTranscribeMetadataOutput) since both Transcriber and RemoteTranscriber
    share identical print behavior.
    """


# ---------------------------------------------------------------------------
# transcribe() — error handling
# ---------------------------------------------------------------------------


class TestTranscribeErrorHandling:
    """Tests for error handling in transcribe()."""

    def test_api_connection_error_wrapped(self, remote_transcriber: RemoteTranscriber) -> None:
        """APIConnectionError should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = APIConnectionError(  # type: ignore[attr-defined]
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

    def test_authentication_error_wrapped(self, remote_transcriber: RemoteTranscriber) -> None:
        """AuthenticationError should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = AuthenticationError(  # type: ignore[attr-defined]
            message="Invalid API key",
            response=mock.MagicMock(status_code=401),
            body=None,
        )

        with mock.patch.object(Path, "exists", return_value=True):
            with pytest.raises(TranscriptionError) as exc_info:
                remote_transcriber.transcribe("fake.mp4")

            assert "non-retryable" in str(exc_info.value)
            assert exc_info.value.filepath == "fake.mp4"

    def test_rate_limit_error_wrapped(self, remote_transcriber: RemoteTranscriber) -> None:
        """RateLimitError should be wrapped in TranscriptionError after retries."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = RateLimitError(  # type: ignore[attr-defined]
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

    def test_bad_request_error_wrapped(self, remote_transcriber: RemoteTranscriber) -> None:
        """BadRequestError should be wrapped in TranscriptionError."""
        remote_transcriber._client.audio.transcriptions.create.side_effect = BadRequestError(  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------
# Retry logic and backoff calculation tests
# ---------------------------------------------------------------------------
# NOTE: Retry/backoff tests are consolidated in test_summarizer.py
# (TestRetryLogic and TestBackoffCalculation) since both Summarizer and
# RemoteTranscriber share identical retry logic implementations
# (_calculate_backoff method and retry loop structure).


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestGetFileSize:
    """Tests for the _get_file_size helper method.

    NOTE: _get_file_size tests are consolidated in test_transcriber.py
    (TestGetFileSize) since both Transcriber and RemoteTranscriber share
    identical _get_file_size implementations.
    """


class TestGetAudioDuration:
    """Tests for the _get_audio_duration helper method.

    NOTE: _get_audio_duration tests are consolidated in test_transcriber.py
    (TestGetAudioDuration) since both Transcriber and RemoteTranscriber share
    identical _get_audio_duration implementations.
    """
