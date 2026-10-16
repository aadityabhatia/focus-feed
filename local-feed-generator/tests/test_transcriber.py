"""Tests for the Transcriber service.

Tests cover:
- check_cuda_available() utility function
- Model singleton pattern (loaded once, reused)
- Configuration loading from env vars
- transcribe() method: file existence check, transcript formatting (no timestamps),
  error handling
- cleanup() method for resource release
- CUDA availability check in _load_model()
"""

import ctypes
import json
import os
from pathlib import Path
from unittest import mock

from local_feed_generator.errors import TranscriptionError
from local_feed_generator.services.transcriber import Transcriber, check_cuda_available
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _set_valid_env(monkeypatch):
    """Set valid environment variables for config."""
    monkeypatch.setenv("BASE_URL", "https://example.com/feeds")
    monkeypatch.setenv("DATA_DIR", "./data")
    monkeypatch.setenv("LLM_ENDPOINT", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_SYSTEM_PROMPT", "Test system prompt")
    monkeypatch.setenv("SUMMARY_PROMPT", "Test summary prompt")
    monkeypatch.setenv("TITLE_PROMPT", "Test title prompt")
    # Clear transcription endpoint to force local mode (not remote).
    # Using setenv with empty string overrides the .env file value (env vars
    # take precedence over .env in pydantic-settings).
    monkeypatch.setenv("TRANSCRIPTION_ENDPOINT", "")
    monkeypatch.setenv("TRANSCRIPTION_API_KEY", "")
    monkeypatch.setenv("TRANSCRIPTION_REMOTE_MODEL", "")
    # Reset config singleton so it picks up new env
    import local_feed_generator.config as config_module

    config_module._config_instance = None


@pytest.fixture
def valid_env(monkeypatch):
    """Provide valid environment variables for all required config fields."""
    _set_valid_env(monkeypatch)
    # Mock CUDA availability check so existing tests that mock WhisperModel
    # can proceed without requiring an actual CUDA installation.
    monkeypatch.setattr(
        "local_feed_generator.services.transcriber.check_cuda_available",
        lambda: True,
    )
    yield
    # Cleanup: reset config singleton
    import local_feed_generator.config as config_module

    config_module._config_instance = None


# ---------------------------------------------------------------------------
# Mock segment class for testing transcribe()
# ---------------------------------------------------------------------------


class MockSegment:
    """Mock Segment object with a text attribute."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockTranscriptionInfo:
    """Mock TranscriptionInfo object."""

    pass


class MockWhisperModel:
    """Mock WhisperModel that returns predefined segments."""

    def __init__(self, segments_data=None, raise_on_transcribe=False):
        self._segments_data = segments_data or []
        self._raise_on_transcribe = raise_on_transcribe
        self.delete_called = False

    def transcribe(self, audio, **kwargs):
        if self._raise_on_transcribe:
            raise RuntimeError("Mock transcription error")
        segments = [MockSegment(text) for text in self._segments_data]
        return segments, MockTranscriptionInfo()

    def delete(self):
        self.delete_called = True


# ---------------------------------------------------------------------------
# Tests: Initialization and model loading
# ---------------------------------------------------------------------------


class TestTranscriberInitialization:
    """Tests for Transcriber initialization."""

    def test_transcriber_starts_with_no_model(self, valid_env):
        """Transcriber should start with no model loaded."""
        transcriber = Transcriber()
        assert transcriber.model is None

    def test_model_property_returns_none_initially(self, valid_env):
        """The model property should return None before loading."""
        transcriber = Transcriber()
        assert transcriber.model is None


# ---------------------------------------------------------------------------
# Tests: Model singleton pattern
# ---------------------------------------------------------------------------


class TestModelSingleton:
    """Tests for the singleton model loading pattern."""

    def test_model_loaded_once_and_reused(self, valid_env):
        """Model should be loaded only once and reused for subsequent calls."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello world"])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            # Create a dummy file so transcribe doesn't fail on existence check
            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")
                transcriber.transcribe("fake2.mp4")

            # WhisperModel should have been instantiated only once
            assert mock_whisper_cls.call_count == 1

    def test_model_loaded_lazily_on_first_transcribe(self, valid_env):
        """Model should not be loaded until transcribe() is called."""
        transcriber = Transcriber()

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            # Model should not be loaded just by creating Transcriber
            assert mock_whisper_cls.call_count == 0

            mock_model = MockWhisperModel(segments_data=["Hello"])
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")

            # Model should be loaded after first transcribe call
            assert mock_whisper_cls.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Configuration loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Tests for configuration loading from env vars."""

    def test_uses_config_defaults(self, valid_env):
        """Transcriber should use config defaults when env vars are not set."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello"])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")

            # Check that WhisperModel was called with config defaults
            # model_name is passed as positional arg, device/compute_type as kwargs
            call_args = mock_whisper_cls.call_args
            assert call_args.args[0] == "large-v3-turbo"
            assert call_args.kwargs["device"] == "cuda"
            assert call_args.kwargs["compute_type"] == "float16"

    def test_uses_config_overrides(self, valid_env):
        """Transcriber should use overridden env var values."""
        import local_feed_generator.config as config_module

        config_module._config_instance = None

        # Set custom transcription config
        os.environ["TRANSCRIPTION_MODEL"] = "tiny"
        os.environ["TRANSCRIPTION_DEVICE"] = "cpu"
        os.environ["TRANSCRIPTION_COMPUTE_TYPE"] = "int8"

        try:
            transcriber = Transcriber()
            mock_model = MockWhisperModel(segments_data=["Hello"])

            with mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls:
                mock_whisper_cls.return_value = mock_model

                with mock.patch.object(Path, "exists", return_value=True):
                    transcriber.transcribe("fake.mp4")

                # model_name is passed as positional arg, device/compute_type as kwargs
                call_args = mock_whisper_cls.call_args
                assert call_args.args[0] == "tiny"
                assert call_args.kwargs["device"] == "cpu"
                assert call_args.kwargs["compute_type"] == "int8"
        finally:
            # Restore defaults
            del os.environ["TRANSCRIPTION_MODEL"]
            del os.environ["TRANSCRIPTION_DEVICE"]
            del os.environ["TRANSCRIPTION_COMPUTE_TYPE"]
            config_module._config_instance = None


# ---------------------------------------------------------------------------
# Tests: transcribe() method
# ---------------------------------------------------------------------------


class TestTranscribeFileExistence:
    """Tests for file existence validation in transcribe()."""

    def test_transcribe_nonexistent_file_raises_error(self, valid_env):
        """transcribe() should raise TranscriptionError for non-existent files."""
        transcriber = Transcriber()

        with mock.patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcriber.transcribe("nonexistent.mp4")

            assert "File not found" in str(exc_info.value)
            assert "nonexistent.mp4" in str(exc_info.value)

    def test_transcribe_nonexistent_file_sets_filepath(self, valid_env):
        """TranscriptionError should include the filepath attribute."""
        transcriber = Transcriber()

        with mock.patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcriber.transcribe("/path/to/nonexistent.mp4")

            assert exc_info.value.filepath == "/path/to/nonexistent.mp4"


class TestTranscribeOutput:
    """Tests for transcript output formatting."""

    def test_transcribe_returns_text_without_timestamps(self, valid_env):
        """transcribe() should return text without timestamp annotations."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(
            segments_data=[
                "Hello, this is a test.",
                "This is the second segment.",
            ]
        )

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        # Result should contain the text but not timestamps
        assert "Hello, this is a test." in result
        assert "This is the second segment." in result
        # No timestamp patterns like [0.00s -> 1.00s] should be present
        assert "[" not in result
        assert "->" not in result

    def test_transcribe_joins_segments_with_newlines(self, valid_env):
        """transcribe() should join segments with newlines."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(
            segments_data=[
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ]
        )

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "First sentence."
        assert lines[1] == "Second sentence."
        assert lines[2] == "Third sentence."

    def test_transcribe_strips_whitespace_from_segments(self, valid_env):
        """transcribe() should strip leading/trailing whitespace from segments."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(
            segments_data=[
                "  Padded text  ",
                "  More text  ",
            ]
        )

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        lines = result.split("\n")
        assert lines[0] == "Padded text"
        assert lines[1] == "More text"

    def test_transcribe_skips_empty_segments(self, valid_env):
        """transcribe() should skip empty segments."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(
            segments_data=[
                "Real text",
                "",
                "   ",
                "More text",
            ]
        )

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "Real text"
        assert lines[1] == "More text"

    def test_transcribe_single_segment(self, valid_env):
        """transcribe() should handle a single segment."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Only one segment."])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        assert result == "Only one segment."

    def test_transcribe_empty_transcript(self, valid_env):
        """transcribe() should return empty string for files with no speech."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=[])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        assert result == ""

    def test_transcribe_all_empty_segments_returns_empty_string(self, valid_env):
        """transcribe() should return empty string when all segments are empty."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["", "   ", ""])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        assert result == ""


# ---------------------------------------------------------------------------
# Tests: Audio metadata and progress output
# ---------------------------------------------------------------------------


class TestTranscribeMetadataOutput:
    """Tests for audio metadata and progress output in transcribe()."""

    # Short aliases for repeated mock.patch paths to stay within line length.
    _MPATH = "local_feed_generator.services.transcriber"

    def test_transcribe_prints_start_metadata_with_duration(self, valid_env, capsys):
        """transcribe() should print file name, duration, and size at start."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello world"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=120.5),
        ):
            mock_whisper_cls.return_value = mock_model
            mock_stat.return_value.st_size = 12345

            transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing lecture.mp4: 120.5s, 12,345 bytes" in captured.out

    def test_transcribe_prints_start_metadata_without_duration(self, valid_env, capsys):
        """transcribe() should print file name and size when duration unavailable."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello world"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=None),
        ):
            mock_whisper_cls.return_value = mock_model
            mock_stat.return_value.st_size = 12345

            transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing lecture.mp4: 12,345 bytes" in captured.out

    def test_transcribe_prints_start_metadata_no_size(self, valid_env, capsys):
        """transcribe() should print just filename when size is unavailable."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello world"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat", side_effect=OSError("stat failed")),
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=None),
        ):
            mock_whisper_cls.return_value = mock_model

            transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing lecture.mp4" in captured.out

    def test_transcribe_prints_completion_time_with_word_count(self, valid_env, capsys):
        """transcribe() should print elapsed time and word count after completion."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello world"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=60.0),
        ):
            mock_whisper_cls.return_value = mock_model
            mock_stat.return_value.st_size = 1000

            transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Completed in" in captured.out
        assert "s" in captured.out
        assert "words)" in captured.out

    def test_transcribe_output_goes_to_stdout(self, valid_env, capsys):
        """transcribe() output should go to stdout, not stderr."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello world"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=30.0),
        ):
            mock_whisper_cls.return_value = mock_model
            mock_stat.return_value.st_size = 5000

            transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Transcribing" in captured.out
        assert "Completed in" in captured.out
        assert captured.err == ""

    def test_transcribe_does_not_print_completion_on_error(self, valid_env, capsys):
        """transcribe() should not print completion time on transcription error."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello"], raise_on_transcribe=True)

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat") as mock_stat,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=30.0),
            pytest.raises(TranscriptionError),
        ):
            mock_whisper_cls.return_value = mock_model
            mock_stat.return_value.st_size = 5000

            transcriber.transcribe("lecture.mp4")

        captured = capsys.readouterr()
        assert "Completed in" not in captured.out


class TestTranscribeSkipAlreadyComplete:
    """Tests for skipping transcription when JSON marks it as complete."""

    _MPATH = "local_feed_generator.services.transcriber"

    def test_transcribe_skips_when_json_status_complete(self, valid_env, tmp_path):
        """Return existing transcript when JSON has status.transcribe=complete."""
        audio_file = tmp_path / "lecture.mp4"
        audio_file.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            json.dumps(
                {
                    "status": {"transcribe": "complete", "summarize": None},
                    "transcript": "Existing transcript text.",
                }
            )
        )

        transcriber = Transcriber()

        # WhisperModel should NOT be instantiated since we skip transcription
        with mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls:
            result = transcriber.transcribe(str(audio_file))

        assert result == "Existing transcript text."
        # Model should not have been loaded
        assert mock_whisper_cls.call_count == 0

    def test_transcribe_proceeds_when_json_status_not_complete(self, valid_env, tmp_path):
        """transcribe() should proceed when JSON status.transcribe is not 'complete'."""
        audio_file = tmp_path / "lecture.mp4"
        audio_file.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            json.dumps(
                {
                    "status": {"transcribe": "in_progress", "summarize": None},
                    "transcript": "",
                }
            )
        )

        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["New transcript"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=60.0),
        ):
            mock_whisper_cls.return_value = mock_model
            result = transcriber.transcribe(str(audio_file))

        assert result == "New transcript"
        assert mock_whisper_cls.call_count == 1

    def test_transcribe_proceeds_when_no_json_exists(self, valid_env, tmp_path):
        """transcribe() should proceed when no adjacent JSON file exists."""
        audio_file = tmp_path / "lecture.mp4"
        audio_file.write_bytes(b"video data")

        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Transcript"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=60.0),
        ):
            mock_whisper_cls.return_value = mock_model
            result = transcriber.transcribe(str(audio_file))

        assert result == "Transcript"
        assert mock_whisper_cls.call_count == 1

    def test_transcribe_proceeds_when_json_has_no_transcript(self, valid_env, tmp_path):
        """transcribe() should proceed when JSON marks complete but has no transcript."""
        audio_file = tmp_path / "lecture.mp4"
        audio_file.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            json.dumps(
                {
                    "status": {"transcribe": "complete", "summarize": None},
                    "transcript": "",
                }
            )
        )

        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Fresh transcript"])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=60.0),
        ):
            mock_whisper_cls.return_value = mock_model
            result = transcriber.transcribe(str(audio_file))

        assert result == "Fresh transcript"
        assert mock_whisper_cls.call_count == 1


class TestGetAudioDuration:
    """Tests for the _get_audio_duration helper method."""

    def test_returns_duration_from_av(self, valid_env, tmp_path):
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
            result = Transcriber._get_audio_duration(audio_file)

        assert result == 42.5

    def test_returns_none_when_no_audio_streams(self, valid_env, tmp_path):
        """_get_audio_duration should return None when no audio streams exist."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        mock_container = mock.MagicMock()
        mock_container.streams.audio = []

        mock_av_module = mock.MagicMock()
        mock_av_module.open.return_value.__enter__.return_value = mock_container

        with mock.patch.dict("sys.modules", {"av": mock_av_module}):
            result = Transcriber._get_audio_duration(audio_file)

        assert result is None

    def test_returns_none_on_av_error(self, valid_env, tmp_path):
        """_get_audio_duration should return None when av.open raises."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        mock_av_module = mock.MagicMock()
        mock_av_module.open.side_effect = RuntimeError("av error")

        with mock.patch.dict("sys.modules", {"av": mock_av_module}):
            result = Transcriber._get_audio_duration(audio_file)

        assert result is None

    def test_returns_none_when_av_not_available(self, valid_env, tmp_path):
        """_get_audio_duration should return None when av import fails."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        with mock.patch.dict("sys.modules", {"av": None}):
            result = Transcriber._get_audio_duration(audio_file)

        assert result is None


class TestGetFileSize:
    """Tests for the _get_file_size helper method."""

    def test_returns_file_size(self, valid_env, tmp_path):
        """_get_file_size should return the file size in bytes."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"x" * 1024)

        result = Transcriber._get_file_size(audio_file)
        assert result == 1024

    def test_returns_none_on_os_error(self, valid_env, tmp_path):
        """_get_file_size should return None when stat fails."""
        audio_file = tmp_path / "nonexistent.mp4"

        result = Transcriber._get_file_size(audio_file)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestTranscribeErrorHandling:
    """Tests for error handling in transcribe()."""

    def test_transcription_error_reraised(self, valid_env):
        """TranscriptionError raised during transcription should be re-raised."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello"], raise_on_transcribe=True)

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with (
                mock.patch.object(Path, "exists", return_value=True),
                pytest.raises(TranscriptionError) as exc_info,
            ):
                transcriber.transcribe("fake.mp4")

            assert "Transcription failed" in str(exc_info.value)
            assert "fake.mp4" in str(exc_info.value)
            assert exc_info.value.filepath == "fake.mp4"

    def test_model_load_error_wrapped_in_transcription_error(self, valid_env):
        """Model loading errors should be wrapped in TranscriptionError."""
        transcriber = Transcriber()

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.side_effect = RuntimeError("CUDA not available")

            with (
                mock.patch.object(Path, "exists", return_value=True),
                pytest.raises(TranscriptionError) as exc_info,
            ):
                transcriber.transcribe("fake.mp4")

            assert "Failed to load transcription model" in str(exc_info.value)
            assert "CUDA not available" in str(exc_info.value)

    def test_transcription_error_is_subclass_of_processing_error(self, valid_env):
        """TranscriptionError should be a ProcessingError subclass."""
        from local_feed_generator.errors import ProcessingError

        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello"], raise_on_transcribe=True)

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with (
                mock.patch.object(Path, "exists", return_value=True),
                pytest.raises(ProcessingError),
            ):
                transcriber.transcribe("fake.mp4")


# ---------------------------------------------------------------------------
# Tests: cleanup() method
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for the cleanup() method."""

    def test_cleanup_sets_model_to_none(self, valid_env):
        """cleanup() should set the model to None."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello"])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")

            assert transcriber.model is not None

            transcriber.cleanup()

            assert transcriber.model is None

    def test_cleanup_calls_model_delete(self, valid_env):
        """cleanup() should call the model's delete method if available."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello"])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")

            transcriber.cleanup()

            assert mock_model.delete_called is True

    def test_cleanup_without_loaded_model_is_noop(self, valid_env):
        """cleanup() should be safe to call when no model is loaded."""
        transcriber = Transcriber()
        # Should not raise
        transcriber.cleanup()
        assert transcriber.model is None

    def test_model_reloaded_after_cleanup(self, valid_env):
        """Model should be reloaded after cleanup() on next transcribe()."""
        transcriber = Transcriber()

        mock_model_1 = MockWhisperModel(segments_data=["First"])
        mock_model_2 = MockWhisperModel(segments_data=["Second"])

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            # First call loads model_1
            mock_whisper_cls.return_value = mock_model_1

            with mock.patch.object(Path, "exists", return_value=True):
                result1 = transcriber.transcribe("fake1.mp4")
                assert result1 == "First"

            # Cleanup
            transcriber.cleanup()
            assert transcriber.model is None

            # Second call should load model_2 (simulating reload)
            mock_whisper_cls.return_value = mock_model_2

            with mock.patch.object(Path, "exists", return_value=True):
                result2 = transcriber.transcribe("fake2.mp4")
                assert result2 == "Second"

            # WhisperModel should have been instantiated twice
            assert mock_whisper_cls.call_count == 2

    def test_cleanup_handles_delete_exception_gracefully(self, valid_env):
        """cleanup() should not raise if model.delete() throws."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=["Hello"])
        # Make delete raise an exception

        def failing_delete():
            raise RuntimeError("Delete failed")

        mock_model.delete = failing_delete

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")

            # Should not raise even though delete() raises
            transcriber.cleanup()
            assert transcriber.model is None

    def test_cleanup_with_model_without_delete_method(self, valid_env):
        """cleanup() should work even if model has no delete method."""
        transcriber = Transcriber()

        class SimpleModel:
            def transcribe(self, audio, **kwargs):
                return [MockSegment("Hello")], MockTranscriptionInfo()

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = SimpleModel()

            with mock.patch.object(Path, "exists", return_value=True):
                transcriber.transcribe("fake.mp4")

            # Should not raise even though SimpleModel has no delete()
            transcriber.cleanup()
            assert transcriber.model is None


# ---------------------------------------------------------------------------
# Tests: check_cuda_available()
# ---------------------------------------------------------------------------


class TestCheckCudaAvailable:
    """Tests for the check_cuda_available() utility function."""

    def test_returns_bool(self, valid_env):
        """check_cuda_available() should always return a bool, never crash."""
        result = check_cuda_available()
        assert isinstance(result, bool)

    def test_returns_true_when_library_loadable(self, valid_env):
        """Should return True when ctypes.CDLL successfully loads a library."""
        original_cdll = ctypes.CDLL

        def mock_cdll(name, *args, **kwargs):
            if "libcublas" in name:
                return original_cdll  # Return a dummy object
            raise OSError(f"cannot open shared object file: {name}")

        # Mock site to prevent finding real libraries in site-packages
        with (
            mock.patch("ctypes.CDLL", side_effect=mock_cdll),
            mock.patch("site.getsitepackages", return_value=[]),
            mock.patch("site.getusersitepackages", return_value=""),
        ):
            assert check_cuda_available() is True

    def test_returns_false_when_no_library_loadable(self, valid_env):
        """Should return False when ctypes.CDLL cannot load any library."""
        # Mock both CDLL and site.getsitepackages to prevent finding real libraries
        with (
            mock.patch("ctypes.CDLL", side_effect=OSError("cannot open shared object file")),
            mock.patch("site.getsitepackages", return_value=[]),
            mock.patch("site.getusersitepackages", return_value=""),
        ):
            assert check_cuda_available() is False

    def test_returns_false_when_ctypes_import_fails(self, valid_env):
        """Should return False if ctypes cannot be imported."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "ctypes":
                raise ImportError("No module named 'ctypes'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            assert check_cuda_available() is False

    def test_tries_versioned_library_names_first(self, valid_env):
        """Should try libcublas.so.12 before falling back to libcublas.so."""
        loaded_names: list[str] = []

        def mock_cdll(name, *args, **kwargs):
            loaded_names.append(name)
            raise OSError(f"cannot open shared object file: {name}")

        # Mock site to prevent finding real libraries in site-packages
        with (
            mock.patch("ctypes.CDLL", side_effect=mock_cdll),
            mock.patch("site.getsitepackages", return_value=[]),
            mock.patch("site.getusersitepackages", return_value=""),
        ):
            result = check_cuda_available()
            assert result is False
            # Should have tried all library names in standard paths
            assert "libcublas.so.12" in loaded_names
            assert "libcublas.so" in loaded_names

    def test_finds_library_in_site_packages(self, valid_env, tmp_path):
        """Should find CUDA libraries bundled in Python site-packages."""
        # Create a fake nvidia/cublas/lib directory in tmp_path
        nvidia_lib_dir = tmp_path / "nvidia" / "cublas" / "lib"
        nvidia_lib_dir.mkdir(parents=True)
        fake_lib = nvidia_lib_dir / "libcublas.so.12"
        fake_lib.write_bytes(b"fake library")

        # CDLL should fail for bare names (standard paths) but succeed for full path
        def mock_cdll(name, *args, **kwargs):
            if name == str(fake_lib):
                return mock.MagicMock()  # Return a mock object for the full path
            raise OSError(f"cannot open shared object file: {name}")

        with (
            mock.patch("ctypes.CDLL", side_effect=mock_cdll),
            mock.patch("site.getsitepackages", return_value=[str(tmp_path)]),
            mock.patch("site.getusersitepackages", return_value=""),
        ):
            assert check_cuda_available() is True


# ---------------------------------------------------------------------------
# Tests: CUDA availability check in _load_model()
# ---------------------------------------------------------------------------


class TestCudaCheckInLoadModel:
    """Tests for the CUDA availability check integrated into _load_model()."""

    def test_cuda_device_raises_when_cuda_unavailable(self, valid_env):
        """_load_model should raise TranscriptionError when CUDA is unavailable."""
        transcriber = Transcriber()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=False,
            ),
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            pytest.raises(TranscriptionError) as exc_info,
        ):
            transcriber._load_model()

            assert "CUDA libraries not found" in str(exc_info.value)
            # WhisperModel should not have been instantiated
            assert mock_whisper_cls.call_count == 0

    def test_cuda_device_does_not_raise_when_cuda_available(self, valid_env):
        """_load_model should proceed when CUDA is available."""
        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=["Hello"])

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=True,
            ),
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
        ):
            mock_whisper_cls.return_value = mock_model
            result = transcriber._load_model()
            assert mock_whisper_cls.call_count == 1
            assert result is mock_model

    def test_cpu_device_skips_cuda_check(self, valid_env):
        """_load_model should not check CUDA when device is 'cpu'."""
        transcriber = Transcriber()

        # Override config to use CPU
        import local_feed_generator.config as config_module

        config_module._config_instance = None
        os.environ["TRANSCRIPTION_DEVICE"] = "cpu"
        config_module.reload_config()

        try:
            with (
                mock.patch(
                    "local_feed_generator.services.transcriber.check_cuda_available"
                ) as mock_check,
                mock.patch(
                    "local_feed_generator.services.transcriber.WhisperModel"
                ) as mock_whisper_cls,
            ):
                mock_model = MockWhisperModel(segments_data=["Hello"])
                mock_whisper_cls.return_value = mock_model
                transcriber._load_model()
                # check_cuda_available should not be called for CPU
                assert mock_check.call_count == 0
        finally:
            del os.environ["TRANSCRIPTION_DEVICE"]
            config_module._config_instance = None

    def test_faster_whisper_cublas_error_wrapped(self, valid_env):
        """Errors mentioning libcublas from faster-whisper are wrapped."""
        transcriber = Transcriber()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=True,
            ),
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            pytest.raises(TranscriptionError) as exc_info,
        ):
            mock_whisper_cls.side_effect = RuntimeError(
                "Error loading libcublas.so.12: cannot open shared object file"
            )
            transcriber._load_model()

            assert "CUDA libraries not found" in str(exc_info.value)

    def test_faster_whisper_cublas_error_wrapped_case_insensitive(self, valid_env):
        """Errors mentioning 'cublas' (any case) from faster-whisper are wrapped."""
        transcriber = Transcriber()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=True,
            ),
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            pytest.raises(TranscriptionError) as exc_info,
        ):
            mock_whisper_cls.side_effect = RuntimeError("CUDA error: cuBLAS initialization failed")
            transcriber._load_model()

            assert "CUDA libraries not found" in str(exc_info.value)

    def test_non_cuda_error_uses_original_message(self, valid_env):
        """Non-CUDA errors should use the original error message."""
        transcriber = Transcriber()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=True,
            ),
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            pytest.raises(TranscriptionError) as exc_info,
        ):
            mock_whisper_cls.side_effect = RuntimeError("Invalid model name")
            transcriber._load_model()

            assert "Failed to load transcription model" in str(exc_info.value)
            assert "Invalid model name" in str(exc_info.value)

    def test_cuda_check_error_propagates_through_transcribe(self, valid_env):
        """The CUDA error should propagate when calling transcribe()."""
        transcriber = Transcriber()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=False,
            ),
            mock.patch.object(Path, "exists", return_value=True),
            pytest.raises(TranscriptionError) as exc_info,
        ):
            transcriber.transcribe("fake.mp4")

            assert "CUDA libraries not found" in str(exc_info.value)

    def test_cublas_error_during_transcription_wrapped_with_cuda_message(self, valid_env):
        """cuBLAS errors during transcription should be wrapped with CUDA_ERROR_MESSAGE."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel()
        # Override transcribe to raise a cuBLAS error
        mock_model.transcribe = mock.Mock(
            side_effect=OSError("Library libcublas.so.12 is not found or cannot be loaded")
        )

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                with pytest.raises(TranscriptionError) as exc_info:
                    transcriber.transcribe("fake.mp4")

                # Should get the helpful CUDA error message, not the raw error
                assert "CUDA libraries not found" in str(exc_info.value)
                assert exc_info.value.filepath == "fake.mp4"

    def test_non_cuda_error_during_transcription_uses_original_message(self, valid_env):
        """Non-CUDA errors during transcription should use the original error message."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel()
        mock_model.transcribe = mock.Mock(side_effect=RuntimeError("Some other error"))

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                with pytest.raises(TranscriptionError) as exc_info:
                    transcriber.transcribe("fake.mp4")

                assert "Transcription failed" in str(exc_info.value)
                assert "Some other error" in str(exc_info.value)
                assert "CUDA libraries" not in str(exc_info.value)
