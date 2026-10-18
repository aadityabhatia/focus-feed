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
    monkeypatch.setenv("LLM_ENDPOINT", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("PROMPT_ARTICLE", "Test article prompt")
    monkeypatch.setenv("PROMPT_ARTICLE_REFLECTION", "Test article reflection prompt")
    monkeypatch.setenv("PROMPT_PODCAST", "Test podcast prompt")
    monkeypatch.setenv("PROMPT_PODCAST_REFLECTION", "Test podcast reflection prompt")
    monkeypatch.setenv("PROMPT_TITLE", "Test title prompt")
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


def _make_av_mock_with_streams(streams_data):
    """Create a mock av module with the given audio streams."""
    mock_streams = []
    for sd in streams_data:
        s = mock.MagicMock()
        if "duration_seconds" in sd:
            s.duration_seconds = sd["duration_seconds"]
        mock_streams.append(s)

    mock_container = mock.MagicMock()
    mock_container.streams.audio = mock_streams

    mock_av_module = mock.MagicMock()
    mock_av_module.open.return_value.__enter__.return_value = mock_container
    return mock_av_module


def _make_av_mock_with_error(error):
    """Create a mock av module that raises on open."""
    mock_av_module = mock.MagicMock()
    mock_av_module.open.side_effect = error
    return mock_av_module


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

    @pytest.mark.parametrize("filepath", ["nonexistent.mp4", "/path/to/nonexistent.mp4"])
    def test_transcribe_nonexistent_file_raises_error(self, valid_env, filepath):
        """transcribe() should raise TranscriptionError for non-existent files."""
        transcriber = Transcriber()

        with mock.patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcriber.transcribe(filepath)

            assert "File not found" in str(exc_info.value)
            assert filepath in str(exc_info.value)
            assert exc_info.value.filepath == filepath


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

    @pytest.mark.parametrize(
        "segments_data,expected_lines",
        [
            (
                ["First sentence.", "Second sentence.", "Third sentence."],
                ["First sentence.", "Second sentence.", "Third sentence."],
            ),
            (["Real text", "", "   ", "More text"], ["Real text", "More text"]),
            (["Only one segment."], ["Only one segment."]),
            ([], []),
            (["", "   ", ""], []),
            (["  Padded text  ", "  More text  "], ["Padded text", "More text"]),
        ],
    )
    def test_transcribe_output_formatting(self, valid_env, segments_data, expected_lines):
        """transcribe() should join segments with newlines, strip whitespace.

        Empty segments are skipped to ensure clean output.
        """
        transcriber = Transcriber()

        mock_model = MockWhisperModel(segments_data=segments_data)

        with mock.patch(
            "local_feed_generator.services.transcriber.WhisperModel"
        ) as mock_whisper_cls:
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                result = transcriber.transcribe("fake.mp4")

        lines = result.split("\n") if result else []
        assert lines == expected_lines


# ---------------------------------------------------------------------------
# Tests: Skip already complete
# ---------------------------------------------------------------------------


class TestTranscribeSkipAlreadyComplete:
    """Tests for skipping transcription when JSON marks it as complete."""

    _MPATH = "local_feed_generator.services.transcriber"

    @pytest.mark.parametrize(
        "json_content,should_skip,expected_result",
        [
            (
                '{"status": {"transcribe": "in_progress", "summarize": null}, "transcript": ""}',
                False,
                "New transcript",
            ),
            (
                '{"status": {"transcribe": "complete", "summarize": null},'
                ' "transcript": "Existing"}',
                True,
                "Existing",
            ),
            (
                '{"status": {"transcribe": "complete", "summarize": null}, "transcript": ""}',
                False,
                "New transcript",
            ),
        ],
    )
    def test_transcribe_skip_or_proceed(
        self, valid_env, tmp_path, json_content, should_skip, expected_result
    ):
        """transcribe() should skip or proceed based on JSON status and transcript availability."""
        audio_file = tmp_path / "lecture.mp4"
        audio_file.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(json_content)

        transcriber = Transcriber()
        mock_model = MockWhisperModel(segments_data=[expected_result])

        with (
            mock.patch(f"{self._MPATH}.WhisperModel") as mock_whisper_cls,
            mock.patch(f"{self._MPATH}.Transcriber._get_audio_duration", return_value=60.0),
        ):
            mock_whisper_cls.return_value = mock_model
            result = transcriber.transcribe(str(audio_file))

        assert result == expected_result
        if should_skip:
            assert mock_whisper_cls.call_count == 0
        else:
            assert mock_whisper_cls.call_count == 1


class TestGetAudioDuration:
    """Tests for the _get_audio_duration helper method."""

    @pytest.mark.parametrize(
        "setup_mock,expected",
        [
            # Duration from av container
            (lambda: _make_av_mock_with_streams([{"duration_seconds": 42.5}]), 42.5),
            # No audio streams
            (lambda: _make_av_mock_with_streams([]), None),
            # av.open raises
            (lambda: _make_av_mock_with_error(RuntimeError("av error")), None),
            # av not available
            (lambda: None, None),
        ],
    )
    def test_get_audio_duration_scenarios(self, valid_env, tmp_path, setup_mock, expected):
        """_get_audio_duration should handle various scenarios."""
        audio_file = tmp_path / "test.mp4"
        audio_file.write_bytes(b"fake audio data")

        if setup_mock is None:
            with mock.patch.dict("sys.modules", {"av": None}):
                result = Transcriber._get_audio_duration(audio_file)
        else:
            av_module = setup_mock()
            with mock.patch.dict("sys.modules", {"av": av_module}):
                result = Transcriber._get_audio_duration(audio_file)

        assert result == expected


class TestGetFileSize:
    """Tests for the _get_file_size helper method."""

    @pytest.mark.parametrize(
        "file_exists,expected",
        [
            (True, 1024),
            (False, None),  # Nonexistent file returns None
        ],
    )
    def test_get_file_size(self, valid_env, tmp_path, file_exists, expected):
        """_get_file_size should return the file size in bytes, or None on error."""
        audio_file = tmp_path / "test.mp4"
        if file_exists:
            audio_file.write_bytes(b"x" * 1024)
        result = Transcriber._get_file_size(audio_file)
        assert result == expected


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
        """cleanup() should be safe to call when no model is loaded or model has no delete."""
        transcriber = Transcriber()

        # Should not raise when no model is loaded
        transcriber.cleanup()
        assert transcriber.model is None

        # Should not raise when model has no delete method
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


# ---------------------------------------------------------------------------
# Tests: CUDA availability check in _load_model()
# ---------------------------------------------------------------------------


class TestCudaCheckInLoadModel:
    """Tests for the CUDA availability check integrated into _load_model()."""

    @pytest.mark.parametrize(
        "cuda_available,whisper_side_effect,expected_error",
        [
            (False, None, "CUDA libraries not found"),
            (
                True,
                RuntimeError("Error loading libcublas.so.12: cannot open shared object file"),
                "CUDA libraries not found",
            ),
            (
                True,
                RuntimeError("CUDA error: cuBLAS initialization failed"),
                "CUDA libraries not found",
            ),
            (
                True,
                RuntimeError("Invalid model name"),
                "Failed to load transcription model",
            ),
        ],
    )
    def test_cuda_error_scenarios(
        self, valid_env, cuda_available, whisper_side_effect, expected_error
    ):
        """_load_model should handle various CUDA and model loading error scenarios."""
        transcriber = Transcriber()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.check_cuda_available",
                return_value=cuda_available,
            ),
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            pytest.raises(TranscriptionError) as exc_info,
        ):
            if whisper_side_effect:
                mock_whisper_cls.side_effect = whisper_side_effect
            transcriber._load_model()

            assert expected_error in str(exc_info.value)
            if not cuda_available:
                assert mock_whisper_cls.call_count == 0
            if "Invalid model name" in str(whisper_side_effect or ""):
                assert "Invalid model name" in str(exc_info.value)

    @pytest.mark.parametrize(
        "device,cuda_check_called,should_succeed",
        [
            ("cuda", True, True),  # CUDA device: check is called, should succeed
            ("cpu", False, True),  # CPU device: check is skipped, should succeed
        ],
    )
    def test_device_handling(self, valid_env, device, cuda_check_called, should_succeed):
        """_load_model should check CUDA only when device is 'cuda'."""
        import local_feed_generator.config as config_module

        config_module._config_instance = None
        os.environ["TRANSCRIPTION_DEVICE"] = device
        config_module.reload_config()

        try:
            transcriber = Transcriber()
            mock_model = MockWhisperModel(segments_data=["Hello"])

            with (
                mock.patch(
                    "local_feed_generator.services.transcriber.check_cuda_available",
                    return_value=True,
                ) as mock_check,
                mock.patch(
                    "local_feed_generator.services.transcriber.WhisperModel"
                ) as mock_whisper_cls,
            ):
                mock_whisper_cls.return_value = mock_model
                result = transcriber._load_model()
                assert mock_check.call_count == (1 if cuda_check_called else 0)
                if should_succeed:
                    assert mock_whisper_cls.call_count == 1
                    assert result is mock_model
        finally:
            del os.environ["TRANSCRIPTION_DEVICE"]
            config_module._config_instance = None

    def test_cublas_error_during_transcription_wrapped_with_cuda_message(self, valid_env):
        """cuBLAS errors during transcription should be wrapped with CUDA_ERROR_MESSAGE."""
        transcriber = Transcriber()

        mock_model = MockWhisperModel()

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            mock.patch.object(
                mock_model,
                "transcribe",
                side_effect=OSError("Library libcublas.so.12 is not found or cannot be loaded"),
            ),
        ):
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

        with (
            mock.patch(
                "local_feed_generator.services.transcriber.WhisperModel"
            ) as mock_whisper_cls,
            mock.patch.object(
                mock_model, "transcribe", side_effect=RuntimeError("Some other error")
            ),
        ):
            mock_whisper_cls.return_value = mock_model

            with mock.patch.object(Path, "exists", return_value=True):
                with pytest.raises(TranscriptionError) as exc_info:
                    transcriber.transcribe("fake.mp4")

                assert "Transcription failed" in str(exc_info.value)
                assert "Some other error" in str(exc_info.value)
                assert "CUDA libraries" not in str(exc_info.value)
