"""Tests for the CLI module (Phase 5).

Tests cover:
- CLI framework: group, shared options, config loading, help text
- Process command: orchestration, progress, summary, exit codes
- Watch command: observer startup, event handling, graceful shutdown, error handling
- Generate-feeds command: skip processing, read JSON, generate feeds, summary
- Validate command: schema validation, orphaned JSON, special characters, exit codes
- Entry point: main() function, pyproject.toml configuration
- Positional args: process only specific files/directories
- --force flag: force processing even if complete
- Logging level: --log-level option, LOG_LEVEL env var, _configure_logging_level
"""

import json
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
import pytest

from focus_feed.cli import app
from focus_feed.metadata import Metadata
from focus_feed.scanner import ScannedFile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_ENV_KEYS = [
    "BASE_URL",
    "LLM_ENDPOINT",
    "LLM_API_KEY",
    "LLM_MODEL",
    "PROMPT_ARTICLE",
    "PROMPT_ARTICLE_REFLECTION",
    "PROMPT_PODCAST",
    "PROMPT_PODCAST_REFLECTION",
    "PROMPT_TITLE",
    "TRANSCRIPTION_ENDPOINT",
    "TRANSCRIPTION_API_KEY",
]


@pytest.fixture
def clear_required_env(monkeypatch):
    """Clear all required env vars, yielding control, then they're auto-restored by monkeypatch."""
    for key in REQUIRED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def _restore_env(reset_config):
    """Apply the shared config reset to every CLI test."""


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with test files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def watch_mocks():
    """Provide a context manager that sets up common watch command mocks.

    Reduces nested ``with mock.patch`` blocks from 4-5 levels to 1.
    """
    from contextlib import contextmanager

    @contextmanager
    def _setup(
        gen_return=None,
        producer_return=None,
        producer_process_return=None,
    ):
        if gen_return is None:
            gen_return = {"podcast": 0, "articles": 0}
        with (
            mock.patch("focus_feed.cli._generate_feeds") as mock_gen,
            mock.patch("focus_feed.producer.Producer") as MockProducer,
            mock.patch("watchdog.observers.Observer") as MockObserver,
            mock.patch("focus_feed.cli.time.sleep") as mock_sleep,
        ):
            mock_gen.return_value = gen_return
            if producer_process_return is not None:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = producer_process_return
                MockProducer.return_value = mock_producer
            elif producer_return is not None:
                MockProducer.return_value = producer_return
            else:
                MockProducer.return_value = mock.MagicMock()
            mock_observer = mock.MagicMock()
            MockObserver.return_value = mock_observer
            mock_sleep.side_effect = KeyboardInterrupt
            yield {
                "mock_gen": mock_gen,
                "MockProducer": MockProducer,
                "MockObserver": MockObserver,
                "mock_observer": mock_observer,
                "mock_sleep": mock_sleep,
            }

    return _setup


def _make_scanned_file(
    filepath: str,
    absolute_path: str | None = None,
    filename: str | None = None,
    filetype: str | None = None,
) -> ScannedFile:
    """Helper to create a ScannedFile for testing."""
    if absolute_path is None:
        absolute_path = f"/data/{filepath}"
    if filename is None:
        filename = Path(filepath).name
    if filetype is None:
        filetype = Path(filepath).suffix.lstrip(".")
    return ScannedFile(
        filepath=filepath,
        absolute_path=absolute_path,
        filename=filename,
        filetype=filetype,
    )


def _make_valid_metadata(
    filepath: str = "videos/lecture.mp4",
    filename: str = "lecture.mp4",
    filetype: str = "mp4",
    error: str | None = None,
) -> dict:
    """Create a valid metadata dictionary for testing.

    Creates metadata with all applicable step maps set to "complete".
    If ``error`` is provided, the error field is set and step maps are
    not included (making the file not feed-eligible).
    """
    meta: dict = {
        "title": "Test Title",
        "description": "Test description",
        "filename": filename,
        "filetype": filetype,
        "date": 1728931200,
        "size": 12345,
    }
    if error:
        meta["error"] = error
    else:
        if filetype == "mp4":
            meta["transcribe"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
            meta["summarize"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
            meta["content"] = "Test transcript"
            meta["summary"] = "Test summary"
        else:
            meta["extract"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
            meta["summarize"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
    return meta


def _make_mock_metadata(feed_eligible: bool = True):
    """Create a mock Metadata object for testing."""
    mock_obj = mock.MagicMock()
    mock_obj.feed_eligible = feed_eligible
    return mock_obj


# ---------------------------------------------------------------------------
# T5.1: CLI Framework Tests
# ---------------------------------------------------------------------------


class TestTranscribeCommand:
    """Tests for the transcribe command (T5.7)."""

    def test_transcribe_help(self, runner):
        """The transcribe command should have help text."""
        result = runner.invoke(app, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--config-file" in result.output
        assert "--force" in result.output

    def test_transcribe_no_files(self, runner, valid_env, tmp_data_dir):
        """Transcribe should report no files when directory is empty."""
        result = runner.invoke(
            app,
            ["transcribe", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No audio/video files need transcription." in result.output

    def test_transcribe_dry_run(self, runner, valid_env, tmp_data_dir):
        """Transcribe --dry-run should list files without processing."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir), "--dry-run"],
            )

        assert result.exit_code == 0
        assert "[Dry run]" in result.output
        assert "lecture.mp4" in result.output
        # Transcriber should NOT have been instantiated in dry-run mode
        mock_transcriber.transcribe.assert_not_called()

    def test_transcribe_skips_files_with_transcript(self, runner, valid_env, tmp_data_dir):
        """Transcribe should skip MP4 files that already have transcribe status='complete'."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        # Create JSON with existing transcript and transcribe step complete
        (tmp_data_dir / "lecture.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Existing transcript",
                }
            )
        )

        result = runner.invoke(
            app,
            ["transcribe", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No audio/video files need transcription." in result.output

    def test_transcribe_processes_files(self, runner, valid_env, tmp_data_dir):
        """Transcribe should process MP4 files without transcripts."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "This is a transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 audio/video file(s) to transcribe." in result.output
        mock_transcriber.transcribe.assert_called_once()

    def test_transcribe_writes_transcript_to_json(self, runner, valid_env, tmp_data_dir):
        """Transcribe should write the transcript to the JSON metadata file."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "This is a transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "lecture.mp4.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["content"] == "This is a transcript."
        assert data["transcribe"]["status"] == "complete"

    def test_transcribe_persists_actual_media_filetype(self, runner, valid_env, tmp_data_dir):
        """Transcribing an audio file should preserve its actual filetype."""
        (tmp_data_dir / "lecture.m4a").write_bytes(b"audio data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "This is an audio transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(app, ["transcribe", str(tmp_data_dir)])

        assert result.exit_code == 0
        data = json.loads((tmp_data_dir / "lecture.m4a.json").read_text())
        assert data["filetype"] == "m4a"

    def test_transcribe_uses_positional_arg_for_directory(self, runner, valid_env, tmp_data_dir):
        """Transcribe should use the directory passed as a positional argument."""
        # Create a file in tmp_data_dir and another in a sibling directory
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        other_dir = tmp_data_dir.parent / "other_data"
        other_dir.mkdir()
        (other_dir / "other.mp4").write_bytes(b"other video")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "Transcript"
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(app, ["transcribe", str(tmp_data_dir)])

        assert result.exit_code == 0
        # Only the file in tmp_data_dir should be processed, not the one in other_dir
        mock_transcriber.transcribe.assert_called_once()

    def test_transcribe_exit_code_1_on_error(self, runner, valid_env, tmp_data_dir):
        """Transcribe should exit with code 1 when transcription fails."""
        (tmp_data_dir / "bad.mp4").write_bytes(b"bad data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.side_effect = RuntimeError("Transcription failed")
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 1
        assert "1 failed" in result.output

    def test_transcribe_failure_persists_step_error(self, runner, valid_env, tmp_data_dir):
        """Transcription failures should be saved in the step map and remain valid metadata."""
        source = tmp_data_dir / "bad.mp4"
        source.write_bytes(b"bad data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            MockTranscriber.return_value.transcribe.side_effect = RuntimeError(
                "Transcription failed"
            )
            result = runner.invoke(app, ["transcribe", str(tmp_data_dir)])

        assert result.exit_code == 1
        data = json.loads((tmp_data_dir / "bad.mp4.json").read_text())
        assert data["transcribe"]["error"] == "Transcription failed"
        assert data["transcribe"]["status"] is None
        metadata = Metadata.from_json_file(tmp_data_dir / "bad.mp4.json")
        assert metadata.transcribe is not None
        assert metadata.transcribe.error == "Transcription failed"
        assert metadata.transcribe.status is None

    def test_transcribe_with_config_file(self, runner, valid_env, tmp_data_dir, tmp_path):
        """Transcribe should accept --config-file option."""
        config_file = tmp_path / "custom.env"
        config_file.write_text(
            "BASE_URL=https://custom.example.com\n"
            "LLM_ENDPOINT=https://llm.example.com/v1\n"
            "LLM_API_KEY=test-key\n"
            "LLM_MODEL=test-model\n"
            "PROMPT_ARTICLE=Test article prompt\n"
            "PROMPT_ARTICLE_REFLECTION=Test article reflection prompt\n"
            "PROMPT_PODCAST=Test podcast prompt\n"
            "PROMPT_PODCAST_REFLECTION=Test podcast reflection prompt\n"
            "PROMPT_TITLE=Test title prompt\n"
        )

        result = runner.invoke(
            app,
            [
                "transcribe",
                str(tmp_data_dir),
                "--config-file",
                str(config_file),
            ],
        )
        assert result.exit_code == 0

    def test_transcribe_force_flag_reprocesses_complete_files(
        self, runner, valid_env, tmp_data_dir
    ):
        """Transcribe --force should re-transcribe files with transcribe status='complete'."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        # Create JSON with transcribe step complete
        (tmp_data_dir / "lecture.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Existing transcript",
                }
            )
        )

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "New transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir), "--force"],
            )

        assert result.exit_code == 0
        assert "Found 1 audio/video file(s) to transcribe." in result.output
        mock_transcriber.transcribe.assert_called_once()

    def test_transcribe_file_flag_processes_specific_file(self, runner, valid_env, tmp_data_dir):
        """Transcribe should process a specific file passed as positional arg."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        (tmp_data_dir / "other.mp4").write_bytes(b"video data")
        # other.mp4 already has a complete transcript, so only lecture.mp4 needs transcription
        (tmp_data_dir / "other.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "other.mp4",
                    "filepath": "other.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Existing transcript",
                }
            )
        )

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "Transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir / "lecture.mp4")],
            )

        assert result.exit_code == 0
        # Only lecture.mp4 should be transcribed
        mock_transcriber.transcribe.assert_called_once_with(str(tmp_data_dir / "lecture.mp4"))

    def test_transcribe_sets_in_progress_before_and_complete_after(
        self, runner, valid_env, tmp_data_dir
    ):
        """Transcribe should set status to 'in_progress' before transcription, then 'complete'."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "This is a transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 0

        # Verify final JSON state has complete status and end_time
        json_path = tmp_data_dir / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcribe"]["status"] == "complete"
        assert data["content"] == "This is a transcript."
        assert data["transcribe"]["end_time"] is not None


class TestFindFilesNeedingTranscription:
    """Tests for the _find_files_needing_transcription helper function."""

    def test_find_files_needing_transcription_finds_mp4_without_json(self, valid_env, tmp_data_dir):
        """Should find MP4 files with no JSON metadata."""
        (tmp_data_dir / "video.mp4").write_bytes(b"video")

        from focus_feed.cli import _find_files_needing_transcription

        result = _find_files_needing_transcription(tmp_data_dir)
        assert len(result) == 1
        assert result[0].name == "video.mp4"

    def test_find_files_needing_transcription_finds_audio_without_json(
        self, valid_env, tmp_data_dir
    ):
        """Should find supported audio files with no JSON metadata."""
        (tmp_data_dir / "lecture.m4a").write_bytes(b"audio")

        from focus_feed.cli import _find_files_needing_transcription

        result = _find_files_needing_transcription(tmp_data_dir)
        assert len(result) == 1
        assert result[0].name == "lecture.m4a"

    def test_find_files_needing_transcription_skips_complete(self, valid_env, tmp_data_dir):
        """Should skip MP4 files with complete transcription."""
        (tmp_data_dir / "video.mp4").write_bytes(b"video")
        (tmp_data_dir / "video.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete"},
                    "filename": "video.mp4",
                    "filetype": "mp4",
                    "content": "Transcript",
                }
            )
        )

        from focus_feed.cli import _find_files_needing_transcription

        result = _find_files_needing_transcription(tmp_data_dir)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "filter_type",
        ["file", "directory"],
    )
    def test_find_files_needing_transcription_paths_filters(
        self, valid_env, tmp_data_dir, filter_type
    ):
        """Should filter results to specific file or directory."""
        (tmp_data_dir / "video1.mp4").write_bytes(b"video")
        (tmp_data_dir / "video2.mp4").write_bytes(b"video")

        from focus_feed.cli import _find_files_needing_transcription

        # Without paths filter: both files returned
        result = _find_files_needing_transcription(tmp_data_dir)
        assert len(result) == 2

        if filter_type == "file":
            # With paths filter for only video1.mp4: only video1.mp4 returned
            result = _find_files_needing_transcription(
                tmp_data_dir, paths=[tmp_data_dir / "video1.mp4"]
            )
            assert len(result) == 1
            assert result[0].name == "video1.mp4"
        else:
            # With paths filter for subdir: only video1.mp4 returned
            subdir = tmp_data_dir / "subdir"
            subdir.mkdir()
            (subdir / "video1.mp4").write_bytes(b"video")
            result = _find_files_needing_transcription(tmp_data_dir, paths=[subdir])
            assert len(result) == 1
            assert result[0].name == "video1.mp4"
