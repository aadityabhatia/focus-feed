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


class TestProcessCommand:
    """Tests for the process command (T5.2)."""

    def test_process_no_files_to_process(self, runner, valid_env, tmp_data_dir):
        """Process should report no files when directory is empty."""
        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            result = runner.invoke(
                app,
                ["process", str(tmp_data_dir)],
            )
        assert result.exit_code == 0
        assert "No files to process." in result.output

    def test_process_finds_files_to_process(self, runner, valid_env, tmp_data_dir):
        """Process should find and report files to process."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        assert "Found 1 file(s) to process." in result.output
        assert "[1/1] Processing" in result.output
        mock_producer.process_file.assert_called_once()

    def test_process_generates_feeds_after_processing(self, runner, valid_env, tmp_data_dir):
        """Process should generate feeds after all files are processed."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        assert "Feeds generated" in result.output
        mock_gen.assert_called_once()

    def test_process_exit_code_0_on_success(self, runner, valid_env, tmp_data_dir):
        """Process should exit with code 0 when all files succeed."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 1}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        assert "1 succeeded" in result.output

    def test_process_exit_code_1_on_errors(self, runner, valid_env, tmp_data_dir):
        """Process should exit with code 1 when any file fails."""
        (tmp_data_dir / "bad.mp4").write_bytes(b"bad data")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=False)
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 1
        assert "1 failed" in result.output

    def test_process_continues_on_error(self, runner, valid_env, tmp_data_dir):
        """Process should continue processing after an error."""
        # Create two files; Scanner sorts by filepath so "bad.mp4" comes first
        (tmp_data_dir / "bad.mp4").write_bytes(b"video data")
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 1}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                # First call (bad.mp4) returns error, second (doc.pdf) returns success
                mock_producer.process_file.side_effect = [
                    _make_mock_metadata(feed_eligible=False),
                    _make_mock_metadata(feed_eligible=True),
                ]
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 1
        assert "1 succeeded" in result.output
        assert "1 failed" in result.output

    def test_process_skips_files_with_existing_json(self, runner, valid_env, tmp_data_dir):
        """Process should only process files without JSON metadata."""
        (tmp_data_dir / "processed.mp4").write_bytes(b"video")
        (tmp_data_dir / "processed.mp4.json").write_text(json.dumps(_make_valid_metadata()))
        (tmp_data_dir / "unprocessed.pdf").write_bytes(b"pdf")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 1}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        assert "Found 1 file(s) to process." in result.output
        # Only unprocessed.pdf should be processed
        call_arg = mock_producer.process_file.call_args[0][0]
        assert call_arg.filename == "unprocessed.pdf"

    def test_process_with_config_file(self, runner, valid_env, tmp_data_dir, tmp_path):
        """Process should accept --config-file option."""
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

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            result = runner.invoke(
                app,
                [
                    "process",
                    str(tmp_data_dir),
                    "--config-file",
                    str(config_file),
                ],
            )

        assert result.exit_code == 0

    def test_process_shows_progress(self, runner, valid_env, tmp_data_dir):
        """Process should show progress for each file."""
        (tmp_data_dir / "file1.mp4").write_bytes(b"video1")
        (tmp_data_dir / "file2.pdf").write_bytes(b"pdf2")

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("focus_feed.producer.Producer") as MockProducer:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
                MockProducer.return_value = mock_producer

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        assert "[1/2]" in result.output
        assert "[2/2]" in result.output
