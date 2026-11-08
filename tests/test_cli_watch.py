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
import time
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


class TestWatchCommand:
    """Tests for the watch command (T5.3)."""

    def test_watch_starts_observer(self, runner, valid_env, tmp_data_dir, watch_mocks):
        """Watch should start a watchdog Observer."""
        with watch_mocks(producer_process_return=_make_mock_metadata(feed_eligible=True)) as mocks:
            result = runner.invoke(
                app,
                ["watch", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        mocks["MockObserver"].assert_called_once()
        mocks["mock_observer"].start.assert_called_once()
        mocks["mock_observer"].stop.assert_called_once()
        mocks["mock_observer"].join.assert_called_once()

    def test_watch_initial_scan_processes_files(self, runner, valid_env, tmp_data_dir, watch_mocks):
        """Watch should process files on initial scan."""
        (tmp_data_dir / "new.mp4").write_bytes(b"video")

        with watch_mocks(
            gen_return={"podcast": 1, "articles": 0},
            producer_process_return=_make_mock_metadata(feed_eligible=True),
        ) as mocks:
            result = runner.invoke(
                app,
                ["watch", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Initial scan: found 1 file(s) to process." in result.output
        mocks["MockProducer"].return_value.process_file.assert_called_once()

    def test_watch_no_files_initial_scan(self, runner, valid_env, tmp_data_dir, watch_mocks):
        """Watch should handle empty directory on initial scan."""
        with watch_mocks():
            result = runner.invoke(
                app,
                ["watch", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Initial scan: no files to process." in result.output

    def test_watch_nonexistent_directory_exits_1(self, runner, valid_env, tmp_path):
        """Watch should exit with code 1 for nonexistent directory."""
        result = runner.invoke(
            app,
            ["watch", str(tmp_path / "nonexistent")],
        )
        assert result.exit_code != 0

    def test_watch_handler_processes_source_file(self, valid_env, tmp_data_dir):
        """Watch handler should process source files on event."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "new.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            handler._handle_event(str(source_file))

        producer.process_file.assert_called_once()

    def test_watch_handler_ignores_unsupported_files(self, valid_env, tmp_data_dir):
        """Watch handler should ignore non-source files."""
        from focus_feed.cli import _WatchHandler

        other_file = tmp_data_dir / "readme.txt"
        other_file.write_text("text")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        handler._handle_event(str(other_file))

        producer.process_file.assert_not_called()

    @pytest.mark.parametrize(
        "filename",
        [".hidden.mp4", "video.mp4.tmp", "index.html", "INDEX.HTML", "feed.json"],
    )
    def test_watch_handler_ignores_hidden_temporary_and_generated_files(
        self, valid_env, tmp_data_dir, filename
    ):
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / filename
        source_file.write_bytes(b"data")
        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            handler._handle_event(str(source_file))

        producer.process_file.assert_not_called()
        mock_gen.assert_not_called()

    def test_watch_json_handler_ignores_hidden_and_temporary_files(self, valid_env, tmp_data_dir):
        from focus_feed.cli import _WatchHandler

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)
        with mock.patch("focus_feed.cli._logger") as mock_logger:
            handler._handle_json_event(tmp_data_dir / ".hidden.json")
            handler._handle_json_event(tmp_data_dir / "video.json.tmp")

        mock_logger.warning.assert_not_called()

    def test_cli_file_discovery_ignores_hidden_temporary_and_generated_files(
        self, valid_env, tmp_data_dir
    ):
        from focus_feed.cli import (
            _expand_paths_to_files,
            _find_json_files,
            _find_source_files,
        )

        for filename in (".hidden.mp4", "video.mp4.tmp", "index.html", "lecture.mp4"):
            (tmp_data_dir / filename).write_bytes(b"data")
        (tmp_data_dir / ".hidden.json").write_text("{}")
        (tmp_data_dir / "metadata.json").write_text("{}")

        expanded = _expand_paths_to_files([tmp_data_dir])
        sources = _find_source_files(tmp_data_dir, ("mp4", "html"))
        json_files = _find_json_files(tmp_data_dir)

        assert {file.filename for file in expanded} == {"lecture.mp4"}
        assert {file.name for file in sources} == {"lecture.mp4"}
        assert {file.name for file in json_files} == {"metadata.json"}

    def test_watch_handler_logs_warning_for_orphaned_json(self, valid_env, tmp_data_dir):
        """Watch handler should log warning for orphaned JSON files."""
        from focus_feed.cli import _WatchHandler

        # Create a JSON file without a source file
        json_file = tmp_data_dir / "orphan.mp4.json"
        json_file.write_text("{}")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._logger") as mock_logger:
            handler._handle_json_event(json_file)

        mock_logger.warning.assert_called_once()
        assert "Orphaned JSON" in mock_logger.warning.call_args[0][0]

    def test_watch_handler_json_with_source_no_warning(self, valid_env, tmp_data_dir):
        """Watch handler should not warn for JSON files with existing source."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "doc.pdf"
        source_file.write_bytes(b"pdf")
        json_file = tmp_data_dir / "doc.pdf.json"
        json_file.write_text("{}")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._logger") as mock_logger:
            handler._handle_json_event(json_file)

        mock_logger.warning.assert_not_called()

    def test_watch_handler_feed_json_no_orphaned_warning(self, valid_env, tmp_data_dir):
        """Watch handler should not warn for feed.json (configuration file)."""
        from focus_feed.cli import _WatchHandler

        json_file = tmp_data_dir / "feed.json"
        json_file.write_text('{"title": "My Feed"}')

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._logger") as mock_logger:
            handler._handle_json_event(json_file)

        mock_logger.warning.assert_not_called()

    def test_watch_graceful_shutdown(self, runner, valid_env, tmp_data_dir, watch_mocks):
        """Watch should shut down gracefully on SIGINT."""
        with watch_mocks() as mocks:
            result = runner.invoke(
                app,
                ["watch", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        mocks["mock_observer"].stop.assert_called_once()
        mocks["mock_observer"].join.assert_called_once()

    def test_watch_handler_continues_after_error(self, valid_env, tmp_data_dir):
        """Watch handler should not crash when file processing raises an exception."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # _handle_event should not raise; it should catch and log the error
        with mock.patch("focus_feed.cli.log_error") as mock_log_error:
            handler._handle_event(str(source_file))

        # Producer was called despite the error (the event was dispatched)
        producer.process_file.assert_called_once()
        # Error should have been logged
        mock_log_error.assert_called_once()
        call_kwargs = mock_log_error.call_args
        assert "bad.mp4" in call_kwargs[1]["filepath"]
        assert call_kwargs[1]["stage"] == "processing"

    def test_watch_handler_logs_error_on_processing_failure(self, valid_env, tmp_data_dir):
        """Watch handler should log an error when file processing fails."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.log_error") as mock_log_error:
            handler._handle_event(str(source_file))

        mock_log_error.assert_called_once()
        # Verify the filepath was passed
        call_kwargs = mock_log_error.call_args
        assert "bad.mp4" in call_kwargs[1]["filepath"]
        assert call_kwargs[1]["stage"] == "processing"

    def test_watch_handler_error_rate_limiting(self, valid_env, tmp_data_dir):
        """Watch handler should suppress detailed errors after rate limit."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # Pre-fill the error timestamps to the threshold
        handler._error_timestamps = [time.monotonic()] * handler._max_errors_per_minute

        with (
            mock.patch("focus_feed.cli.log_error") as mock_log_error,
            mock.patch("focus_feed.cli._logger") as mock_logger,
        ):
            handler._handle_event(str(source_file))

        # log_error should NOT have been called (rate limited)
        mock_log_error.assert_not_called()
        # A warning summary should have been logged instead
        mock_logger.warning.assert_called_once()
        assert "rate limit" in mock_logger.warning.call_args[0][0].lower()

    def test_watch_handler_rate_limiting_resets_after_window(self, valid_env, tmp_data_dir):
        """Watch handler should resume detailed logging after the window expires."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # Pre-fill with old timestamps (outside the 60-second window)
        handler._error_timestamps = [time.monotonic() - 61] * handler._max_errors_per_minute

        with mock.patch("focus_feed.cli.log_error") as mock_log_error:
            handler._handle_event(str(source_file))

        # log_error SHOULD have been called because old timestamps were trimmed
        mock_log_error.assert_called_once()

    def test_watch_handler_error_timestamps_trimmed(self, valid_env, tmp_data_dir):
        """Watch handler should trim old error timestamps on each error."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # Add a mix of recent and old timestamps
        handler._error_timestamps = [
            time.monotonic() - 61,  # old, should be trimmed
            time.monotonic() - 30,  # recent, should be kept
        ]

        with mock.patch("focus_feed.cli.log_error"):
            handler._handle_event(str(source_file))

        # The old timestamp should have been trimmed, leaving only 2 entries
        assert len(handler._error_timestamps) == 2

    def test_watch_handler_continues_after_multiple_errors(self, valid_env, tmp_data_dir):
        """Watch handler should continue processing after multiple errors."""
        from focus_feed.cli import _WatchHandler

        source_file1 = tmp_data_dir / "bad1.mp4"
        source_file1.write_bytes(b"video")
        source_file2 = tmp_data_dir / "bad2.mp4"
        source_file2.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.log_error"):
            # First error
            handler._handle_event(str(source_file1))
            # Second error - should not crash
            handler._handle_event(str(source_file2))

        # Both files should have been attempted
        assert producer.process_file.call_count == 2

    # -- T6.2: Event Debouncing Tests -------------------------------------

    def test_debounce_rapid_events_same_file_single_processing(self, valid_env, tmp_data_dir):
        """Multiple rapid events for the same file should be debounced into a single processing."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # First event at t=0 — should process
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

            # Second event at t=0.3 — within debounce window, should be skipped
            mock_monotonic.return_value = 0.3
            handler._handle_event(str(source_file))

            # Third event at t=0.6 — still within window, should be skipped
            mock_monotonic.return_value = 0.6
            handler._handle_event(str(source_file))

        producer.process_file.assert_called_once()

    def test_debounce_different_files_independent(self, valid_env, tmp_data_dir):
        """Different files should not be blocked by each other's debounce timers."""
        from focus_feed.cli import _WatchHandler

        file_a = tmp_data_dir / "file_a.mp4"
        file_a.write_bytes(b"video a")
        file_b = tmp_data_dir / "file_b.mp4"
        file_b.write_bytes(b"video b")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # Event for file_a at t=0
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(file_a))

            # Event for file_b at t=0.1 — different file, should process
            mock_monotonic.return_value = 0.1
            handler._handle_event(str(file_b))

            # Rapid event for file_a at t=0.2 — within debounce, should be skipped
            mock_monotonic.return_value = 0.2
            handler._handle_event(str(file_a))

        # Each file should have been processed once
        assert producer.process_file.call_count == 2
        call_filenames = [call.args[0].filename for call in producer.process_file.call_args_list]
        assert "file_a.mp4" in call_filenames
        assert "file_b.mp4" in call_filenames

    def test_debounce_allows_reprocessing_after_interval(self, valid_env, tmp_data_dir):
        """A file should be processed again after the debounce interval expires."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # First event at t=0 — should process
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

            # Second event within window at t=0.5 — should be skipped
            mock_monotonic.return_value = 0.5
            handler._handle_event(str(source_file))

        assert producer.process_file.call_count == 1

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # Third event after window at t=2.0 — should process again
            mock_monotonic.return_value = 2.0
            handler._handle_event(str(source_file))

        assert producer.process_file.call_count == 2

    def test_debounce_extends_window_on_rapid_events(self, valid_env, tmp_data_dir):
        """Rapid events should keep extending the debounce window."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # First event at t=0 — should process
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

            # Event at t=0.9 — within 1.0 window, should be skipped, extends window
            mock_monotonic.return_value = 0.9
            handler._handle_event(str(source_file))

            # Event at t=1.5 — 1.5 - 0.9 = 0.6 < 1.0, still within window, skipped
            mock_monotonic.return_value = 1.5
            handler._handle_event(str(source_file))

            # Event at t=2.0 — 2.0 - 1.5 = 0.5 < 1.0, still within window, skipped
            mock_monotonic.return_value = 2.0
            handler._handle_event(str(source_file))

        # Only the first event should have triggered processing
        producer.process_file.assert_called_once()

    def test_debounce_default_interval_is_one_second(self, valid_env, tmp_data_dir):
        """The default debounce interval should be 1.0 second."""
        from focus_feed.cli import _WatchHandler

        assert _WatchHandler._debounce_interval == 1.0

    def test_debounce_configurable_interval(self, valid_env, tmp_data_dir):
        """The debounce interval should be configurable via class attribute."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        # Create a subclass with a custom debounce interval
        class FastDebounceHandler(_WatchHandler):
            _debounce_interval = 0.5

        producer = mock.MagicMock()
        handler = FastDebounceHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # First event at t=0 — should process
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

            # Event at t=0.3 — within 0.5 window, should be skipped
            mock_monotonic.return_value = 0.3
            handler._handle_event(str(source_file))

            # Event at t=0.6 — 0.6 - 0.3 = 0.3 < 0.5, still within window, skipped
            mock_monotonic.return_value = 0.6
            handler._handle_event(str(source_file))

        producer.process_file.assert_called_once()

    def test_debounce_does_not_block_json_events(self, valid_env, tmp_data_dir):
        """Debounce should not prevent JSON orphaned-file detection."""
        from focus_feed.cli import _WatchHandler

        json_file = tmp_data_dir / "orphan.mp4.json"
        json_file.write_text("{}")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            # First JSON event at t=0
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(json_file))

            # Rapid second JSON event at t=0.3 — should be debounced
            mock_monotonic.return_value = 0.3
            handler._handle_event(str(json_file))

        # Producer should not have been called (JSON events don't process)
        producer.process_file.assert_not_called()

    def test_debounce_first_event_always_processed(self, valid_env, tmp_data_dir):
        """The first event for a file should always be processed (no prior timestamp)."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # _last_event should start empty
        assert handler._last_event == {}

        with mock.patch("focus_feed.cli.time.monotonic") as mock_monotonic:
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

        # First event should have been processed
        producer.process_file.assert_called_once()
        # Timestamp should now be recorded
        assert str(source_file) in handler._last_event

    # -- T6.3: Periodic Rescan Fallback Tests -------------------------------

    def test_rescan_interval_default_is_300_seconds(self, valid_env, tmp_data_dir):
        """The default rescan interval should be 300 seconds (5 minutes)."""
        from focus_feed.cli import _WatchHandler

        assert _WatchHandler._rescan_interval == 300

    def test_periodic_rescan_processes_new_files(self, valid_env, tmp_data_dir):
        """_periodic_rescan should process files found by the scanner."""
        from focus_feed.cli import _WatchHandler

        # Create a source file without JSON metadata
        source_file = tmp_data_dir / "missed.mp4"
        source_file.write_bytes(b"video data")

        producer = mock.MagicMock()
        producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            handler._periodic_rescan()

        # Scanner should have found the file and producer should have processed it
        producer.process_file.assert_called_once()
        call_arg = producer.process_file.call_args[0][0]
        assert call_arg.filename == "missed.mp4"
        # Feeds should have been regenerated
        mock_gen.assert_called_once()

    def test_periodic_rescan_ignores_hidden_temporary_and_generated_files(
        self, valid_env, tmp_data_dir
    ):
        from focus_feed.cli import _WatchHandler

        for filename in (".hidden.mp4", "video.mp4.tmp", "index.html"):
            (tmp_data_dir / filename).write_bytes(b"data")
        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        handler._periodic_rescan()

        producer.process_file.assert_not_called()

    def test_periodic_rescan_no_files_logs_info(self, valid_env, tmp_data_dir):
        """_periodic_rescan should handle empty directory gracefully."""
        from focus_feed.cli import _WatchHandler

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("focus_feed.cli._logger") as mock_logger:
                handler._periodic_rescan()

        producer.process_file.assert_not_called()
        mock_gen.assert_not_called()
        # Should log that no files were found
        mock_logger.info.assert_any_call("Periodic rescan: no new files to process.")

    def test_periodic_rescan_sets_last_rescan_time(self, valid_env, tmp_data_dir):
        """_periodic_rescan should record the timestamp of the last rescan."""
        from focus_feed.cli import _WatchHandler

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        assert handler.last_rescan_time is None

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("focus_feed.cli.time") as mock_time:
                mock_time.time.return_value = 1234567890.0
                handler._periodic_rescan()

        assert handler.last_rescan_time == 1234567890.0

    def test_periodic_rescan_logs_start_and_complete(self, valid_env, tmp_data_dir):
        """_periodic_rescan should log start and completion events."""
        from focus_feed.cli import _WatchHandler

        # Create a source file so the rescan has work to do
        source_file = tmp_data_dir / "new.mp4"
        source_file.write_bytes(b"video data")

        producer = mock.MagicMock()
        producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            with mock.patch("focus_feed.cli._logger") as mock_logger:
                handler._periodic_rescan()

        # Should log start
        mock_logger.info.assert_any_call("Periodic rescan started for %s", handler.data_dir)
        # Should log completion
        mock_logger.info.assert_any_call("Periodic rescan complete. Feeds regenerated.")

    def test_periodic_rescan_continues_after_processing_error(self, valid_env, tmp_data_dir):
        """_periodic_rescan should continue processing after an error."""
        from focus_feed.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("focus_feed.cli.log_error"):
                # Should not raise
                handler._periodic_rescan()

        producer.process_file.assert_called_once()
        # Feeds should still be regenerated even after errors
        mock_gen.assert_called_once()

    def test_periodic_rescan_skips_already_processed_files(self, valid_env, tmp_data_dir):
        """_periodic_rescan should not reprocess files with up-to-date JSON."""
        from focus_feed.cli import _WatchHandler

        # Create a source file with complete JSON metadata
        source_file = tmp_data_dir / "processed.mp4"
        source_file.write_bytes(b"video")
        json_file = tmp_data_dir / "processed.mp4.json"
        json_file.write_text(
            json.dumps(
                _make_valid_metadata(
                    filepath="processed.mp4",
                    filename="processed.mp4",
                    filetype="mp4",
                )
            )
        )

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            handler._periodic_rescan()

        # Scanner should have skipped the file (JSON is up-to-date)
        producer.process_file.assert_not_called()

    def test_watch_rescan_interval_env_var_override(
        self, runner, valid_env, tmp_data_dir, monkeypatch, watch_mocks
    ):
        """Watch should respect RESCAN_INTERVAL environment variable."""
        monkeypatch.setenv("RESCAN_INTERVAL", "60")

        with watch_mocks():
            result = runner.invoke(
                app,
                ["watch", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Periodic rescan enabled (every 60s)." in result.output

    def test_watch_rescan_disabled_when_zero(
        self, runner, valid_env, tmp_data_dir, monkeypatch, watch_mocks
    ):
        """Watch should disable periodic rescan when RESCAN_INTERVAL is 0."""
        monkeypatch.setenv("RESCAN_INTERVAL", "0")

        with watch_mocks():
            result = runner.invoke(
                app,
                ["watch", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Periodic rescan disabled." in result.output
