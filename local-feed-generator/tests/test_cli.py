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
import logging
from pathlib import Path
import time
from typing import Any
from unittest import mock

import click
from click.testing import CliRunner
from dotenv import load_dotenv
from local_feed_generator.cli import app, main
from local_feed_generator.scanner import ScannedFile
import pytest

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
]


@pytest.fixture
def clear_required_env(monkeypatch):
    """Clear all required env vars, yielding control, then they're auto-restored by monkeypatch."""
    for key in REQUIRED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


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
    import local_feed_generator.config as config_module

    config_module._config_instance = None


@pytest.fixture
def valid_env(monkeypatch):
    """Provide valid environment variables for all required config fields."""
    _set_valid_env(monkeypatch)
    # Mock CUDA availability check so tests that mock the Transcriber or
    # invoke the transcribe CLI command can proceed without an actual
    # CUDA installation.
    monkeypatch.setattr(
        "local_feed_generator.services.transcriber.check_cuda_available",
        lambda: True,
    )
    yield
    import local_feed_generator.config as config_module

    config_module._config_instance = None


@pytest.fixture(autouse=True)
def _restore_env():
    """Reset the config singleton after every test.

    This fixture ensures the config singleton is reset after each test
    so that environment state does not leak between test modules.
    """
    yield
    # Reset the config singleton
    import local_feed_generator.config as config_module

    config_module._config_instance = None


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
            mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen,
            mock.patch("local_feed_generator.producer.Producer") as MockProducer,
            mock.patch("watchdog.observers.Observer") as MockObserver,
            mock.patch("local_feed_generator.cli.time.sleep") as mock_sleep,
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
        "filepath": filepath,
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


class TestCliFramework:
    """Tests for the CLI framework (T5.1)."""

    def test_app_is_click_group(self):
        """The CLI should be a Click group."""
        assert isinstance(app, click.Group)

    def test_app_has_process_command(self):
        """The CLI should have a 'process' command."""
        assert "process" in app.commands

    def test_app_has_watch_command(self):
        """The CLI should have a 'watch' command."""
        assert "watch" in app.commands

    def test_app_has_feeds_command(self):
        """The CLI should have a 'feeds' command."""
        assert "feeds" in app.commands

    def test_app_has_validate_command(self):
        """The CLI should have a 'validate' command."""
        assert "validate" in app.commands

    def test_help_text(self, runner):
        """The CLI should display help text."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Local Feed Generator" in result.output

    def test_help_text_lists_log_level_option(self, runner):
        """The --help output should mention --log-level."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--log-level" in result.output

    def test_log_level_option_with_valid_value(self, runner):
        """The --log-level option should accept valid level names."""
        # Use a subcommand that will exit quickly; we only care that the
        # group callback runs without error.
        result = runner.invoke(app, ["--log-level", "DEBUG", "process", "--help"])
        assert result.exit_code == 0

    def test_log_level_option_case_insensitive(self, runner):
        """The --log-level option should accept lowercase level names."""
        result = runner.invoke(app, ["--log-level", "debug", "process", "--help"])
        assert result.exit_code == 0

    def test_log_level_option_short_flag(self, runner):
        """The -l short flag should work as an alias for --log-level."""
        result = runner.invoke(app, ["-l", "DEBUG", "process", "--help"])
        assert result.exit_code == 0

    def test_log_level_option_invalid_value(self, runner):
        """The --log-level option should reject invalid level names."""
        result = runner.invoke(app, ["--log-level", "INVALID", "process", "--help"])
        assert result.exit_code == 2

    def test_log_level_env_var(self, runner, monkeypatch):
        """The LOG_LEVEL env var should set the log level."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0

    def test_log_level_cli_overrides_env_var(self, runner, monkeypatch):
        """The --log-level CLI option should take precedence over LOG_LEVEL env var."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        result = runner.invoke(app, ["--log-level", "ERROR", "process", "--help"])
        assert result.exit_code == 0

    def test_configure_logging_level_sets_debug(self):
        """_configure_logging_level should set the logger level to DEBUG."""
        from local_feed_generator.cli import _configure_logging_level

        _configure_logging_level("DEBUG")
        _logger = logging.getLogger("local_feed_generator")
        assert _logger.level == logging.DEBUG

    def test_configure_logging_level_sets_critical(self):
        """_configure_logging_level should set the logger level to CRITICAL."""
        from local_feed_generator.cli import _configure_logging_level

        _configure_logging_level("CRITICAL")
        _logger = logging.getLogger("local_feed_generator")
        assert _logger.level == logging.CRITICAL

    def test_configure_logging_level_keeps_error_log_at_error(self):
        """_configure_logging_level should keep the error_log logger at ERROR level.

        The error log logger must always remain at ERROR level so that errors
        are always written to error.log regardless of the user's log level.
        """
        from local_feed_generator.cli import _configure_logging_level

        _configure_logging_level("WARNING")
        error_log_logger = logging.getLogger("local_feed_generator.error_log")
        assert error_log_logger.level == logging.ERROR

        _configure_logging_level("CRITICAL")
        error_log_logger = logging.getLogger("local_feed_generator.error_log")
        assert error_log_logger.level == logging.ERROR

    def test_configure_logging_level_invalid_falls_back_to_info(self):
        """_configure_logging_level should fall back to INFO for unknown levels."""
        from local_feed_generator.cli import _configure_logging_level

        _configure_logging_level("UNKNOWN")
        _logger = logging.getLogger("local_feed_generator")
        assert _logger.level == logging.INFO

    def test_configure_logging_level_case_insensitive(self):
        """_configure_logging_level should handle lowercase input."""
        from local_feed_generator.cli import _configure_logging_level

        _configure_logging_level("debug")
        _logger = logging.getLogger("local_feed_generator")
        assert _logger.level == logging.DEBUG

    def test_process_help(self, runner):
        """The process command should have help text."""
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "--config-file" in result.output
        assert "--force" in result.output
        assert "PATHS" in result.output

    def test_watch_help(self, runner):
        """The watch command should have help text."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--config-file" in result.output
        assert "--force" in result.output
        assert "PATHS" in result.output

    def test_feeds_help(self, runner):
        """The feeds command should have help text."""
        result = runner.invoke(app, ["feeds", "--help"])
        assert result.exit_code == 0
        assert "--config-file" in result.output
        assert "--force" in result.output
        assert "PATHS" in result.output

    def test_validate_help(self, runner):
        """The validate command should have help text."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--config-file" in result.output
        assert "--force" in result.output
        assert "PATHS" in result.output

    def test_config_loading_with_valid_env(self, runner, valid_env, tmp_data_dir):
        """Config should load successfully with valid environment variables."""
        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )
        assert result.exit_code == 0

    def test_config_error_exits_with_code_1(self, runner, tmp_path, monkeypatch):
        """CLI should exit with code 1 when config is invalid."""
        # Clear all required env vars
        required_keys = [
            "BASE_URL",
            "LLM_ENDPOINT",
            "LLM_API_KEY",
            "LLM_MODEL",
            "PROMPT_ARTICLE",
            "PROMPT_ARTICLE_REFLECTION",
            "PROMPT_PODCAST",
            "PROMPT_PODCAST_REFLECTION",
            "PROMPT_TITLE",
        ]
        for key in required_keys:
            monkeypatch.delenv(key, raising=False)
        import local_feed_generator.config as config_module

        config_module._config_instance = None

        result = runner.invoke(
            app,
            ["validate", str(tmp_path / "data")],
        )
        assert result.exit_code != 0

    def test_config_loading_from_env_file_without_config_file(
        self, runner, tmp_path, monkeypatch, clear_required_env
    ):
        """_load_config should load .env from cwd when config_file is None.

        This verifies the fix for the false CRITICAL warning: when env vars
        exist in a .env file (but not in os.environ), the CRITICAL warning
        should not fire.
        """
        # Write a .env file with all required vars
        env_content = (
            "BASE_URL=https://example.com/feeds\n"
            "LLM_ENDPOINT=https://llm.example.com/v1\n"
            "LLM_API_KEY=secret-key\n"
            "LLM_MODEL=my-model\n"
            "PROMPT_ARTICLE=Article prompt\n"
            "PROMPT_ARTICLE_REFLECTION=Article reflection prompt\n"
            "PROMPT_PODCAST=Podcast prompt\n"
            "PROMPT_PODCAST_REFLECTION=Podcast reflection prompt\n"
            "PROMPT_TITLE=Title prompt\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        import local_feed_generator.config as config_module

        config_module._config_instance = None

        monkeypatch.chdir(tmp_path)
        real_load_dotenv = load_dotenv

        def mock_load_dotenv(*args, **kwargs):
            return real_load_dotenv(env_file, override=False)

        with (
            mock.patch("local_feed_generator.config._logger") as mock_logger,
            mock.patch("dotenv.load_dotenv", side_effect=mock_load_dotenv),
        ):
            from local_feed_generator.cli import _load_config

            _load_config(None)

        # No CRITICAL log should have been emitted
        mock_logger.critical.assert_not_called()

    def test_config_loading_critical_when_env_truly_missing(
        self, runner, tmp_path, monkeypatch, clear_required_env
    ):
        """CRITICAL warning should fire when env vars are truly missing.

        When no .env file exists and env vars are not set, the CRITICAL
        warning should still fire.
        """
        import local_feed_generator.config as config_module

        config_module._config_instance = None

        monkeypatch.chdir(tmp_path)

        # No .env file exists, so CRITICAL should fire via config._logger
        # and _load_config should raise click.ClickException.
        # Mock load_dotenv to prevent it from finding the project root's
        # .env file via find_dotenv() parent directory search.
        with (
            mock.patch("local_feed_generator.config._logger") as mock_logger,
            mock.patch("dotenv.load_dotenv"),
        ):
            from local_feed_generator.cli import _load_config

            with pytest.raises(click.ClickException):
                _load_config(None)

        critical_calls = list(mock_logger.critical.call_args_list)
        assert len(critical_calls) > 0
        assert "Missing required configuration" in critical_calls[0].args[0]

    def test_data_dir_option_validates_directory(self, runner, valid_env, tmp_path):
        """Positional args should reject non-existent paths."""
        result = runner.invoke(
            app,
            ["process", str(tmp_path / "nonexistent")],
        )
        assert result.exit_code != 0

    def test_version_option(self, runner):
        """The CLI should support --version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# T5.2: Process Command Tests
# ---------------------------------------------------------------------------


class TestProcessCommand:
    """Tests for the process command (T5.2)."""

    def test_process_no_files_to_process(self, runner, valid_env, tmp_data_dir):
        """Process should report no files when directory is empty."""
        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 1}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 1}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 1}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
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


# ---------------------------------------------------------------------------
# T5.3: Watch Command Tests
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
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "new.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            handler._handle_event(str(source_file))

        producer.process_file.assert_called_once()

    def test_watch_handler_ignores_unsupported_files(self, valid_env, tmp_data_dir):
        """Watch handler should ignore non-source files."""
        from local_feed_generator.cli import _WatchHandler

        other_file = tmp_data_dir / "readme.txt"
        other_file.write_text("text")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        handler._handle_event(str(other_file))

        producer.process_file.assert_not_called()

    def test_watch_handler_logs_warning_for_orphaned_json(self, valid_env, tmp_data_dir):
        """Watch handler should log warning for orphaned JSON files."""
        from local_feed_generator.cli import _WatchHandler

        # Create a JSON file without a source file
        json_file = tmp_data_dir / "orphan.mp4.json"
        json_file.write_text("{}")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._logger") as mock_logger:
            handler._handle_json_event(json_file)

        mock_logger.warning.assert_called_once()
        assert "Orphaned JSON" in mock_logger.warning.call_args[0][0]

    def test_watch_handler_json_with_source_no_warning(self, valid_env, tmp_data_dir):
        """Watch handler should not warn for JSON files with existing source."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "doc.pdf"
        source_file.write_bytes(b"pdf")
        json_file = tmp_data_dir / "doc.pdf.json"
        json_file.write_text("{}")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._logger") as mock_logger:
            handler._handle_json_event(json_file)

        mock_logger.warning.assert_not_called()

    def test_watch_handler_feed_json_no_orphaned_warning(self, valid_env, tmp_data_dir):
        """Watch handler should not warn for feed.json (configuration file)."""
        from local_feed_generator.cli import _WatchHandler

        json_file = tmp_data_dir / "feed.json"
        json_file.write_text('{"title": "My Feed"}')

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._logger") as mock_logger:
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
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # _handle_event should not raise; it should catch and log the error
        with mock.patch("local_feed_generator.cli.log_error") as mock_log_error:
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
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.log_error") as mock_log_error:
            handler._handle_event(str(source_file))

        mock_log_error.assert_called_once()
        # Verify the filepath was passed
        call_kwargs = mock_log_error.call_args
        assert "bad.mp4" in call_kwargs[1]["filepath"]
        assert call_kwargs[1]["stage"] == "processing"

    def test_watch_handler_error_rate_limiting(self, valid_env, tmp_data_dir):
        """Watch handler should suppress detailed errors after rate limit."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # Pre-fill the error timestamps to the threshold
        handler._error_timestamps = [time.monotonic()] * handler._max_errors_per_minute

        with (
            mock.patch("local_feed_generator.cli.log_error") as mock_log_error,
            mock.patch("local_feed_generator.cli._logger") as mock_logger,
        ):
            handler._handle_event(str(source_file))

        # log_error should NOT have been called (rate limited)
        mock_log_error.assert_not_called()
        # A warning summary should have been logged instead
        mock_logger.warning.assert_called_once()
        assert "rate limit" in mock_logger.warning.call_args[0][0].lower()

    def test_watch_handler_rate_limiting_resets_after_window(self, valid_env, tmp_data_dir):
        """Watch handler should resume detailed logging after the window expires."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # Pre-fill with old timestamps (outside the 60-second window)
        handler._error_timestamps = [time.monotonic() - 61] * handler._max_errors_per_minute

        with mock.patch("local_feed_generator.cli.log_error") as mock_log_error:
            handler._handle_event(str(source_file))

        # log_error SHOULD have been called because old timestamps were trimmed
        mock_log_error.assert_called_once()

    def test_watch_handler_error_timestamps_trimmed(self, valid_env, tmp_data_dir):
        """Watch handler should trim old error timestamps on each error."""
        from local_feed_generator.cli import _WatchHandler

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

        with mock.patch("local_feed_generator.cli.log_error"):
            handler._handle_event(str(source_file))

        # The old timestamp should have been trimmed, leaving only 2 entries
        assert len(handler._error_timestamps) == 2

    def test_watch_handler_continues_after_multiple_errors(self, valid_env, tmp_data_dir):
        """Watch handler should continue processing after multiple errors."""
        from local_feed_generator.cli import _WatchHandler

        source_file1 = tmp_data_dir / "bad1.mp4"
        source_file1.write_bytes(b"video")
        source_file2 = tmp_data_dir / "bad2.mp4"
        source_file2.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.log_error"):
            # First error
            handler._handle_event(str(source_file1))
            # Second error - should not crash
            handler._handle_event(str(source_file2))

        # Both files should have been attempted
        assert producer.process_file.call_count == 2

    # -- T6.2: Event Debouncing Tests -------------------------------------

    def test_debounce_rapid_events_same_file_single_processing(self, valid_env, tmp_data_dir):
        """Multiple rapid events for the same file should be debounced into a single processing."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
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
        from local_feed_generator.cli import _WatchHandler

        file_a = tmp_data_dir / "file_a.mp4"
        file_a.write_bytes(b"video a")
        file_b = tmp_data_dir / "file_b.mp4"
        file_b.write_bytes(b"video b")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
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
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
            # First event at t=0 — should process
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

            # Second event within window at t=0.5 — should be skipped
            mock_monotonic.return_value = 0.5
            handler._handle_event(str(source_file))

        assert producer.process_file.call_count == 1

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
            # Third event after window at t=2.0 — should process again
            mock_monotonic.return_value = 2.0
            handler._handle_event(str(source_file))

        assert producer.process_file.call_count == 2

    def test_debounce_extends_window_on_rapid_events(self, valid_env, tmp_data_dir):
        """Rapid events should keep extending the debounce window."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
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
        from local_feed_generator.cli import _WatchHandler

        assert _WatchHandler._debounce_interval == 1.0

    def test_debounce_configurable_interval(self, valid_env, tmp_data_dir):
        """The debounce interval should be configurable via class attribute."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        # Create a subclass with a custom debounce interval
        class FastDebounceHandler(_WatchHandler):
            _debounce_interval = 0.5

        producer = mock.MagicMock()
        handler = FastDebounceHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
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
        from local_feed_generator.cli import _WatchHandler

        json_file = tmp_data_dir / "orphan.mp4.json"
        json_file.write_text("{}")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
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
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "video.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        # _last_event should start empty
        assert handler._last_event == {}

        with mock.patch("local_feed_generator.cli.time.monotonic") as mock_monotonic:
            mock_monotonic.return_value = 0.0
            handler._handle_event(str(source_file))

        # First event should have been processed
        producer.process_file.assert_called_once()
        # Timestamp should now be recorded
        assert str(source_file) in handler._last_event

    # -- T6.3: Periodic Rescan Fallback Tests -------------------------------

    def test_rescan_interval_default_is_300_seconds(self, valid_env, tmp_data_dir):
        """The default rescan interval should be 300 seconds (5 minutes)."""
        from local_feed_generator.cli import _WatchHandler

        assert _WatchHandler._rescan_interval == 300

    def test_periodic_rescan_processes_new_files(self, valid_env, tmp_data_dir):
        """_periodic_rescan should process files found by the scanner."""
        from local_feed_generator.cli import _WatchHandler

        # Create a source file without JSON metadata
        source_file = tmp_data_dir / "missed.mp4"
        source_file.write_bytes(b"video data")

        producer = mock.MagicMock()
        producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            handler._periodic_rescan()

        # Scanner should have found the file and producer should have processed it
        producer.process_file.assert_called_once()
        call_arg = producer.process_file.call_args[0][0]
        assert call_arg.filename == "missed.mp4"
        # Feeds should have been regenerated
        mock_gen.assert_called_once()

    def test_periodic_rescan_no_files_logs_info(self, valid_env, tmp_data_dir):
        """_periodic_rescan should handle empty directory gracefully."""
        from local_feed_generator.cli import _WatchHandler

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("local_feed_generator.cli._logger") as mock_logger:
                handler._periodic_rescan()

        producer.process_file.assert_not_called()
        mock_gen.assert_not_called()
        # Should log that no files were found
        mock_logger.info.assert_any_call("Periodic rescan: no new files to process.")

    def test_periodic_rescan_sets_last_rescan_time(self, valid_env, tmp_data_dir):
        """_periodic_rescan should record the timestamp of the last rescan."""
        from local_feed_generator.cli import _WatchHandler

        producer = mock.MagicMock()
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        assert handler.last_rescan_time is None

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("local_feed_generator.cli.time") as mock_time:
                mock_time.time.return_value = 1234567890.0
                handler._periodic_rescan()

        assert handler.last_rescan_time == 1234567890.0

    def test_periodic_rescan_logs_start_and_complete(self, valid_env, tmp_data_dir):
        """_periodic_rescan should log start and completion events."""
        from local_feed_generator.cli import _WatchHandler

        # Create a source file so the rescan has work to do
        source_file = tmp_data_dir / "new.mp4"
        source_file.write_bytes(b"video data")

        producer = mock.MagicMock()
        producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 1, "articles": 0}
            with mock.patch("local_feed_generator.cli._logger") as mock_logger:
                handler._periodic_rescan()

        # Should log start
        mock_logger.info.assert_any_call("Periodic rescan started for %s", handler.data_dir)
        # Should log completion
        mock_logger.info.assert_any_call("Periodic rescan complete. Feeds regenerated.")

    def test_periodic_rescan_continues_after_processing_error(self, valid_env, tmp_data_dir):
        """_periodic_rescan should continue processing after an error."""
        from local_feed_generator.cli import _WatchHandler

        source_file = tmp_data_dir / "bad.mp4"
        source_file.write_bytes(b"video")

        producer = mock.MagicMock()
        producer.process_file.side_effect = RuntimeError("Processing failed")
        handler = _WatchHandler(data_dir=tmp_data_dir, producer=producer)

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            with mock.patch("local_feed_generator.cli.log_error"):
                # Should not raise
                handler._periodic_rescan()

        producer.process_file.assert_called_once()
        # Feeds should still be regenerated even after errors
        mock_gen.assert_called_once()

    def test_periodic_rescan_skips_already_processed_files(self, valid_env, tmp_data_dir):
        """_periodic_rescan should not reprocess files with up-to-date JSON."""
        from local_feed_generator.cli import _WatchHandler

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

        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
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


class TestFeedsCommand:
    """Tests for the feeds command (T5.4)."""

    def test_feeds_skips_processing(self, runner, valid_env, tmp_data_dir):
        """Feeds should not process any files."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        # Create a JSON metadata file
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(_make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf"))
        )

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        mock_generator.generate_all_feeds.assert_called_once()

    def test_feeds_reads_existing_json(self, runner, valid_env, tmp_data_dir):
        """Feeds should read existing JSON metadata files."""
        # Create a valid MP4 metadata file
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video")
        (tmp_data_dir / "lecture.mp4.json").write_text(json.dumps(_make_valid_metadata()))
        # Create a valid PDF metadata file
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(_make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf"))
        )

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "1 item(s)" in result.output

    def test_feeds_empty_directory(self, runner, valid_env, tmp_data_dir):
        """Feeds should handle empty directory."""
        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "0 item(s)" in result.output

    def test_feeds_with_config_file(self, runner, valid_env, tmp_data_dir, tmp_path):
        """Feeds should accept --config-file option."""
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

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                [
                    "feeds",
                    str(tmp_data_dir),
                    "--config-file",
                    str(config_file),
                ],
            )

        assert result.exit_code == 0

    def test_feeds_does_not_count_error_metadata(self, runner, valid_env, tmp_data_dir):
        """Feeds should not count error metadata in the summary."""
        # Success metadata
        (tmp_data_dir / "good.mp4").write_bytes(b"video")
        (tmp_data_dir / "good.mp4.json").write_text(
            json.dumps(
                _make_valid_metadata(filepath="good.mp4", filename="good.mp4", filetype="mp4")
            )
        )
        # Error metadata
        (tmp_data_dir / "bad.mp4").write_bytes(b"video")
        error_meta = _make_valid_metadata(
            filepath="bad.mp4", filename="bad.mp4", filetype="mp4", error="Processing failed"
        )
        (tmp_data_dir / "bad.mp4.json").write_text(json.dumps(error_meta))

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        # Only 1 podcast item (the success one)
        assert "1 item(s)" in result.output

    def test_feeds_finalizes_incomplete_mp4_metadata(self, runner, valid_env, tmp_data_dir):
        """Feeds should finalize incomplete MP4 metadata before generating feeds."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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
                    "content": "Some transcript content",
                    "summary": "Some summary",
                }
            )
        )

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcribe"]["status"] == "complete"
        assert data["summarize"]["status"] == "complete"
        assert data["summarize"]["end_time"] is not None

    def test_feeds_skips_finalization_for_complete_files(self, runner, valid_env, tmp_data_dir):
        """Feeds should not re-finalize files that are already feed-eligible."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        # Already complete metadata (extract + summarize both complete)
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Existing Title",
                    "description": "Existing Description",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        # File should still be valid and feed-eligible
        json_path = tmp_data_dir / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["extract"]["status"] == "complete"
        assert data["summarize"]["status"] == "complete"

    def test_feeds_generates_title_for_mp4_without_title(self, runner, valid_env, tmp_data_dir):
        """Feeds finalization should generate a title for MP4 files when missing."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        (tmp_data_dir / "lecture.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Some transcript content",
                    "summary": "Some summary",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("Generated Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
                mock_generator = mock.MagicMock()
                mock_generator.generate_all_feeds.return_value = None
                MockGen.return_value = mock_generator

                result = runner.invoke(
                    app,
                    ["feeds", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["title"]["generated"] == "Generated Title"
        mock_summarizer.generate_title.assert_called_once()


class TestValidateCommand:
    """Tests for the validate command (T5.5)."""

    def test_validate_valid_metadata_no_errors(self, runner, valid_env, tmp_data_dir):
        """Validate should report no errors for valid metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(_make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf"))
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "All files are valid" in result.output

    def test_validate_invalid_json_schema(self, runner, valid_env, tmp_data_dir):
        """Validate should report errors for invalid JSON metadata."""
        (tmp_data_dir / "bad.pdf").write_bytes(b"pdf")
        # Invalid JSON - missing required fields (but has all step maps complete
        # so it won't be finalized)
        (tmp_data_dir / "bad.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Missing fields",
                }
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_validate_detects_orphaned_json(self, runner, valid_env, tmp_data_dir):
        """Validate should detect orphaned JSON files."""
        # JSON file without a source file
        (tmp_data_dir / "orphan.mp4.json").write_text(
            json.dumps(
                _make_valid_metadata(filepath="orphan.mp4", filename="orphan.mp4", filetype="mp4")
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0  # warnings, not errors
        assert "Orphaned JSON" in result.output
        assert "WARNING" in result.output

    def test_validate_does_not_flag_feed_json_as_orphaned(self, runner, valid_env, tmp_data_dir):
        """Validate should not flag feed.json as orphaned JSON."""
        (tmp_data_dir / "feed.json").write_text('{"title": "My Feed"}')

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "Orphaned JSON" not in result.output

    def test_validate_detects_special_characters(self, runner, valid_env, tmp_data_dir):
        """Validate should warn about special characters in filenames."""
        (tmp_data_dir / "file&test.mp4").write_bytes(b"video")

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "special characters" in result.output

    def test_validate_exit_code_0_when_only_warnings(self, runner, valid_env, tmp_data_dir):
        """Validate should exit 0 when only warnings are found (no errors)."""
        (tmp_data_dir / "orphan.mp4.json").write_text(
            json.dumps(
                _make_valid_metadata(filepath="orphan.mp4", filename="orphan.mp4", filetype="mp4")
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "Warnings" in result.output

    def test_validate_exit_code_1_when_errors(self, runner, valid_env, tmp_data_dir):
        """Validate should exit 1 when errors are found."""
        (tmp_data_dir / "bad.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "bad.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Missing fields",
                }
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 1

    def test_validate_reports_checked_count(self, runner, valid_env, tmp_data_dir):
        """Validate should report the number of JSON files checked."""
        (tmp_data_dir / "doc1.pdf").write_bytes(b"pdf1")
        (tmp_data_dir / "doc1.pdf.json").write_text(
            json.dumps(
                _make_valid_metadata(filepath="doc1.pdf", filename="doc1.pdf", filetype="pdf")
            )
        )
        (tmp_data_dir / "doc2.html").write_bytes(b"html2")
        (tmp_data_dir / "doc2.html.json").write_text(
            json.dumps(
                _make_valid_metadata(filepath="doc2.html", filename="doc2.html", filetype="html")
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "Checked 2 JSON metadata file(s)." in result.output

    def test_validate_empty_directory(self, runner, valid_env, tmp_data_dir):
        """Validate should handle empty directory."""
        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "Checked 0 JSON metadata file(s)." in result.output

    def test_validate_with_config_file(self, runner, valid_env, tmp_data_dir, tmp_path):
        """Validate should accept --config-file option."""
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
                "validate",
                str(tmp_data_dir),
                "--config-file",
                str(config_file),
            ],
        )

        assert result.exit_code == 0

    def test_validate_ignores_version_field(self, runner, valid_env, tmp_data_dir):
        """Validate should ignore the version field (no longer validated)."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["version"] = "2.0"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0

    def test_validate_multiple_errors(self, runner, valid_env, tmp_data_dir):
        """Validate should report all errors."""
        (tmp_data_dir / "bad1.pdf").write_bytes(b"pdf1")
        (tmp_data_dir / "bad1.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Missing fields",
                }
            )
        )
        (tmp_data_dir / "bad2.html").write_bytes(b"html2")
        (tmp_data_dir / "bad2.html.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Missing fields",
                }
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 1
        assert "Errors (2)" in result.output

    def test_validate_finalizes_incomplete_pdf_metadata(self, runner, valid_env, tmp_data_dir):
        """Validate should finalize incomplete PDF metadata before validation."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "in_progress", "input_size": 0, "output_size": 0},
                    "title": "PDF Title",
                    "description": "PDF Description",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["extract"]["status"] == "complete"
        assert data["extract"]["end_time"] is not None

    def test_validate_finalizes_and_reports_no_errors(self, runner, valid_env, tmp_data_dir):
        """Validate should finalize incomplete metadata and report no errors."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "in_progress", "input_size": 0, "output_size": 0},
                    "title": "PDF Title",
                    "description": "PDF Description",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "All files are valid" in result.output


class TestEntryPoint:
    """Tests for the entry point configuration (T5.6)."""

    def test_main_function_exists(self):
        """The main() function should be callable."""
        assert callable(main)

    def test_main_invokes_app(self, runner, valid_env, tmp_data_dir):
        """main() should invoke the app group."""
        with mock.patch("local_feed_generator.cli.app") as mock_app:
            main()
        mock_app.assert_called_once()

    def test_main_via_cli_runner(self, runner, valid_env, tmp_data_dir):
        """main() should work via CliRunner when called as the command."""
        with mock.patch("local_feed_generator.cli._generate_feeds") as mock_gen:
            mock_gen.return_value = {"podcast": 0, "articles": 0}
            result = runner.invoke(app, ["validate", str(tmp_data_dir)])
        assert result.exit_code == 0

    def test_all_commands_accessible_from_app(self):
        """All subcommands should be accessible from the app group."""
        result_commands = set(app.commands.keys())
        expected = {
            "process",
            "watch",
            "feeds",
            "validate",
            "transcribe",
            "summarize",
            "extract",
            "title",
            "create-feed",
        }
        assert expected.issubset(result_commands)

    def test_pyproject_entry_point(self):
        """The pyproject.toml should have the correct entry point."""
        import tomllib

        pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        scripts = config.get("project", {}).get("scripts", {})
        assert "local-feed-generator" in scripts
        assert scripts["local-feed-generator"] == "local_feed_generator.cli:main"


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestCliIntegration:
    """Integration-style tests for the CLI."""

    def test_process_and_generate_feeds_workflow(self, runner, valid_env, tmp_data_dir):
        """Full workflow: process files then generate feeds."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("local_feed_generator.producer.Producer") as MockProducer:
            mock_producer = mock.MagicMock()
            mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
            MockProducer.return_value = mock_producer

            with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
                mock_generator = mock.MagicMock()
                mock_generator.generate_all_feeds.return_value = None
                MockGen.return_value = mock_generator

                result = runner.invoke(
                    app,
                    ["process", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        mock_producer.process_file.assert_called_once()
        mock_generator.generate_all_feeds.assert_called_once()

    def test_validate_after_processing(self, runner, valid_env, tmp_data_dir):
        """Validate should work after processing."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(_make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf"))
        )

        result = runner.invoke(
            app,
            ["validate", str(tmp_data_dir)],
        )

        assert result.exit_code == 0
        assert "All files are valid" in result.output

    def test_feeds_after_processing(self, runner, valid_env, tmp_data_dir):
        """Feeds should work after processing."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(_make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf"))
        )

        with mock.patch("local_feed_generator.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "1 item(s)" in result.output

    def test_cli_group_with_no_command_shows_help(self, runner):
        """Invoking the CLI with no subcommand should show help."""
        result = runner.invoke(app, [], catch_exceptions=False)
        # Click groups exit with code 2 when no subcommand is given
        # and help is displayed
        assert result.exit_code == 2
        assert "process" in result.output
        assert "watch" in result.output
        assert "feeds" in result.output
        assert "validate" in result.output


# ---------------------------------------------------------------------------
# T5.7-T5.10: Granular Subcommand Tests
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
        assert "No MP4 files need transcription." in result.output

    def test_transcribe_dry_run(self, runner, valid_env, tmp_data_dir):
        """Transcribe --dry-run should list files without processing."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
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
        assert "No MP4 files need transcription." in result.output

    def test_transcribe_processes_files(self, runner, valid_env, tmp_data_dir):
        """Transcribe should process MP4 files without transcripts."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "This is a transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 MP4 file(s) to transcribe." in result.output
        mock_transcriber.transcribe.assert_called_once()

    def test_transcribe_writes_transcript_to_json(self, runner, valid_env, tmp_data_dir):
        """Transcribe should write the transcript to the JSON metadata file."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
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

    def test_transcribe_uses_positional_arg_for_directory(self, runner, valid_env, tmp_data_dir):
        """Transcribe should use the directory passed as a positional argument."""
        # Create a file in tmp_data_dir and another in a sibling directory
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        other_dir = tmp_data_dir.parent / "other_data"
        other_dir.mkdir()
        (other_dir / "other.mp4").write_bytes(b"other video")

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
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

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.side_effect = RuntimeError("Transcription failed")
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 1
        assert "1 failed" in result.output

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

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
            mock_transcriber = mock.MagicMock()
            mock_transcriber.transcribe.return_value = "New transcript."
            MockTranscriber.return_value = mock_transcriber

            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir), "--force"],
            )

        assert result.exit_code == 0
        assert "Found 1 MP4 file(s) to transcribe." in result.output
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

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
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

    def test_transcribe_cuda_unavailable_shows_error(self, runner, valid_env, tmp_data_dir):
        """Transcribe should show a clear error when CUDA is unavailable."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch(
            "local_feed_generator.services.transcriber.check_cuda_available",
            return_value=False,
        ):
            result = runner.invoke(
                app,
                ["transcribe", str(tmp_data_dir)],
            )

        assert result.exit_code == 1
        assert "CUDA libraries not found" in result.output
        assert "TRANSCRIPTION_DEVICE=cpu" in result.output

    def test_transcribe_sets_in_progress_before_and_complete_after(
        self, runner, valid_env, tmp_data_dir
    ):
        """Transcribe should set status to 'in_progress' before transcription, then 'complete'."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")

        with mock.patch("local_feed_generator.services.transcriber.Transcriber") as MockTranscriber:
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


class TestSummarizeCommand:
    """Tests for the summarize command (T5.8)."""

    def test_summarize_help(self, runner):
        """The summarize command should have help text."""
        result = runner.invoke(app, ["summarize", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--config-file" in result.output
        assert "--force" in result.output

    def test_summarize_no_files(self, runner, valid_env, tmp_data_dir):
        """Summarize should report no files when none need summarization."""
        result = runner.invoke(
            app,
            ["summarize", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No files need summarization." in result.output

    def test_summarize_dry_run(self, runner, valid_env, tmp_data_dir):
        """Summarize --dry-run should list files without processing."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir), "--dry-run"],
            )

        assert result.exit_code == 0
        assert "[Dry run]" in result.output
        assert "lecture.mp4" in result.output
        mock_summarizer.summarize.assert_not_called()

    def test_summarize_processes_files(self, runner, valid_env, tmp_data_dir):
        """Summarize should process MP4 files with transcripts but no summaries."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Key takeaways.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to summarize." in result.output
        mock_summarizer.summarize.assert_called_once()

    def test_summarize_skips_files_without_transcript(self, runner, valid_env, tmp_data_dir):
        """Summarize should skip MP4 files without a transcript."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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
                    "content": "",
                }
            )
        )

        result = runner.invoke(
            app,
            ["summarize", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No files need summarization." in result.output

    def test_summarize_writes_summary_to_json(self, runner, valid_env, tmp_data_dir):
        """Summarize should write the summary to the JSON metadata file."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Key takeaways.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["summary"] == "Key takeaways."
        assert data["description"] == ""  # MP4: description stays blank
        assert data["summarize"]["status"] == "complete"

    def test_summarize_force_flag_resummarizes_complete_files(
        self, runner, valid_env, tmp_data_dir
    ):
        """Summarize --force should re-summarize files with status['summarize']='complete'."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        (tmp_data_dir / "lecture.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Existing transcript",
                    "summary": "Old summary",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("New summary.", 200, 60)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir), "--force"],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to summarize." in result.output
        mock_summarizer.summarize.assert_called_once()

    def test_summarize_sets_in_progress_before_and_complete_after(
        self, runner, valid_env, tmp_data_dir
    ):
        """Summarize should set status to 'in_progress' before summarization, then 'complete'."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Key takeaways.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0

        # Verify final JSON state has complete status
        json_path = tmp_data_dir / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["summarize"]["status"] == "complete"
        assert data["summary"] == "Key takeaways."
        assert data["description"] == ""  # MP4: description stays blank

    def test_summarize_outputs_token_counts(self, runner, valid_env, tmp_data_dir):
        """Summarize command should output token counts from the LLM response."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = (
                "Key takeaways from this video.",
                1234,
                567,
            )
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Tokens in: 1234, Tokens out: 567" in result.output

    def test_summarize_outputs_word_count(self, runner, valid_env, tmp_data_dir):
        """Summarize command should output word count of the summary."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
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

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            # "Key takeaways from this video." has 5 words
            mock_summarizer.summarize.return_value = (
                "Key takeaways from this video.",
                100,
                50,
            )
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Summarized 5 words." in result.output

    def test_summarize_processes_pdf_with_content(self, runner, valid_env, tmp_data_dir):
        """Summarize should process PDF files that have content but no summary."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "PDF Description",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Extracted PDF text content.",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Key takeaways.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to summarize." in result.output
        mock_summarizer.summarize.assert_called_once()

    def test_summarize_pdf_preserves_description(self, runner, valid_env, tmp_data_dir):
        """Summarize should keep the existing description for PDF files."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "PDF Description",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Extracted PDF text content.",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Key takeaways.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["summary"] == "Key takeaways."
        assert data["description"] == "PDF Description"  # PDF: description preserved
        assert data["summarize"]["status"] == "complete"

    def test_summarize_processes_html_with_content(self, runner, valid_env, tmp_data_dir):
        """Summarize should process HTML files that have content but no summary."""
        (tmp_data_dir / "page.html").write_bytes(b"<html><body>Content</body></html>")
        (tmp_data_dir / "page.html.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "HTML Description",
                    "filename": "page.html",
                    "filepath": "page.html",
                    "filetype": "html",
                    "date": 1728931200,
                    "size": 512,
                    "content": "Extracted HTML text content.",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Key takeaways.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to summarize." in result.output
        mock_summarizer.summarize.assert_called_once()

    def test_summarize_skips_files_without_content(self, runner, valid_env, tmp_data_dir):
        """Summarize should skip files without content."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "",
                }
            )
        )

        result = runner.invoke(
            app,
            ["summarize", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No files need summarization." in result.output

    def test_summarize_dry_run_includes_pdf(self, runner, valid_env, tmp_data_dir):
        """Summarize --dry-run should list PDF files that need summarization."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Extracted text content.",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir), "--dry-run"],
            )

        assert result.exit_code == 0
        assert "[Dry run]" in result.output
        assert "doc.pdf" in result.output
        mock_summarizer.summarize.assert_not_called()

    def test_summarize_file_flag_processes_pdf(self, runner, valid_env, tmp_data_dir):
        """Summarize should process a specific file passed as positional arg."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Extracted text content.",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Summary.", 100, 50)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["summarize", str(tmp_data_dir / "doc.pdf")],
            )

        assert result.exit_code == 0
        mock_summarizer.summarize.assert_called_once()


class TestFindFilesNeedingSummary:
    """Tests for the _find_files_needing_summary helper function."""

    @pytest.mark.parametrize(
        "filename,filetype,content_key,content_value,extra_steps",
        [
            (
                "lecture.mp4",
                "mp4",
                "content",
                "Some transcript",
                {"transcribe": {"status": "complete", "input_size": 0, "output_size": 0}},
            ),
            (
                "doc.pdf",
                "pdf",
                "content",
                "Extracted PDF text.",
                {"extract": {"status": "complete", "input_size": 0, "output_size": 0}},
            ),
            (
                "page.html",
                "html",
                "content",
                "Extracted HTML text.",
                {"extract": {"status": "complete", "input_size": 0, "output_size": 0}},
            ),
        ],
    )
    def test_find_files_needing_summary_finds_mp4_pdf_html(
        self,
        valid_env,
        tmp_data_dir,
        filename,
        filetype,
        content_key,
        content_value,
        extra_steps,
    ):
        """Should find files of each type with content but no summary."""
        (tmp_data_dir / filename).write_bytes(b"data")
        metadata = {
            "summarize": {"status": None, "input_size": 0, "output_size": 0},
            "title": "Test",
            "description": "Desc" if filetype != "mp4" else "",
            "filename": filename,
            "filepath": filename,
            "filetype": filetype,
            "date": 1728931200,
            "size": 1024 if filetype != "html" else 512,
            content_key: content_value,
        }
        metadata.update(extra_steps)
        (tmp_data_dir / f"{filename}.json").write_text(json.dumps(metadata))

        from local_feed_generator.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir)
        assert len(result) == 1
        assert result[0].name == filename

    def test_find_files_needing_summary_skips_no_content(self, valid_env, tmp_data_dir):
        """Should skip files without content."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "",
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir)
        assert len(result) == 0

    def test_find_files_needing_summary_skips_complete_summarize(self, valid_env, tmp_data_dir):
        """Should skip files where summarize status is already complete."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Extracted text.",
                    "summary": "Existing summary",
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir)
        assert len(result) == 0

    def test_find_files_needing_summary_force_includes_complete(self, valid_env, tmp_data_dir):
        """Force should include files with summarize status='complete'."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Extracted text.",
                    "summary": "Existing summary",
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir, force=True)
        assert len(result) == 1

    def test_find_files_needing_summary_skips_no_json(self, valid_env, tmp_data_dir):
        """Should skip files without JSON metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")

        from local_feed_generator.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir)
        assert len(result) == 0

    def test_find_files_needing_summary_mixed_types(self, valid_env, tmp_data_dir):
        """Should find files of all supported types that need summarization."""
        # MP4 with content, no summary
        (tmp_data_dir / "video.mp4").write_bytes(b"video")
        (tmp_data_dir / "video.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "",
                    "filename": "video.mp4",
                    "filepath": "video.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Transcript",
                }
            )
        )
        # PDF with content, no summary
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "PDF text",
                }
            )
        )
        # HTML with content and summary complete (should be skipped)
        (tmp_data_dir / "page.html").write_bytes(b"html")
        (tmp_data_dir / "page.html.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "page.html",
                    "filepath": "page.html",
                    "filetype": "html",
                    "date": 1728931200,
                    "size": 512,
                    "content": "HTML text",
                    "summary": "Summary",
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir)
        assert len(result) == 2
        names = [f.name for f in result]
        assert "video.mp4" in names
        assert "doc.pdf" in names
        assert "page.html" not in names

    @pytest.mark.parametrize(
        "filter_type",
        ["file", "directory"],
    )
    def test_find_files_needing_summary_paths_filters(self, valid_env, tmp_data_dir, filter_type):
        """Should filter results to specific file or directory."""
        # Create a subdirectory with a file
        subdir = tmp_data_dir / "subdir"
        subdir.mkdir()
        (subdir / "doc1.pdf").write_bytes(b"pdf")
        (subdir / "doc1.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc1.pdf",
                    "filepath": "doc1.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Content 1",
                }
            )
        )
        # Create a file in the parent directory
        (tmp_data_dir / "doc2.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc2.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "title": "Test",
                    "description": "Desc",
                    "filename": "doc2.pdf",
                    "filepath": "doc2.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "Content 2",
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_summary

        # Without paths filter: both files returned
        result = _find_files_needing_summary(tmp_data_dir)
        assert len(result) == 2

        if filter_type == "file":
            # With paths filter for only doc1.pdf: only doc1.pdf returned
            result = _find_files_needing_summary(tmp_data_dir, paths=[tmp_data_dir / "doc2.pdf"])
            assert len(result) == 1
            assert result[0].name == "doc2.pdf"
        else:
            # With paths filter for subdir: only doc1.pdf returned
            result = _find_files_needing_summary(tmp_data_dir, paths=[subdir])
            assert len(result) == 1
            assert result[0].name == "doc1.pdf"


class TestFindFilesNeedingTranscription:
    """Tests for the _find_files_needing_transcription helper function."""

    def test_find_files_needing_transcription_finds_mp4_without_json(self, valid_env, tmp_data_dir):
        """Should find MP4 files with no JSON metadata."""
        (tmp_data_dir / "video.mp4").write_bytes(b"video")

        from local_feed_generator.cli import _find_files_needing_transcription

        result = _find_files_needing_transcription(tmp_data_dir)
        assert len(result) == 1
        assert result[0].name == "video.mp4"

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

        from local_feed_generator.cli import _find_files_needing_transcription

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

        from local_feed_generator.cli import _find_files_needing_transcription

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


class TestFindFilesNeedingMetadataExtraction:
    """Tests for the _find_files_needing_metadata_extraction helper function."""

    def test_find_files_needing_metadata_extraction_finds_pdf_without_json(
        self, valid_env, tmp_data_dir
    ):
        """Should find PDF files with no JSON metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")

        from local_feed_generator.cli import _find_files_needing_metadata_extraction

        result = _find_files_needing_metadata_extraction(tmp_data_dir)
        assert len(result) == 1
        assert result[0].name == "doc.pdf"

    def test_find_files_needing_metadata_extraction_skips_complete_with_title(
        self, valid_env, tmp_data_dir
    ):
        """Should skip PDF files with complete extraction and a title."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete"},
                    "title": "My Doc",
                    "filename": "doc.pdf",
                    "filetype": "pdf",
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_metadata_extraction

        result = _find_files_needing_metadata_extraction(tmp_data_dir)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "filter_type",
        ["file", "directory"],
    )
    def test_find_files_needing_metadata_extraction_paths_filters(
        self, valid_env, tmp_data_dir, filter_type
    ):
        """Should filter results to specific file or directory."""
        (tmp_data_dir / "doc1.pdf").write_bytes(b"pdf")
        (tmp_data_dir / "doc2.pdf").write_bytes(b"pdf")

        from local_feed_generator.cli import _find_files_needing_metadata_extraction

        # Without paths filter: both files returned
        result = _find_files_needing_metadata_extraction(tmp_data_dir)
        assert len(result) == 2

        if filter_type == "file":
            # With paths filter for only doc1.pdf: only doc1.pdf returned
            result = _find_files_needing_metadata_extraction(
                tmp_data_dir, paths=[tmp_data_dir / "doc1.pdf"]
            )
            assert len(result) == 1
            assert result[0].name == "doc1.pdf"
        else:
            # With paths filter for subdir: only doc1.pdf returned
            subdir = tmp_data_dir / "subdir"
            subdir.mkdir()
            (subdir / "doc1.pdf").write_bytes(b"pdf")
            result = _find_files_needing_metadata_extraction(tmp_data_dir, paths=[subdir])
            assert len(result) == 1
            assert result[0].name == "doc1.pdf"

    """Tests for the extract command (T5.9)."""

    def test_extract_help(self, runner):
        """The extract command should have help text."""
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--config-file" in result.output
        assert "--force" in result.output

    def test_extract_no_files(self, runner, valid_env, tmp_data_dir):
        """Extract should report no files when directory is empty."""
        result = runner.invoke(
            app,
            ["extract", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No PDF/HTML files need metadata extraction." in result.output

    def test_extract_dry_run(self, runner, valid_env, tmp_data_dir):
        """Extract --dry-run should list files without processing."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("local_feed_generator.services.extractors.PDFExtractor") as MockExtractor:
            mock_extractor = mock.MagicMock()
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(
                app,
                ["extract", str(tmp_data_dir), "--dry-run"],
            )

        assert result.exit_code == 0
        assert "[Dry run]" in result.output
        assert "doc.pdf" in result.output
        mock_extractor.extract.assert_not_called()

    @pytest.mark.parametrize(
        "filename,content,extractor_path,expected_title,expected_description",
        [
            ("doc.pdf", b"pdf data", "PDFExtractor", "PDF Title", "PDF Description"),
            (
                "page.html",
                b"<html><title>Test</title></html>",
                "HTMLExtractor",
                "HTML Title",
                "HTML Description",
            ),
        ],
    )
    def test_extract_processes_pdf_and_html(
        self,
        runner,
        valid_env,
        tmp_data_dir,
        filename,
        content,
        extractor_path,
        expected_title,
        expected_description,
    ):
        """Extract should process PDF and HTML files without status['extract']='complete'."""
        (tmp_data_dir / filename).write_bytes(content)

        extractor_target = f"local_feed_generator.services.extractors.{extractor_path}"
        with mock.patch(extractor_target) as MockExtractor:
            mock_extractor = mock.MagicMock()
            mock_extractor.extract.return_value = {
                "title": expected_title,
                "description": expected_description,
                "date": 1728931200,
                "size": 1024,
                "filetype": filename.rsplit(".", 1)[-1],
            }
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(
                app,
                ["extract", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 PDF/HTML file(s)" in result.output
        mock_extractor.extract.assert_called_once()

    def test_extract_skips_files_with_title(self, runner, valid_env, tmp_data_dir):
        """Extract should skip files that already have extract status='complete'."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Existing Title",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        result = runner.invoke(
            app,
            ["extract", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No PDF/HTML files need metadata extraction." in result.output

    def test_extract_writes_to_json(self, runner, valid_env, tmp_data_dir):
        """Extract should write extracted data to JSON metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("local_feed_generator.services.extractors.PDFExtractor") as MockExtractor:
            mock_extractor = mock.MagicMock()
            mock_extractor.extract.return_value = {
                "title": "PDF Title",
                "description": "PDF Description",
                "date": 1728931200,
                "size": 1024,
                "filetype": "pdf",
            }
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(
                app,
                ["extract", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "doc.pdf.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["title"] == "PDF Title"
        assert data["description"] == "PDF Description"
        assert data["filetype"] == "pdf"
        assert data["extract"]["status"] == "complete"

    def test_extract_force_flag_reprocesses_complete_files(self, runner, valid_env, tmp_data_dir):
        """Extract --force should re-extract files with extract status='complete'."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "Existing Title",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.extractors.PDFExtractor") as MockExtractor:
            mock_extractor = mock.MagicMock()
            mock_extractor.extract.return_value = {
                "title": "New Title",
                "description": "New Description",
                "date": 1728931200,
                "size": 2048,
                "filetype": "pdf",
            }
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(
                app,
                ["extract", str(tmp_data_dir), "--force"],
            )

        assert result.exit_code == 0
        assert "Found 1 PDF/HTML file(s)" in result.output
        mock_extractor.extract.assert_called_once()

    def test_extract_sets_in_progress_before_and_complete_after(
        self, runner, valid_env, tmp_data_dir
    ):
        """Extract should set status to 'in_progress' before extraction, then 'complete' after."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("local_feed_generator.services.extractors.PDFExtractor") as MockExtractor:
            mock_extractor = mock.MagicMock()
            mock_extractor.extract.return_value = {
                "title": "PDF Title",
                "description": "PDF Description",
                "date": 1728931200,
                "size": 1024,
                "filetype": "pdf",
            }
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(
                app,
                ["extract", str(tmp_data_dir)],
            )

        assert result.exit_code == 0

        # Verify final JSON state has complete status and end_time
        json_path = tmp_data_dir / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["extract"]["status"] == "complete"
        assert data["title"] == "PDF Title"
        assert data["extract"]["end_time"] is not None


# ---------------------------------------------------------------------------
# T5.10: Title Command Tests
# ---------------------------------------------------------------------------


class TestTitleCommand:
    """Tests for the title command (T5.10)."""

    def test_title_help(self, runner):
        """The title command should have help text."""
        result = runner.invoke(app, ["title", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--config-file" in result.output
        assert "--force" in result.output

    def test_title_no_files_needing_titles(self, runner, valid_env, tmp_data_dir):
        """Title should report no files when all have generated titles."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": {"filename": "doc", "generated": "Proper Title"},
                    "description": "Desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        result = runner.invoke(
            app,
            ["title", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No files need title generation." in result.output

    def test_title_dry_run(self, runner, valid_env, tmp_data_dir):
        """Title --dry-run should list files without processing."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir), "--dry-run"],
            )

        assert result.exit_code == 0
        assert "[Dry run]" in result.output
        assert "doc.pdf" in result.output
        mock_summarizer.generate_title.assert_not_called()

    def test_title_processes_pdf_with_empty_title(self, runner, valid_env, tmp_data_dir):
        """Title should process PDF files with empty titles."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("Generated Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to generate titles for." in result.output
        mock_summarizer.generate_title.assert_called_once()

    def test_title_writes_title_to_json(self, runner, valid_env, tmp_data_dir):
        """Title command should write the generated title to the JSON metadata file."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("Generated Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        json_path = tmp_data_dir / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["title"]["generated"] == "Generated Title"

    def test_title_skips_files_with_proper_title(self, runner, valid_env, tmp_data_dir):
        """Title should skip files that already have a generated title."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": {"filename": "doc", "generated": "A Proper Title"},
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "No files need title generation." in result.output
        mock_summarizer.generate_title.assert_not_called()

    def test_title_force_regenerates_for_proper_title(self, runner, valid_env, tmp_data_dir):
        """Title --force should regenerate titles even for files with generated titles."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": {"filename": "doc", "generated": "Existing Title"},
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("New Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir), "--force"],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to generate titles for." in result.output
        mock_summarizer.generate_title.assert_called_once()

    def test_title_processes_mp4_with_transcript(self, runner, valid_env, tmp_data_dir):
        """Title should process MP4 files using the transcript as content."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        (tmp_data_dir / "lecture.mp4.json").write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "Summary",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1024,
                    "content": "This is the transcript content.",
                    "summary": "Summary",
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("Video Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to generate titles for." in result.output
        # Should use transcript as content
        call_arg = mock_summarizer.generate_title.call_args[0][0]
        assert "This is the transcript content." in call_arg

    def test_title_processes_html_with_empty_title(self, runner, valid_env, tmp_data_dir):
        """Title should process HTML files with empty titles."""
        (tmp_data_dir / "page.html").write_text("<html><body>Content</body></html>")
        (tmp_data_dir / "page.html.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "page.html",
                    "filepath": "page.html",
                    "filetype": "html",
                    "date": 1728931200,
                    "size": 512,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("HTML Page Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to generate titles for." in result.output

    def test_title_file_flag_processes_specific_file(self, runner, valid_env, tmp_data_dir):
        """Title should process a specific file passed as positional arg."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )
        (tmp_data_dir / "other.pdf").write_bytes(b"other pdf")
        (tmp_data_dir / "other.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": {"filename": "other", "extracted": None, "generated": "Other Title"},
                    "description": "",
                    "filename": "other.pdf",
                    "filepath": "other.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir / "doc.pdf")],
            )

        assert result.exit_code == 0
        # Only doc.pdf should be processed (other.pdf already has a generated title)
        assert "Found 1 file(s) to generate titles for." in result.output

    def test_title_file_not_found(self, runner, valid_env, tmp_data_dir):
        """Title should error when file does not exist."""
        result = runner.invoke(
            app,
            ["title", str(tmp_data_dir / "nonexistent.pdf")],
        )
        assert result.exit_code != 0
        assert "Path does not exist" in result.output

    def test_title_processes_files_with_filename_stem_title(self, runner, valid_env, tmp_data_dir):
        """Title should process files where title equals the filename stem (not a proper title)."""
        (tmp_data_dir / "my_document.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "my_document.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "my document",  # filename stem with underscores replaced
                    "description": "",
                    "filename": "my_document.pdf",
                    "filepath": "my_document.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("New Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Found 1 file(s) to generate titles for." in result.output
        mock_summarizer.generate_title.assert_called_once()

    def test_title_skips_files_without_json(self, runner, valid_env, tmp_data_dir):
        """Title should skip files that don't have a JSON metadata file."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        result = runner.invoke(
            app,
            ["title", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No files need title generation." in result.output

    def test_title_exit_code_1_on_error(self, runner, valid_env, tmp_data_dir):
        """Title should exit with code 1 when title generation fails."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.side_effect = RuntimeError("LLM failed")
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 1
        assert "1 failed" in result.output

    def test_title_outputs_generated_title(self, runner, valid_env, tmp_data_dir):
        """Title command should output the generated title."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("local_feed_generator.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("My Document Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Title: My Document Title" in result.output


class TestFindFilesNeedingTitle:
    """Tests for the _find_files_needing_title helper function."""

    def test_find_files_needing_title_finds_empty_title(self, valid_env, tmp_data_dir):
        """Should find files with empty titles."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        result = _find_files_needing_title(tmp_data_dir)
        assert len(result) == 1
        assert result[0].name == "doc.pdf"

    def test_find_files_needing_title_finds_filename_stem_title(self, valid_env, tmp_data_dir):
        """Should find files where title is a string (no generated key)."""
        (tmp_data_dir / "my_doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "my_doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "my doc",  # string title, no generated key
                    "description": "",
                    "filename": "my_doc.pdf",
                    "filepath": "my_doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        result = _find_files_needing_title(tmp_data_dir)
        assert len(result) == 1

    def test_find_files_needing_title_skips_proper_title(self, valid_env, tmp_data_dir):
        """Should skip files with generated titles."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": {"filename": "doc", "generated": "A Proper Title"},
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        result = _find_files_needing_title(tmp_data_dir)
        assert len(result) == 0

    def test_find_files_needing_title_force_includes_proper_title(self, valid_env, tmp_data_dir):
        """Force should include files with generated titles."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": {"filename": "doc", "generated": "A Proper Title"},
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        result = _find_files_needing_title(tmp_data_dir, force=True)
        assert len(result) == 1

    def test_find_files_needing_title_skips_orphaned_json(self, valid_env, tmp_data_dir):
        """Should skip JSON files without a source file."""
        (tmp_data_dir / "orphan.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "orphan.pdf",
                    "filepath": "orphan.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        result = _find_files_needing_title(tmp_data_dir)
        assert len(result) == 0

    def test_find_files_needing_title_skips_ignored_filenames(self, valid_env, tmp_data_dir):
        """Should skip JSON files with ignored names (podcast.xml, etc.)."""
        # Create a JSON file that would be ignored
        (tmp_data_dir / "podcast.xml.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "podcast.xml",
                    "filepath": "podcast.xml",
                    "filetype": "html",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        result = _find_files_needing_title(tmp_data_dir)
        assert len(result) == 0

    def test_find_files_needing_title_paths_filters_to_specific_file(self, valid_env, tmp_data_dir):
        """Should only return files matching the specified paths."""
        # Create two files needing title generation
        (tmp_data_dir / "doc1.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc1.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc1.pdf",
                    "filepath": "doc1.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )
        (tmp_data_dir / "doc2.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc2.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc2.pdf",
                    "filepath": "doc2.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        # Without paths filter: both files returned
        result = _find_files_needing_title(tmp_data_dir)
        assert len(result) == 2

        # With paths filter for only doc1.pdf: only doc1.pdf returned
        result = _find_files_needing_title(tmp_data_dir, paths=[tmp_data_dir / "doc1.pdf"])
        assert len(result) == 1
        assert result[0].name == "doc1.pdf"

    def test_find_files_needing_title_paths_filters_to_directory(self, valid_env, tmp_data_dir):
        """Should return files within the specified directory."""
        # Create a subdirectory with a file
        subdir = tmp_data_dir / "subdir"
        subdir.mkdir()
        (subdir / "doc1.pdf").write_bytes(b"pdf data")
        (subdir / "doc1.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc1.pdf",
                    "filepath": "doc1.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )
        # Create a file in the parent directory
        (tmp_data_dir / "doc2.pdf").write_bytes(b"pdf data")
        (tmp_data_dir / "doc2.pdf.json").write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                    "title": "",
                    "description": "",
                    "filename": "doc2.pdf",
                    "filepath": "doc2.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        from local_feed_generator.cli import _find_files_needing_title

        # With paths filter for subdir: only doc1.pdf returned
        result = _find_files_needing_title(tmp_data_dir, paths=[subdir])
        assert len(result) == 1
        assert result[0].name == "doc1.pdf"


class TestGetContentForTitle:
    """Tests for the _get_content_for_title helper function."""

    @pytest.mark.parametrize(
        "filetype,metadata_dict,expected",
        [
            ("mp4", {"content": "This is the transcript."}, "This is the transcript."),
            ("mp4", {}, ""),
            ("txt", {}, ""),
        ],
    )
    def test_get_content_for_title_basic(
        self, valid_env, tmp_path, filetype, metadata_dict, expected
    ):
        """Should return content from metadata for MP4, empty for unsupported types."""
        from local_feed_generator.cli import _get_content_for_title

        if filetype == "mp4":
            file_path = Path("/fake/path/lecture.mp4")
        else:
            file_path = tmp_path / f"file.{filetype}"
        result = _get_content_for_title(file_path, metadata_dict)
        assert result == expected

    @pytest.mark.parametrize(
        "filetype,content",
        [
            ("pdf", "PDF content here."),
            ("html", "<html>HTML content</html>"),
        ],
    )
    def test_get_content_for_title_reads_file(self, valid_env, tmp_path, filetype, content):
        """Should read file content for PDF and HTML files."""
        from local_feed_generator.cli import _get_content_for_title

        source = tmp_path / f"file.{filetype}"
        source.write_text(content)
        metadata_dict: dict[str, Any] = {}
        result = _get_content_for_title(source, metadata_dict)
        assert content in result

    def test_get_content_for_title_truncates_long_content(self, valid_env, tmp_path):
        """Should truncate content to 8000 characters for PDF/HTML files."""
        from local_feed_generator.cli import _get_content_for_title

        source = tmp_path / "doc.pdf"
        source.write_text("x" * 20000)
        metadata_dict: dict[str, Any] = {}
        result = _get_content_for_title(source, metadata_dict)
        assert len(result) <= 8000

    def test_get_content_for_title_pdf_handles_read_error(self, valid_env, tmp_path):
        """Should return empty string when file cannot be read."""
        from local_feed_generator.cli import _get_content_for_title

        source = tmp_path / "doc.pdf"
        source.write_text("content")
        # Mock the file read to raise an error (avoids chmod which is platform-dependent)
        with mock.patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
            metadata_dict: dict[str, Any] = {}
            result = _get_content_for_title(source, metadata_dict)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests for _is_dict_feed_eligible with content/summarize
# ---------------------------------------------------------------------------


class TestIsDictFeedEligible:
    """Tests for the _is_dict_feed_eligible helper function."""

    def test_pdf_eligible_with_extract_and_summarize(self, valid_env):
        """PDF should be eligible if both extract and summarize are complete."""
        from local_feed_generator.scanner import _is_dict_feed_eligible

        metadata = {
            "extract": {"status": "complete"},
            "summarize": {"status": "complete"},
            "filetype": "pdf",
            "title": "Test",
            "filename": "doc.pdf",
            "filepath": "doc.pdf",
            "size": 1024,
        }
        assert _is_dict_feed_eligible(metadata, "pdf") is True

    def test_pdf_not_eligible_with_extract_only(self, valid_env):
        """PDF with only extract (no summarize) should not be eligible."""
        from local_feed_generator.scanner import _is_dict_feed_eligible

        metadata = {
            "extract": {"status": "complete"},
            "filetype": "pdf",
            "title": "Test",
            "filename": "doc.pdf",
            "filepath": "doc.pdf",
            "size": 1024,
        }
        assert _is_dict_feed_eligible(metadata, "pdf") is False

    def test_html_eligible_with_extract_and_summarize(self, valid_env):
        """HTML should be eligible if both extract and summarize are complete."""
        from local_feed_generator.scanner import _is_dict_feed_eligible

        metadata = {
            "extract": {"status": "complete"},
            "summarize": {"status": "complete"},
            "filetype": "html",
            "title": "Test",
            "filename": "page.html",
            "filepath": "page.html",
            "size": 512,
        }
        assert _is_dict_feed_eligible(metadata, "html") is True

    def test_html_not_eligible_with_extract_only(self, valid_env):
        """HTML with only extract (no summarize) should not be eligible."""
        from local_feed_generator.scanner import _is_dict_feed_eligible

        metadata = {
            "extract": {"status": "complete"},
            "filetype": "html",
            "title": "Test",
            "filename": "page.html",
            "filepath": "page.html",
            "size": 512,
        }
        assert _is_dict_feed_eligible(metadata, "html") is False

    def test_pdf_not_eligible_with_error(self, valid_env):
        """PDF with error should not be eligible."""
        from local_feed_generator.scanner import _is_dict_feed_eligible

        metadata = {
            "extract": {"status": "complete"},
            "summarize": {"status": "complete"},
            "error": "Some error",
            "filetype": "pdf",
            "title": "Test",
            "filename": "doc.pdf",
            "filepath": "doc.pdf",
            "size": 1024,
        }
        assert _is_dict_feed_eligible(metadata, "pdf") is False

    def test_pdf_not_eligible_without_extract(self, valid_env):
        """PDF without extract step should not be eligible."""
        from local_feed_generator.scanner import _is_dict_feed_eligible

        metadata = {
            "filetype": "pdf",
            "title": "Test",
            "filename": "doc.pdf",
            "filepath": "doc.pdf",
            "size": 1024,
        }
        assert _is_dict_feed_eligible(metadata, "pdf") is False


# ---------------------------------------------------------------------------
# Create-feed command tests
# ---------------------------------------------------------------------------


class TestCreateFeedCommand:
    """Tests for the create-feed command."""

    def test_create_feed_help(self, runner):
        """The create-feed command should have help text."""
        result = runner.invoke(app, ["create-feed", "--help"])
        assert result.exit_code == 0
        assert "DIRECTORY" in result.output

    def test_create_feed_creates_feed_json(self, runner, tmp_path):
        """create-feed should create a feed.json file with template content."""
        target_dir = tmp_path / "my-course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        feed_json_path = target_dir / "feed.json"
        assert feed_json_path.exists()

        data = json.loads(feed_json_path.read_text(encoding="utf-8"))
        assert data["title"] == "Feed Title"
        assert data["description"] == "Feed description"

    def test_create_feed_no_link_field(self, runner, tmp_path):
        """The generated feed.json should NOT contain a link field."""
        target_dir = tmp_path / "my-course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert "link" not in data

    def test_create_feed_creates_directory_if_missing(self, runner, tmp_path):
        """create-feed should create the target directory if it doesn't exist."""
        target_dir = tmp_path / "nested" / "deep" / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        assert (target_dir / "feed.json").exists()

    def test_create_feed_overwrites_existing(self, runner, tmp_path):
        """create-feed should overwrite an existing feed.json."""
        target_dir = tmp_path / "course"
        target_dir.mkdir()
        (target_dir / "feed.json").write_text(
            json.dumps({"title": "Old Title", "description": "Old Desc"}),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert data["title"] == "Feed Title"
        assert data["description"] == "Feed description"

    def test_create_feed_output_message(self, runner, tmp_path):
        """create-feed should print a confirmation message with the path."""
        target_dir = tmp_path / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        assert "Created feed.json" in result.output
        assert str(target_dir / "feed.json") in result.output

    def test_create_feed_includes_language_and_author(self, runner, tmp_path):
        """create-feed should include language and author fields in the template."""
        target_dir = tmp_path / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert data["language"] == "en-us"
        assert data["author"] == "Feed Author"

    def test_create_feed_includes_image(self, runner, tmp_path):
        """create-feed should include an image dict in the template."""
        target_dir = tmp_path / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert "image" in data
        assert data["image"]["url"] == "https://www.example.com/logo.jpg"
        assert data["image"]["title"] == "Logo"
        assert data["image"]["link"] == "https://www.example.com/"
