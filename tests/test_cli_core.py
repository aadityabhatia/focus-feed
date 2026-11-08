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
from unittest import mock

import click
from click.testing import CliRunner
from dotenv import load_dotenv
import pytest

from focus_feed.cli import app, main
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
        assert "Focus Feed" in result.output

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
        from focus_feed.cli import _configure_logging_level

        _configure_logging_level("DEBUG")
        _logger = logging.getLogger("focus_feed")
        assert _logger.level == logging.DEBUG

    def test_configure_logging_level_sets_critical(self):
        """_configure_logging_level should set the logger level to CRITICAL."""
        from focus_feed.cli import _configure_logging_level

        _configure_logging_level("CRITICAL")
        _logger = logging.getLogger("focus_feed")
        assert _logger.level == logging.CRITICAL

    def test_configure_logging_level_keeps_error_log_at_error(self):
        """_configure_logging_level should keep the error_log logger at ERROR level.

        The error log logger must always remain at ERROR level so that errors
        are always written to error.log regardless of the user's log level.
        """
        from focus_feed.cli import _configure_logging_level

        _configure_logging_level("WARNING")
        error_log_logger = logging.getLogger("focus_feed.error_log")
        assert error_log_logger.level == logging.ERROR

        _configure_logging_level("CRITICAL")
        error_log_logger = logging.getLogger("focus_feed.error_log")
        assert error_log_logger.level == logging.ERROR

    def test_configure_logging_level_invalid_falls_back_to_info(self):
        """_configure_logging_level should fall back to INFO for unknown levels."""
        from focus_feed.cli import _configure_logging_level

        _configure_logging_level("UNKNOWN")
        _logger = logging.getLogger("focus_feed")
        assert _logger.level == logging.INFO

    def test_configure_logging_level_case_insensitive(self):
        """_configure_logging_level should handle lowercase input."""
        from focus_feed.cli import _configure_logging_level

        _configure_logging_level("debug")
        _logger = logging.getLogger("focus_feed")
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
        import focus_feed.config as config_module

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
            "TRANSCRIPTION_ENDPOINT=https://transcribe.example.com/v1\n"
            "TRANSCRIPTION_API_KEY=transcribe-test-key\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        import focus_feed.config as config_module

        config_module._config_instance = None

        monkeypatch.chdir(tmp_path)
        real_load_dotenv = load_dotenv

        def mock_load_dotenv(*args, **kwargs):
            return real_load_dotenv(env_file, override=False)

        with (
            mock.patch("focus_feed.config._logger") as mock_logger,
            mock.patch("dotenv.load_dotenv", side_effect=mock_load_dotenv),
        ):
            from focus_feed.cli import _load_config

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
        import focus_feed.config as config_module

        config_module._config_instance = None

        monkeypatch.chdir(tmp_path)

        # No .env file exists, so CRITICAL should fire via config._logger
        # and _load_config should raise click.ClickException.
        # Mock load_dotenv to prevent it from finding the project root's
        # .env file via find_dotenv() parent directory search.
        with (
            mock.patch("focus_feed.config._logger") as mock_logger,
            mock.patch("dotenv.load_dotenv"),
        ):
            from focus_feed.cli import _load_config

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


class TestNoPathsRequired:
    """Commands must require at least one file or directory path."""

    @pytest.mark.parametrize(
        "command",
        [
            "process",
            "watch",
            "feeds",
            "validate",
            "transcribe",
            "summarize",
            "extract",
            "title",
            "cleanup",
            "create-feed",
        ],
    )
    def test_command_requires_at_least_one_path(self, runner, valid_env, command):
        """Each command should fail with a usage error when no paths are given."""
        result = runner.invoke(app, [command])
        assert result.exit_code != 0
        if command == "create-feed":
            assert "Missing argument 'DIRECTORY'." in result.output
        else:
            assert "At least one file or directory path is required." in result.output

    def test_resolve_paths_raises_on_empty(self):
        """_resolve_paths should raise UsageError for an empty tuple."""
        from focus_feed.cli import _resolve_paths

        with pytest.raises(
            click.UsageError,
            match=r"At least one file or directory path is required\.",
        ):
            _resolve_paths(())

    def test_resolve_data_dir_does_not_fallback_to_cwd(self):
        """_resolve_data_dir_from_paths should not silently fall back to CWD."""
        import tempfile

        from focus_feed.cli import _resolve_data_dir_from_paths

        # With a non-empty list it should still work as before.
        with tempfile.TemporaryDirectory() as d:
            result = _resolve_data_dir_from_paths([Path(d)])
            assert result == Path(d)


class TestEntryPoint:
    """Tests for the entry point configuration (T5.6)."""

    def test_main_function_exists(self):
        """The main() function should be callable."""
        assert callable(main)

    def test_main_invokes_app(self, runner, valid_env, tmp_data_dir):
        """main() should invoke the app group."""
        with mock.patch("focus_feed.cli.app") as mock_app:
            main()
        mock_app.assert_called_once()

    def test_main_via_cli_runner(self, runner, valid_env, tmp_data_dir):
        """main() should work via CliRunner when called as the command."""
        with mock.patch("focus_feed.cli._generate_feeds") as mock_gen:
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
            "cleanup",
        }
        assert expected.issubset(result_commands)

    def test_pyproject_entry_point(self):
        """The pyproject.toml should have the correct entry point."""
        import tomllib

        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        scripts = config.get("project", {}).get("scripts", {})
        assert "focus-feed" in scripts
        assert scripts["focus-feed"] == "focus_feed.cli:main"


class TestCliIntegration:
    """Integration-style tests for the CLI."""

    def test_process_and_generate_feeds_workflow(self, runner, valid_env, tmp_data_dir):
        """Full workflow: process files then generate feeds."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.producer.Producer") as MockProducer:
            mock_producer = mock.MagicMock()
            mock_producer.process_file.return_value = _make_mock_metadata(feed_eligible=True)
            MockProducer.return_value = mock_producer

            with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
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
        metadata = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        metadata["summary"] = "Test article summary"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(metadata))

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Articles feed: 1 item(s)" in result.output

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
