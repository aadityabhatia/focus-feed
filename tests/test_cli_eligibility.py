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

from pathlib import Path
from unittest import mock

from click.testing import CliRunner
import pytest

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


class TestIsDictFeedEligible:
    """Tests for the _is_dict_feed_eligible helper function."""

    def test_pdf_eligible_with_extract_and_summarize(self, valid_env):
        """PDF should be eligible if both extract and summarize are complete."""
        from focus_feed.scanner import _is_dict_feed_eligible

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
        from focus_feed.scanner import _is_dict_feed_eligible

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
        from focus_feed.scanner import _is_dict_feed_eligible

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
        from focus_feed.scanner import _is_dict_feed_eligible

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
        from focus_feed.scanner import _is_dict_feed_eligible

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
        from focus_feed.scanner import _is_dict_feed_eligible

        metadata = {
            "filetype": "pdf",
            "title": "Test",
            "filename": "doc.pdf",
            "filepath": "doc.pdf",
            "size": 1024,
        }
        assert _is_dict_feed_eligible(metadata, "pdf") is False
