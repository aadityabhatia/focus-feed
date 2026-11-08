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
