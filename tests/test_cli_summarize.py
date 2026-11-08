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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

    def test_summarize_failure_persists_step_error(self, runner, valid_env, tmp_data_dir):
        """Summarization failures should be saved in the summarize step map."""
        source = tmp_data_dir / "lecture.mp4"
        source.write_bytes(b"video data")
        (tmp_data_dir / "lecture.mp4.json").write_text(
            json.dumps(
                {
                    "title": {"filename": "lecture", "extracted": None, "generated": None},
                    "filename": "lecture.mp4",
                    "filetype": "mp4",
                    "content": "Transcript content",
                    "transcribe": {"status": "complete"},
                    "summarize": {"status": None},
                }
            )
        )

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            MockSummarizer.return_value.summarize.side_effect = RuntimeError("Summarization failed")
            result = runner.invoke(app, ["summarize", str(tmp_data_dir)])

        assert result.exit_code == 1
        data = json.loads((tmp_data_dir / "lecture.mp4.json").read_text())
        assert data["summarize"]["error"] == "Summarization failed"
        assert data["summarize"]["status"] is None

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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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
                "lecture.m4a",
                "m4a",
                "content",
                "Some audio transcript",
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

        from focus_feed.cli import _find_files_needing_summary

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

        from focus_feed.cli import _find_files_needing_summary

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

        from focus_feed.cli import _find_files_needing_summary

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

        from focus_feed.cli import _find_files_needing_summary

        result = _find_files_needing_summary(tmp_data_dir, force=True)
        assert len(result) == 1

    def test_find_files_needing_summary_skips_no_json(self, valid_env, tmp_data_dir):
        """Should skip files without JSON metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")

        from focus_feed.cli import _find_files_needing_summary

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

        from focus_feed.cli import _find_files_needing_summary

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

        from focus_feed.cli import _find_files_needing_summary

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
