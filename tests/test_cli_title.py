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
from typing import Any
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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
        # Verify generate_title step map is tracked
        assert "generate_title" in data
        assert data["generate_title"]["status"] == "complete"
        assert data["generate_title"]["model"] == "test-model"
        assert data["generate_title"]["input_tokens"] == 30
        assert data["generate_title"]["output_tokens"] == 5

    def test_title_fills_filename_and_uses_millisecond_timestamps(
        self, runner, valid_env, tmp_data_dir
    ):
        """Title updates fill title.filename and use the shared timestamp helper."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        json_path = tmp_data_dir / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "title": {"filename": "", "extracted": None, "generated": None},
                    "filename": "lecture.mp4",
                    "filetype": "mp4",
                    "content": "Transcript content",
                    "transcribe": {"status": "complete"},
                    "summarize": {"status": "complete"},
                }
            )
        )

        with (
            mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer,
            mock.patch("focus_feed.cli.now_timestamp", side_effect=[123.456, 123.789]),
        ):
            MockSummarizer.return_value.generate_title.return_value = ("Lecture Title", 10, 5)
            result = runner.invoke(app, ["title", str(tmp_data_dir)])

        assert result.exit_code == 0
        finalized = json.loads(json_path.read_text())
        assert finalized["title"]["filename"] == "lecture"
        assert finalized["generate_title"]["start_time"] == 123.456
        assert finalized["generate_title"]["end_time"] == 123.789

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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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
                    "title": {
                        "filename": "doc",
                        "extracted": "Extracted title",
                        "generated": "Existing Title",
                    },
                    "description": "",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1024,
                }
            )
        )

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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
        call_kwargs = mock_summarizer.generate_title.call_args.kwargs
        assert call_kwargs["extracted"] == "Extracted title"

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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.side_effect = RuntimeError("LLM failed")
            MockSummarizer.return_value = mock_summarizer

            result = runner.invoke(
                app,
                ["title", str(tmp_data_dir)],
            )

        assert result.exit_code == 1
        assert "1 failed" in result.output

    def test_title_failure_persists_step_error(self, runner, valid_env, tmp_data_dir):
        """Title-generation failures should be saved in the generate_title step map."""
        source = tmp_data_dir / "doc.pdf"
        source.write_bytes(b"pdf data content")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "title": {"filename": "", "extracted": None, "generated": None},
                    "filename": "doc.pdf",
                    "filetype": "pdf",
                    "content": "Document content",
                    "extract": {"status": "complete"},
                }
            )
        )

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            MockSummarizer.return_value.generate_title.side_effect = RuntimeError("Title failed")
            result = runner.invoke(app, ["title", str(tmp_data_dir)])

        assert result.exit_code == 1
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert data["generate_title"]["error"] == "Title failed"
        assert data["generate_title"]["status"] is None

    def test_transcribe_and_summarize_output_loads_as_metadata(
        self, runner, valid_env, tmp_data_dir
    ):
        """Granular command output includes required title and size metadata."""
        source = tmp_data_dir / "lecture.mp4"
        source.write_bytes(b"video data")

        with mock.patch("focus_feed.services.transcriber.Transcriber") as MockTranscriber:
            MockTranscriber.return_value.transcribe.return_value = "Transcript content"
            transcribe_result = runner.invoke(app, ["transcribe", str(tmp_data_dir)])
        assert transcribe_result.exit_code == 0

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            MockSummarizer.return_value.summarize.return_value = ("Key takeaways", 10, 5)
            summarize_result = runner.invoke(app, ["summarize", str(tmp_data_dir)])
        assert summarize_result.exit_code == 0

        metadata = Metadata.from_json_file(tmp_data_dir / "lecture.mp4.json")
        assert isinstance(metadata.title, dict)
        assert metadata.title["filename"] == "lecture"
        assert metadata.size == source.stat().st_size

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

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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

        from focus_feed.cli import _find_files_needing_title

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
            ("m4a", {"content": "This is the audio transcript."}, "This is the audio transcript."),
            ("mp4", {}, ""),
            ("txt", {}, ""),
        ],
    )
    def test_get_content_for_title_basic(
        self, valid_env, tmp_path, filetype, metadata_dict, expected
    ):
        """Should return content from metadata for media types, empty for unsupported types."""
        from focus_feed.cli import _get_content_for_title

        if filetype in ("mp4", "m4a"):
            file_path = Path(f"/fake/path/lecture.{filetype}")
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
        from focus_feed.cli import _get_content_for_title

        source = tmp_path / f"file.{filetype}"
        source.write_text(content)
        metadata_dict: dict[str, Any] = {}
        result = _get_content_for_title(source, metadata_dict)
        assert content in result

    def test_get_content_for_title_truncates_long_content(self, valid_env, tmp_path):
        """Should truncate content to 8000 characters for PDF/HTML files."""
        from focus_feed.cli import _get_content_for_title

        source = tmp_path / "doc.pdf"
        source.write_text("x" * 20000)
        metadata_dict: dict[str, Any] = {}
        result = _get_content_for_title(source, metadata_dict)
        assert len(result) <= 8000

    def test_get_content_for_title_pdf_handles_read_error(self, valid_env, tmp_path):
        """Should return empty string when file cannot be read."""
        from focus_feed.cli import _get_content_for_title

        source = tmp_path / "doc.pdf"
        source.write_text("content")
        # Mock the file read to raise an error (avoids chmod which is platform-dependent)
        with mock.patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
            metadata_dict: dict[str, Any] = {}
            result = _get_content_for_title(source, metadata_dict)
        assert result == ""
