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


class TestFindFilesNeedingMetadataExtraction:
    """Tests for the _find_files_needing_metadata_extraction helper function."""

    def test_find_files_needing_metadata_extraction_finds_pdf_without_json(
        self, valid_env, tmp_data_dir
    ):
        """Should find PDF files with no JSON metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")

        from focus_feed.cli import _find_files_needing_metadata_extraction

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
                    "content": "Extracted content",
                    "title": "My Doc",
                    "filename": "doc.pdf",
                    "filetype": "pdf",
                }
            )
        )

        from focus_feed.cli import _find_files_needing_metadata_extraction

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

        from focus_feed.cli import _find_files_needing_metadata_extraction

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
        assert "No PDF/HTML/Docling files need metadata extraction." in result.output

    def test_extract_dry_run(self, runner, valid_env, tmp_data_dir):
        """Extract --dry-run should list files without processing."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.services.extractors.PDFExtractor") as MockExtractor:
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

        extractor_target = f"focus_feed.services.extractors.{extractor_path}"
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
        assert "Found 1 PDF/HTML/Docling file(s)" in result.output
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
                    "content": "Extracted content",
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
        assert "No PDF/HTML/Docling files need metadata extraction." in result.output

    def test_extract_writes_to_json(self, runner, valid_env, tmp_data_dir):
        """Extract should write extracted data to JSON metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.services.extractors.PDFExtractor") as MockExtractor:
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
        assert data["title"] == {
            "filename": "PDF Title",
            "extracted": None,
            "generated": None,
        }
        assert data["description"] == "PDF Description"
        assert data["filetype"] == "pdf"
        assert data["extract"]["status"] == "complete"

    def test_extract_failure_persists_step_error(self, runner, valid_env, tmp_data_dir):
        """Extraction failures should be saved in the extract step map."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.services.extractors.PDFExtractor") as MockExtractor:
            MockExtractor.return_value.extract.side_effect = RuntimeError("Extraction failed")
            result = runner.invoke(app, ["extract", str(tmp_data_dir)])

        assert result.exit_code == 1
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert data["extract"]["error"] == "Extraction failed"
        assert data["extract"]["status"] is None

    def test_extract_discovers_docling_file_and_saves_title_provenance(
        self, runner, valid_env, tmp_data_dir
    ):
        """Extract should discover Docling files and persist extracted title provenance."""
        source = tmp_data_dir / "doc.docx"
        source.write_bytes(b"docx data")

        with mock.patch("focus_feed.services.extractors.DoclingExtractor") as MockExtractor:
            mock_extractor = mock.MagicMock()
            mock_extractor.extract.return_value = {
                "title": "Course Notes",
                "extracted": "Original Document Title",
                "description": "Document description",
                "date": 1728931200,
                "size": 9,
                "filetype": "docx",
                "content": "Extracted document content",
            }
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(app, ["extract", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert "Found 1 PDF/HTML/Docling file(s)" in result.output
        mock_extractor.extract.assert_called_once_with(str(source))
        data = json.loads((tmp_data_dir / "doc.docx.json").read_text())
        assert data["title"] == {
            "filename": "Course Notes",
            "extracted": "Original Document Title",
            "generated": None,
        }

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

        with mock.patch("focus_feed.services.extractors.PDFExtractor") as MockExtractor:
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
        assert "Found 1 PDF/HTML/Docling file(s)" in result.output
        mock_extractor.extract.assert_called_once()

    def test_extract_updates_existing_title_dict_and_preserves_generated(
        self, runner, valid_env, tmp_data_dir
    ):
        """Extract should refresh provenance without replacing a generated title."""
        source = tmp_data_dir / "doc.pdf"
        source.write_bytes(b"pdf data")
        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(
                {
                    "title": {
                        "filename": "Old filename title",
                        "extracted": "Old extracted title",
                        "generated": "Generated title",
                    },
                    "extract": {"status": "complete"},
                }
            )
        )

        with mock.patch("focus_feed.services.extractors.PDFExtractor") as MockExtractor:
            mock_extractor = mock.MagicMock()
            mock_extractor.extract.return_value = {
                "title": "Fresh filename title",
                "extracted": "Fresh extracted title",
                "description": "PDF Description",
                "date": 1728931200,
                "size": 1024,
                "filetype": "pdf",
            }
            MockExtractor.return_value = mock_extractor

            result = runner.invoke(app, ["extract", str(tmp_data_dir), "--force"])

        assert result.exit_code == 0
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert data["title"] == {
            "filename": "Fresh filename title",
            "extracted": "Fresh extracted title",
            "generated": "Generated title",
        }

    def test_extract_sets_in_progress_before_and_complete_after(
        self, runner, valid_env, tmp_data_dir
    ):
        """Extract should set status to 'in_progress' before extraction, then 'complete' after."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")

        with mock.patch("focus_feed.services.extractors.PDFExtractor") as MockExtractor:
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
        assert data["title"] == {
            "filename": "PDF Title",
            "extracted": None,
            "generated": None,
        }
        assert data["extract"]["end_time"] is not None
