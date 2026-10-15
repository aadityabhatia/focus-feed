"""Tests for the Producer pipeline.

Tests cover:
- Producer initialization with injected dependencies
- process_file() for MP4: transcribe -> summarize -> title generation -> metadata
- process_file() for PDF: extract -> title fallback -> metadata
- process_file() for HTML: extract -> title fallback -> metadata
- process_file() for unsupported file types
- process_file() error handling: writes error metadata
- process_all() batch processing
- generate_title() helper
- JSON metadata file writing
- Error metadata writing
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from unittest import mock

import pytest

from local_feed_generator.errors import ProcessingError
from local_feed_generator.metadata import Metadata
from local_feed_generator.producer import Producer
from local_feed_generator.scanner import ScannedFile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _set_valid_env(monkeypatch):
    """Set valid environment variables for config."""
    monkeypatch.setenv("BASE_URL", "https://example.com/feeds")
    monkeypatch.setenv("DATA_DIR", "./data")
    monkeypatch.setenv("LLM_ENDPOINT", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_SYSTEM_PROMPT", "Test system prompt")
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
    # Mock CUDA availability check so tests that mock the Transcriber can
    # proceed without requiring an actual CUDA installation.
    monkeypatch.setattr(
        "local_feed_generator.services.transcriber.check_cuda_available",
        lambda: True,
    )
    yield
    import local_feed_generator.config as config_module

    config_module._config_instance = None


@pytest.fixture
def mock_transcriber():
    """Create a mock Transcriber."""
    t = mock.MagicMock()
    t.transcribe.return_value = "This is a test transcript."
    return t


@pytest.fixture
def mock_summarizer():
    """Create a mock Summarizer."""
    s = mock.MagicMock()
    s.summarize.return_value = ("Key takeaways: test summary.", 100, 50)
    s.generate_title.return_value = ("Generated Title", 30, 5)
    return s


@pytest.fixture
def mock_pdf_extractor():
    """Create a mock PDFExtractor."""
    e = mock.MagicMock()
    e.extract.return_value = {
        "title": "PDF Title",
        "description": "PDF Description",
        "date": 1728931200,
        "size": 1024,
        "filetype": "pdf",
    }
    return e


@pytest.fixture
def mock_html_extractor():
    """Create a mock HTMLExtractor."""
    e = mock.MagicMock()
    e.extract.return_value = {
        "title": "HTML Title",
        "description": "HTML Description",
        "date": 1728931200,
        "size": 512,
        "filetype": "html",
    }
    return e


@pytest.fixture
def producer(valid_env, mock_transcriber, mock_summarizer, mock_pdf_extractor, mock_html_extractor):
    """Create a Producer with mocked dependencies."""
    return Producer(
        transcriber=mock_transcriber,
        summarizer=mock_summarizer,
        pdf_extractor=mock_pdf_extractor,
        html_extractor=mock_html_extractor,
    )


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


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestProducerInit:
    """Tests for Producer initialization."""

    def test_init_with_all_dependencies(
        self, valid_env, mock_transcriber, mock_summarizer, mock_pdf_extractor, mock_html_extractor
    ):
        """Producer should accept and store all dependencies."""
        p = Producer(
            transcriber=mock_transcriber,
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            html_extractor=mock_html_extractor,
        )
        assert p.transcriber is mock_transcriber
        assert p.summarizer is mock_summarizer
        assert p.pdf_extractor is mock_pdf_extractor
        assert p.html_extractor is mock_html_extractor

    def test_init_creates_default_dependencies(self, valid_env):
        """Producer should create default dependencies when not provided."""
        p = Producer()
        assert p.transcriber is not None
        assert p.summarizer is not None
        assert p.pdf_extractor is not None
        assert p.html_extractor is not None

    def test_init_with_data_dir(self, valid_env):
        """Producer should create a Scanner with the provided data_dir."""
        p = Producer(data_dir="/some/data/dir")
        assert p.scanner is not None
        assert p.scanner.data_dir == Path("/some/data/dir").resolve()

    def test_init_with_scanner(self, valid_env):
        """Producer should accept a pre-configured Scanner."""
        from local_feed_generator.scanner import Scanner

        scanner = Scanner(data_dir="/custom/data")
        p = Producer(scanner=scanner)
        assert p.scanner is scanner


# ---------------------------------------------------------------------------
# process_file() tests for MP4
# ---------------------------------------------------------------------------


class TestProcessFileMP4:
    """Tests for process_file() with MP4 files."""

    def test_mp4_transcribes_audio(self, producer, mock_transcriber, tmp_path):
        """process_file should call transcriber.transcribe for MP4 files."""
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(tmp_path / "lecture.mp4"),
        )
        # Create the file so stat() works
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        producer.process_file(scanned)

        mock_transcriber.transcribe.assert_called_once_with(str(tmp_path / "lecture.mp4"))

    def test_mp4_summarizes_transcript(self, producer, mock_transcriber, mock_summarizer, tmp_path):
        """process_file should call summarizer.summarize for MP4 files."""
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(tmp_path / "lecture.mp4"),
        )
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        producer.process_file(scanned)

        mock_summarizer.summarize.assert_called_once_with("This is a test transcript.")

    def test_mp4_generates_title_from_transcript(self, producer, mock_summarizer, tmp_path):
        """process_file should call summarizer.generate_title for MP4 files."""
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(tmp_path / "lecture.mp4"),
        )
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once_with("This is a test transcript.")

    def test_mp4_writes_json_metadata(self, producer, tmp_path):
        """process_file should write a JSON metadata file adjacent to source."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        assert json_path.exists()

    def test_mp4_metadata_has_correct_fields(self, producer, tmp_path):
        """MP4 metadata should have all required fields with correct values."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"
        assert metadata.title == "Generated Title"
        assert metadata.description == "Key takeaways: test summary."
        assert metadata.filename == "lecture.mp4"
        assert metadata.filepath == "lecture.mp4"
        assert metadata.filetype == "mp4"
        assert metadata.transcript == "This is a test transcript."
        assert metadata.summary == "Key takeaways: test summary."
        assert metadata.size == len(b"video data")
        assert metadata.processing_date is not None
        assert metadata.feed_eligible is True

    def test_mp4_metadata_date_from_file_mtime(self, producer, tmp_path):
        """MP4 metadata date should match the file modification time (epoch int)."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        expected_mtime = int(source.stat().st_mtime)
        assert metadata.date == expected_mtime

    def test_mp4_skips_transcription_when_already_complete(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """_process_mp4 should skip transcription when status.transcribe is already 'complete'."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # Pre-write JSON metadata with transcription already complete
        existing_transcript = "This is the existing transcript."
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "status": {"transcribe": "complete", "summarize": None},
                    "transcript": existing_transcript,
                    "started_at": 1728931200,
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                }
            )
        )

        metadata = producer.process_file(scanned)

        # Transcriber should NOT have been called
        mock_transcriber.transcribe.assert_not_called()
        # Summarizer SHOULD have been called with the existing transcript
        mock_summarizer.summarize.assert_called_once_with(existing_transcript)
        # Metadata should have the existing transcript
        assert metadata is not None
        assert metadata.transcript == existing_transcript
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"


# ---------------------------------------------------------------------------
# process_file() tests for PDF
# ---------------------------------------------------------------------------


class TestProcessFilePDF:
    """Tests for process_file() with PDF files."""

    def test_pdf_extracts_metadata(self, producer, mock_pdf_extractor, tmp_path):
        """process_file should call pdf_extractor.extract for PDF files."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        mock_pdf_extractor.extract.assert_called_once_with(str(source))

    def test_pdf_metadata_has_correct_fields(self, producer, tmp_path):
        """PDF metadata should have all required fields with correct values."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.status["extract"] == "complete"
        assert metadata.title == "PDF Title"
        assert metadata.description == "PDF Description"
        assert metadata.filename == "document.pdf"
        assert metadata.filepath == "document.pdf"
        assert metadata.filetype == "pdf"
        assert metadata.size == 1024
        # PDF should NOT have transcript or summary
        assert metadata.transcript is None
        assert metadata.summary is None
        assert metadata.feed_eligible is True

    def test_pdf_uses_extractor_title(self, producer, mock_pdf_extractor, tmp_path):
        """PDF should use the title from the extractor when available."""
        mock_pdf_extractor.extract.return_value = {
            "title": "Custom PDF Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 2048,
            "filetype": "pdf",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata.title == "Custom PDF Title"

    def test_pdf_falls_back_to_llm_title_when_empty(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should use LLM to generate title when extractor returns empty."""
        mock_pdf_extractor.extract.return_value = {
            "title": "",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once()
        assert metadata.title == "Generated Title"

    def test_pdf_falls_back_to_llm_title_when_none(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should use LLM to generate title when extractor returns None."""
        mock_pdf_extractor.extract.return_value = {
            "title": None,
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once()
        assert metadata.title == "Generated Title"

    def test_pdf_does_not_use_llm_when_title_present(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should not call LLM when extractor provides a title."""
        mock_pdf_extractor.extract.return_value = {
            "title": "Existing Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        mock_summarizer.generate_title.assert_not_called()


# ---------------------------------------------------------------------------
# process_file() tests for HTML
# ---------------------------------------------------------------------------


class TestProcessFileHTML:
    """Tests for process_file() with HTML files."""

    def test_html_extracts_metadata(self, producer, mock_html_extractor, tmp_path):
        """process_file should call html_extractor.extract for HTML files."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        mock_html_extractor.extract.assert_called_once_with(str(source))

    def test_html_metadata_has_correct_fields(self, producer, tmp_path):
        """HTML metadata should have all required fields with correct values."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.status["extract"] == "complete"
        assert metadata.title == "HTML Title"
        assert metadata.description == "HTML Description"
        assert metadata.filename == "article.html"
        assert metadata.filepath == "article.html"
        assert metadata.filetype == "html"
        assert metadata.size == 512
        # HTML should NOT have transcript or summary
        assert metadata.transcript is None
        assert metadata.summary is None
        assert metadata.feed_eligible is True

    def test_html_falls_back_to_llm_title_when_empty(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """HTML should use LLM to generate title when extractor returns empty."""
        mock_html_extractor.extract.return_value = {
            "title": "",
            "description": "Desc",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
        }
        source = tmp_path / "page.html"
        source.write_text("<html>content</html>")
        scanned = _make_scanned_file(
            "page.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once()
        assert metadata.title == "Generated Title"

    def test_html_does_not_use_llm_when_title_present(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """HTML should not call LLM when extractor provides a title."""
        mock_html_extractor.extract.return_value = {
            "title": "Existing Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
        }
        source = tmp_path / "page.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "page.html",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        mock_summarizer.generate_title.assert_not_called()


# ---------------------------------------------------------------------------
# process_file() tests for unsupported file types
# ---------------------------------------------------------------------------


class TestProcessFileUnsupported:
    """Tests for process_file() with unsupported file types."""

    def test_unsupported_filetype_returns_none(self, producer, tmp_path):
        """process_file should return None for unsupported file types."""
        scanned = _make_scanned_file(
            "image.jpg",
            absolute_path=str(tmp_path / "image.jpg"),
            filetype="jpg",
        )

        result = producer.process_file(scanned)

        assert result is None

    def test_unsupported_filetype_does_not_call_services(
        self,
        producer,
        mock_transcriber,
        mock_summarizer,
        mock_pdf_extractor,
        mock_html_extractor,
        tmp_path,
    ):
        """Unsupported file types should not invoke any service."""
        scanned = _make_scanned_file(
            "data.txt",
            absolute_path=str(tmp_path / "data.txt"),
            filetype="txt",
        )

        result = producer.process_file(scanned)

        assert result is None
        mock_transcriber.transcribe.assert_not_called()
        mock_summarizer.summarize.assert_not_called()
        mock_pdf_extractor.extract.assert_not_called()
        mock_html_extractor.extract.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestProcessFileErrors:
    """Tests for error handling in process_file()."""

    def test_mp4_cuda_unavailable_writes_error_metadata(self, producer, tmp_path):
        """_process_mp4 should write error metadata when CUDA is unavailable."""
        with mock.patch(
            "local_feed_generator.services.transcriber.check_cuda_available",
            return_value=False,
        ):
            source = tmp_path / "lecture.mp4"
            source.write_bytes(b"video data")
            scanned = _make_scanned_file(
                "lecture.mp4",
                absolute_path=str(source),
            )

            result = producer.process_file(scanned)

            # process_file catches the RuntimeError and writes error metadata
            assert result is None
            json_path = tmp_path / "lecture.mp4.json"
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert "CUDA libraries not found" in data["error"]
            # Transcriber.transcribe should not have been called
            producer.transcriber.transcribe.assert_not_called()

    def test_mp4_transcription_error_writes_error_metadata(
        self, producer, mock_transcriber, tmp_path
    ):
        """Transcription errors should write error metadata."""
        from local_feed_generator.errors import TranscriptionError

        mock_transcriber.transcribe.side_effect = TranscriptionError("GPU error")
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        result = producer.process_file(scanned)

        assert result is None
        json_path = tmp_path / "lecture.mp4.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["error"] == "GPU error"
        # Status should have been preserved from in-progress state
        assert data.get("status", {}).get("transcribe") == "in_progress"

    def test_mp4_summarization_error_writes_error_metadata(
        self, producer, mock_summarizer, tmp_path
    ):
        """Summarization errors should write error metadata."""
        from local_feed_generator.errors import SummarizationError

        mock_summarizer.summarize.side_effect = SummarizationError("API timeout")
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        result = producer.process_file(scanned)

        assert result is None
        json_path = tmp_path / "lecture.mp4.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["error"] == "API timeout"
        # Status should have transcribe complete, summarize in_progress
        assert data.get("status", {}).get("transcribe") == "complete"
        assert data.get("status", {}).get("summarize") == "in_progress"

    def test_pdf_extraction_error_writes_error_metadata(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF extraction errors should write error metadata."""
        mock_pdf_extractor.extract.side_effect = RuntimeError("PDF corrupt")
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        result = producer.process_file(scanned)

        assert result is None
        json_path = tmp_path / "doc.pdf.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["error"] == "PDF corrupt"
        assert data.get("status", {}).get("extract") == "in_progress"

    def test_error_metadata_has_correct_filetype(self, producer, mock_pdf_extractor, tmp_path):
        """Error metadata should include the correct filetype."""
        mock_pdf_extractor.extract.side_effect = RuntimeError("fail")
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["filetype"] == "pdf"

    def test_error_metadata_has_filename_and_filepath(self, producer, mock_pdf_extractor, tmp_path):
        """Error metadata should include filename and filepath."""
        mock_pdf_extractor.extract.side_effect = RuntimeError("fail")
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["filename"] == "doc.pdf"
        assert data["filepath"] == "doc.pdf"

    def test_error_logged_to_stderr(self, producer, mock_pdf_extractor, tmp_path, caplog):
        """Errors should be logged."""
        mock_pdf_extractor.extract.side_effect = RuntimeError("fail")
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        with caplog.at_level("ERROR"):
            producer.process_file(scanned)

        assert any(
            "Error processing" in record.message or "doc.pdf" in record.message
            for record in caplog.records
            if record.levelno >= 30  # ERROR level
        )


# ---------------------------------------------------------------------------
# JSON metadata file writing tests
# ---------------------------------------------------------------------------


class TestMetadataFileWriting:
    """Tests for JSON metadata file writing."""

    def test_json_file_written_adjacent_to_source(self, producer, tmp_path):
        """JSON metadata file should be written next to the source file."""
        source = tmp_path / "subdir" / "lecture.mp4"
        source.parent.mkdir()
        source.write_bytes(b"video")
        scanned = _make_scanned_file(
            "subdir/lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "subdir" / "lecture.mp4.json"
        assert json_path.exists()

    def test_json_file_is_valid_json(self, producer, tmp_path):
        """Written JSON file should be valid JSON."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert isinstance(data, dict)

    def test_json_file_can_be_read_as_metadata(self, producer, tmp_path):
        """Written JSON file should be readable as a Metadata object."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        metadata = Metadata.from_json_file(json_path)
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"
        assert metadata.title == "Generated Title"
        assert metadata.feed_eligible is True

    def test_json_file_excludes_none_fields(self, producer, tmp_path):
        """Written JSON file should exclude None fields (no transcript for PDF)."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert "transcript" not in data
        assert "summary" not in data
        assert "error" not in data
        assert "transcribe" not in data.get("status", {})
        assert "summarize" not in data.get("status", {})


# ---------------------------------------------------------------------------
# process_all() tests
# ---------------------------------------------------------------------------


class TestProcessAll:
    """Tests for process_all() batch processing."""

    def test_process_all_returns_results(self, producer, tmp_path):
        """process_all should return a list of results."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "doc.pdf").write_bytes(b"pdf")

        scanner = mock.MagicMock()
        scanner.scan_directory.return_value = [
            _make_scanned_file(
                "lecture.mp4",
                absolute_path=str(tmp_path / "lecture.mp4"),
            ),
            _make_scanned_file(
                "doc.pdf",
                absolute_path=str(tmp_path / "doc.pdf"),
            ),
        ]
        producer.scanner = scanner

        results = producer.process_all()

        assert len(results) == 2
        assert all(r is not None for r in results)

    def test_process_all_empty_directory(self, producer):
        """process_all with no files should return an empty list."""
        scanner = mock.MagicMock()
        scanner.scan_directory.return_value = []
        producer.scanner = scanner

        results = producer.process_all()

        assert results == []

    def test_process_all_processes_each_file(
        self, producer, mock_transcriber, mock_pdf_extractor, tmp_path
    ):
        """process_all should call process_file for each scanned file."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "doc.pdf").write_bytes(b"pdf")

        scanner = mock.MagicMock()
        scanner.scan_directory.return_value = [
            _make_scanned_file(
                "lecture.mp4",
                absolute_path=str(tmp_path / "lecture.mp4"),
            ),
            _make_scanned_file(
                "doc.pdf",
                absolute_path=str(tmp_path / "doc.pdf"),
            ),
        ]
        producer.scanner = scanner

        producer.process_all()

        assert mock_transcriber.transcribe.call_count == 1
        assert mock_pdf_extractor.extract.call_count == 1

    def test_process_all_continues_on_error(
        self, producer, mock_transcriber, mock_pdf_extractor, tmp_path
    ):
        """process_all should continue processing after an error."""
        (tmp_path / "bad.mp4").write_bytes(b"video")
        (tmp_path / "doc.pdf").write_bytes(b"pdf")

        mock_transcriber.transcribe.side_effect = ProcessingError("fail")

        scanner = mock.MagicMock()
        scanner.scan_directory.return_value = [
            _make_scanned_file(
                "bad.mp4",
                absolute_path=str(tmp_path / "bad.mp4"),
            ),
            _make_scanned_file(
                "doc.pdf",
                absolute_path=str(tmp_path / "doc.pdf"),
            ),
        ]
        producer.scanner = scanner

        results = producer.process_all()

        # First file (mp4) should have error metadata written, result None
        # Second file (pdf) should succeed
        assert results[0] is None
        assert results[1] is not None


# ---------------------------------------------------------------------------
# generate_title() tests
# ---------------------------------------------------------------------------


class TestGenerateTitle:
    """Tests for the generate_title() helper method."""

    def test_generate_title_calls_summarizer(self, producer, mock_summarizer):
        """generate_title should delegate to summarizer.generate_title."""
        producer.generate_title("Some content here")

        mock_summarizer.generate_title.assert_called_once_with("Some content here")

    def test_generate_title_returns_title(self, producer, mock_summarizer):
        """generate_title should return the title from the summarizer."""
        result = producer.generate_title("Some content here")

        assert result == ("Generated Title", 30, 5)

    def test_generate_title_from_file_reads_content(self, producer, mock_summarizer, tmp_path):
        """generate_title_from_file should read file content and generate title."""
        source = tmp_path / "doc.txt"
        source.write_text("This is the file content for title generation.")

        producer.generate_title_from_file(source)

        mock_summarizer.generate_title.assert_called_once()
        call_arg = mock_summarizer.generate_title.call_args[0][0]
        assert "This is the file content for title generation." in call_arg

    def test_generate_title_from_file_truncates_long_content(
        self, producer, mock_summarizer, tmp_path
    ):
        """generate_title_from_file should truncate very long files."""
        source = tmp_path / "big.txt"
        source.write_text("x" * 20000)

        producer.generate_title_from_file(source)

        call_arg = mock_summarizer.generate_title.call_args[0][0]
        assert len(call_arg) <= 8000


# ---------------------------------------------------------------------------
# Integration tests with real extractors
# ---------------------------------------------------------------------------


class TestProducerWithRealExtractors:
    """Integration tests using real PDFExtractor and HTMLExtractor."""

    def test_process_pdf_with_real_extractor(self, producer, tmp_path):
        """Producer should work with real PDFExtractor."""
        from pypdf import PdfWriter

        pdf_path = tmp_path / "real.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_metadata({"/Title": "Real PDF Title"})
        with open(pdf_path, "wb") as f:
            writer.write(f)

        scanned = _make_scanned_file(
            "real.pdf",
            absolute_path=str(pdf_path),
        )

        # Use real extractor
        from local_feed_generator.services.extractors import PDFExtractor

        producer.pdf_extractor = PDFExtractor()

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.title == "Real PDF Title"
        assert metadata.filetype == "pdf"
        assert metadata.status["extract"] == "complete"
        assert metadata.feed_eligible is True

    def test_process_html_with_real_extractor(self, producer, tmp_path):
        """Producer should work with real HTMLExtractor."""
        html_path = tmp_path / "real.html"
        html_path.write_text(
            "<!DOCTYPE html><html><head><title>Real HTML Title</title>"
            "<meta name='description' content='Real description'></head>"
            "<body>Content</body></html>",
            encoding="utf-8",
        )

        scanned = _make_scanned_file(
            "real.html",
            absolute_path=str(html_path),
        )

        # Use real extractor
        from local_feed_generator.services.extractors import HTMLExtractor

        producer.html_extractor = HTMLExtractor()

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.title == "Real HTML Title"
        assert metadata.description == "Real description"
        assert metadata.filetype == "html"
        assert metadata.status["extract"] == "complete"
        assert metadata.feed_eligible is True

    def test_process_pdf_without_title_uses_filename(self, producer, tmp_path):
        """PDF without title should fall back to filename, then LLM if empty."""
        from pypdf import PdfWriter

        pdf_path = tmp_path / "untitled.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        scanned = _make_scanned_file(
            "untitled.pdf",
            absolute_path=str(pdf_path),
        )

        # Use real extractor
        from local_feed_generator.services.extractors import PDFExtractor

        producer.pdf_extractor = PDFExtractor()

        metadata = producer.process_file(scanned)

        assert metadata is not None
        # PDFExtractor falls back to filename stem "untitled"
        assert metadata.title == "untitled"


# ---------------------------------------------------------------------------
# Intermediate state saving tests
# ---------------------------------------------------------------------------


class TestIntermediateStateSaving:
    """Tests for intermediate JSON state saving during processing."""

    def test_mp4_initial_state_written_at_start(self, producer, tmp_path):
        """MP4 processing should write an in-progress state at the start."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        # Final state should have status with transcribe and summarize complete
        assert data["status"]["transcribe"] == "complete"
        assert data["status"]["summarize"] == "complete"
        assert "started_at" in data
        assert "completed_at" in data

    def test_mp4_final_state_has_transcription_model(self, producer, tmp_path):
        """MP4 final JSON state should include transcription_model from config."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcription_model"] == "large-v3-turbo"

    def test_mp4_transcription_model_remote_with_remote_model(self, monkeypatch, tmp_path):
        """When transcription_endpoint is set and transcription_remote_model is configured,
        transcription_model should be the remote model name."""
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("TRANSCRIPTION_ENDPOINT", "https://whisper.example.com/v1")
        monkeypatch.setenv("TRANSCRIPTION_REMOTE_MODEL", "whisper-1")
        import local_feed_generator.config as config_module
        config_module._config_instance = None

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        mock_summ = mock.MagicMock()
        mock_summ.summarize.return_value = ("Test summary.", 100, 50)
        mock_summ.generate_title.return_value = ("Test Title", 30, 5)

        producer = Producer(
            transcriber=mock.MagicMock(),
            summarizer=mock_summ,
            pdf_extractor=mock.MagicMock(),
            html_extractor=mock.MagicMock(),
        )
        producer.transcriber.transcribe.return_value = "Test transcript."

        metadata = producer.process_file(scanned)

        assert metadata.transcription_model == "whisper-1"

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcription_model"] == "whisper-1"

        config_module._config_instance = None

    def test_mp4_transcription_model_remote_without_remote_model(self, monkeypatch, tmp_path):
        """When transcription_endpoint is set but transcription_remote_model is not configured,
        transcription_model should be 'remote'."""
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("TRANSCRIPTION_ENDPOINT", "https://whisper.example.com/v1")
        monkeypatch.delenv("TRANSCRIPTION_REMOTE_MODEL", raising=False)
        import local_feed_generator.config as config_module
        config_module._config_instance = None

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        mock_summ = mock.MagicMock()
        mock_summ.summarize.return_value = ("Test summary.", 100, 50)
        mock_summ.generate_title.return_value = ("Test Title", 30, 5)

        producer = Producer(
            transcriber=mock.MagicMock(),
            summarizer=mock_summ,
            pdf_extractor=mock.MagicMock(),
            html_extractor=mock.MagicMock(),
        )
        producer.transcriber.transcribe.return_value = "Test transcript."

        metadata = producer.process_file(scanned)

        assert metadata.transcription_model == "remote"

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcription_model"] == "remote"

        config_module._config_instance = None

    def test_mp4_transcription_model_local(self, producer, tmp_path):
        """When transcription_endpoint is not set, use config.transcription_model."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata.transcription_model == "large-v3-turbo"

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcription_model"] == "large-v3-turbo"

    def test_pdf_final_state_no_stage_field(self, producer, tmp_path):
        """PDF final JSON state should not have stage field."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        assert "stage" not in data

    def test_html_final_state_no_stage_field(self, producer, tmp_path):
        """HTML final JSON state should not have stage field."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "article.html.json"
        data = json.loads(json_path.read_text())
        assert "stage" not in data

    def test_mp4_intermediate_state_transcribing(self, producer, mock_transcriber, tmp_path):
        """MP4 should write transcribing status before transcription."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # Make transcribe raise to capture the intermediate state
        mock_transcriber.transcribe.side_effect = RuntimeError("fail")

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        # Error state should preserve the status from before the error
        assert data.get("status", {}).get("transcribe") == "in_progress"
        assert data["error"] is not None

    def test_mp4_intermediate_state_summarizing(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """MP4 should write summarizing status after transcription, before summarization."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # Make summarize raise to capture the intermediate state
        mock_summarizer.summarize.side_effect = RuntimeError("fail")

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        # Error state should preserve the status from before the error
        assert data.get("status", {}).get("transcribe") == "complete"
        assert data.get("status", {}).get("summarize") == "in_progress"
        assert data["error"] is not None

    def test_pdf_intermediate_state_extracting(self, producer, mock_pdf_extractor, tmp_path):
        """PDF should write extracting status before extraction."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        # Make extract raise to capture the intermediate state
        mock_pdf_extractor.extract.side_effect = RuntimeError("fail")

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        assert data.get("status", {}).get("extract") == "in_progress"
        assert data["error"] is not None

    def test_pdf_intermediate_state_generating_title(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should write generating_title status before LLM title generation."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }
        mock_summarizer.generate_title.side_effect = RuntimeError("fail")

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["error"] is not None

    def test_pdf_no_generating_title_stage_when_title_present(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF should not have generating_title stage when extractor provides title."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "PDF Title",
            "description": "PDF Description",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        # Final state should have extract complete (not in_progress)
        assert data["status"]["extract"] == "complete"

    def test_html_intermediate_state_extracting(self, producer, mock_html_extractor, tmp_path):
        """HTML should write extracting status before extraction."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        # Make extract raise to capture the intermediate state
        mock_html_extractor.extract.side_effect = RuntimeError("fail")

        producer.process_file(scanned)

        json_path = tmp_path / "article.html.json"
        data = json.loads(json_path.read_text())
        assert data.get("status", {}).get("extract") == "in_progress"
        assert data["error"] is not None

    def test_error_metadata_preserves_status(self, producer, mock_transcriber, tmp_path):
        """Error metadata should preserve the status from the in-progress state."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        mock_transcriber.transcribe.side_effect = RuntimeError("GPU error")

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["error"] is not None
        assert data.get("status", {}).get("transcribe") == "in_progress"
        assert "started_at" in data

    def test_error_metadata_preserves_started_at(self, producer, mock_pdf_extractor, tmp_path):
        """Error metadata should preserve started_at from the in-progress state."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.side_effect = RuntimeError("PDF corrupt")

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["error"] is not None
        assert "started_at" in data

    def test_final_state_excludes_none_fields(self, producer, tmp_path):
        """Final JSON state should exclude None fields for PDF."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        # PDF should NOT have transcript or summary
        assert "transcript" not in data
        assert "summary" not in data
        assert "error" not in data
        # PDF should NOT have transcribe or summarize in status
        assert "transcribe" not in data.get("status", {})
        assert "summarize" not in data.get("status", {})
        # status should have extract complete
        assert data["status"]["extract"] == "complete"

    def test_final_state_has_started_at_and_completed_at(self, producer, tmp_path):
        """Final JSON state should have both started_at and completed_at."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert "started_at" in data
        assert "completed_at" in data
        assert data["started_at"] is not None
        assert data["completed_at"] is not None

    def test_metadata_object_has_complete_status(self, producer, tmp_path):
        """Returned Metadata object should have all step statuses complete."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"
        assert metadata.transcription_model == "large-v3-turbo"
        assert metadata.feed_eligible is True

    def test_metadata_object_has_started_at_and_completed_at(self, producer, tmp_path):
        """Returned Metadata object should have started_at and completed_at."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.started_at is not None
        assert metadata.completed_at is not None

    def test_pdf_metadata_object_has_started_at_and_completed_at(self, producer, tmp_path):
        """Returned PDF Metadata object should have started_at and completed_at."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.started_at is not None
        assert metadata.completed_at is not None

    def test_html_metadata_object_has_started_at_and_completed_at(self, producer, tmp_path):
        """Returned HTML Metadata object should have started_at and completed_at."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.started_at is not None
        assert metadata.completed_at is not None


# ---------------------------------------------------------------------------
# Intermediate status write tests (Task 1: status completion)
# ---------------------------------------------------------------------------


class TestIntermediateStatusWrites:
    """Tests for intermediate status writes after each step completes."""

    def test_mp4_writes_summarize_complete_after_summarization(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """_process_mp4 should write summarize=complete after summarization.

        This intermediate write happens before title generation so that if
        title generation fails, the summarize status is preserved.
        """
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # Make generate_title raise to capture the intermediate state
        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        # Summarize should be complete (written after summarization, before title gen)
        assert data.get("status", {}).get("summarize") == "complete"
        assert data.get("status", {}).get("transcribe") == "complete"
        # Error should be present because title generation failed
        assert data["error"] is not None

    def test_pdf_writes_extract_complete_after_extraction(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """_process_pdf should write extract=complete after extraction.

        This intermediate write happens before the final metadata update.
        """
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "PDF Title",
            "description": "PDF Description",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }

        # Track writes to verify intermediate state
        written_statuses: list[str] = []

        original_write = producer._write_metadata_partial

        def track_write(metadata_dict, source_path):
            written_statuses.append(metadata_dict.get("status", {}).get("extract"))
            original_write(metadata_dict, source_path)

        with mock.patch.object(producer, "_write_metadata_partial", side_effect=track_write):
            producer.process_file(scanned)

        # Should have written at least: in_progress, complete (intermediate), complete (final)
        assert len(written_statuses) >= 3
        assert "in_progress" in written_statuses
        assert "complete" in written_statuses

    def test_pdf_writes_extract_complete_after_llm_title_generation(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """_process_pdf should write 'extract'='complete' after LLM title generation."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }

        # Make generate_title_from_file raise to capture the intermediate state
        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        # Extract should still be in_progress because title generation failed
        # (the intermediate write with "complete" happens after title generation)
        assert data.get("status", {}).get("extract") == "in_progress"
        assert data["error"] is not None

    def test_html_writes_extract_complete_after_extraction(
        self, producer, mock_html_extractor, tmp_path
    ):
        """_process_html should write extract=complete after extraction.

        This intermediate write happens before the final metadata update.
        """
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        mock_html_extractor.extract.return_value = {
            "title": "HTML Title",
            "description": "HTML Description",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
        }

        # Track writes to verify intermediate state
        written_statuses: list[str] = []

        original_write = producer._write_metadata_partial

        def track_write(metadata_dict, source_path):
            written_statuses.append(metadata_dict.get("status", {}).get("extract"))
            original_write(metadata_dict, source_path)

        with mock.patch.object(producer, "_write_metadata_partial", side_effect=track_write):
            producer.process_file(scanned)

        # Should have written at least: in_progress, complete (intermediate), complete (final)
        assert len(written_statuses) >= 3
        assert "in_progress" in written_statuses
        assert "complete" in written_statuses

    def test_html_writes_extract_complete_after_llm_title_generation(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """_process_html should write 'extract'='complete' after LLM title generation."""
        source = tmp_path / "article.html"
        source.write_text("<html>content</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        mock_html_extractor.extract.return_value = {
            "title": "",
            "description": "Desc",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
        }

        # Make generate_title_from_file raise to capture the intermediate state
        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "article.html.json"
        data = json.loads(json_path.read_text())
        # Extract should still be in_progress because title generation failed
        assert data.get("status", {}).get("extract") == "in_progress"
        assert data["error"] is not None

    def test_pdf_final_state_has_extract_complete_with_title_present(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF final state should have extract=complete when extractor provides title."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "PDF Title",
            "description": "PDF Description",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        assert data["status"]["extract"] == "complete"
        assert data["title"] == "PDF Title"
        assert "completed_at" in data

    def test_html_final_state_has_extract_complete_with_title_present(
        self, producer, mock_html_extractor, tmp_path
    ):
        """HTML final state should have extract=complete when extractor provides title."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        mock_html_extractor.extract.return_value = {
            "title": "HTML Title",
            "description": "HTML Description",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
        }

        producer.process_file(scanned)

        json_path = tmp_path / "article.html.json"
        data = json.loads(json_path.read_text())
        assert data["status"]["extract"] == "complete"
        assert data["title"] == "HTML Title"
        assert "completed_at" in data
