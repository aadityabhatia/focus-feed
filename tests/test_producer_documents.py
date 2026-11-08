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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from focus_feed.errors import ProcessingError
from focus_feed.producer import Producer
from focus_feed.scanner import ScannedFile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        "title": "document",
        "extracted": "PDF Title",
        "description": "PDF Description",
        "date": 1728931200,
        "size": 1024,
        "filetype": "pdf",
        "content": "PDF plaintext content",
    }
    return e


@pytest.fixture
def mock_html_extractor():
    """Create a mock HTMLExtractor."""
    e = mock.MagicMock()
    e.extract.return_value = {
        "title": "article",
        "extracted": "HTML Title",
        "description": "HTML Description",
        "date": 1728931200,
        "size": 512,
        "filetype": "html",
        "content": "HTML plaintext content",
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


class TestProcessFilePDF:
    """Tests for process_file() with PDF files."""

    @pytest.mark.parametrize(
        "docling_result,docling_error",
        [
            ({"content": "Docling Markdown", "title": "doc"}, None),
            ({"content": None}, None),
            (None, ConnectionError("unavailable")),
            (None, RuntimeError("service failure")),
            (None, ValueError("invalid response")),
        ],
    )
    def test_pdf_uses_docling_and_never_falls_back_when_configured(
        self,
        producer,
        mock_pdf_extractor,
        tmp_path,
        docling_result,
        docling_error,
    ):
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        producer.docling_extractor = mock.MagicMock()
        producer.docling_extractor.extract.return_value = docling_result
        producer.docling_extractor.extract.side_effect = docling_error
        with mock.patch(
            "focus_feed.producer.get_config",
            return_value=SimpleNamespace(
                docling_api="https://docling.example.com", llm_model="test-model"
            ),
        ):
            metadata = producer.process_file(_make_scanned_file("document.pdf", str(source)))

        producer.docling_extractor.extract.assert_called_once_with(str(source))
        mock_pdf_extractor.extract.assert_not_called()
        if docling_error or not (docling_result and docling_result.get("content")):
            assert metadata is None
        else:
            assert metadata is not None

    def test_pdf_skips_docling_when_not_configured(self, producer, mock_pdf_extractor, tmp_path):
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        producer.docling_extractor = mock.MagicMock()
        with mock.patch(
            "focus_feed.producer.get_config",
            return_value=SimpleNamespace(docling_api=None, llm_model="test-model"),
        ):
            producer.process_file(_make_scanned_file("document.pdf", str(source)))

        producer.docling_extractor.extract.assert_not_called()
        mock_pdf_extractor.extract.assert_called_once_with(str(source))

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
        assert metadata.extract.status == "complete"
        assert metadata.final_title == "Generated Title"  # LLM-generated title
        assert metadata.description == "PDF Description"
        assert metadata.filename == "document.pdf"
        assert metadata.source_filepath == "document.pdf"
        assert metadata.filetype == "pdf"
        assert metadata.size == 1024
        # PDF should NOT have content from MP4 fields
        assert metadata.content == "PDF plaintext content"
        assert metadata.feed_eligible is True

    def test_pdf_uses_filename_title_not_extractor_title(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF should use filename-derived title when no content is available for LLM."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted": "Custom PDF Title",
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

        assert metadata.title.get("extracted") == "Custom PDF Title"
        assert metadata.final_title == "doc"

    def test_pdf_calls_llm_for_title_when_content_available(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should call LLM for title generation when content is available."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted": "PDF Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
            "content": "PDF plaintext content",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        # LLM should be called for title generation on PDF files with content
        mock_summarizer.generate_title.assert_called_once()
        assert metadata.final_title == "Generated Title"

    @pytest.mark.parametrize(
        "content_present,should_call_llm",
        [
            ("Some content here", True),
            ("", False),
            (None, False),
        ],
    )
    def test_pdf_title_generation_logic(
        self,
        producer,
        mock_pdf_extractor,
        mock_summarizer,
        tmp_path,
        content_present,
        should_call_llm,
    ):
        """PDF should call LLM for title only when content is available."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted": "PDF Title" if content_present else None,
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
            "content": content_present,
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        if should_call_llm:
            mock_summarizer.generate_title.assert_called_once()
        else:
            mock_summarizer.generate_title.assert_not_called()

    def test_pdf_always_calls_llm_when_title_present(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should call LLM for title generation even when extractor provides a title."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted": "Existing Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
            "content": "Some content here",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once()

    def test_pdf_content_none_when_extractor_returns_no_content(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF should have content=None when extractor doesn't return content."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted": None,
            "description": "PDF Description",
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

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.content is None


class TestProcessFileDocling:
    """Tests for document formats that require Docling extraction."""

    @pytest.mark.parametrize("filetype", ["docx", "doc", "pptx", "ppt", "xlsx", "xls"])
    def test_docling_formats_use_docling_pipeline(
        self, producer, mock_summarizer, mock_pdf_extractor, filetype, tmp_path
    ):
        source = tmp_path / f"document.{filetype}"
        source.write_bytes(b"document data")
        producer.docling_extractor = mock.MagicMock()
        producer.docling_extractor.extract.return_value = {
            "title": "document",
            "extracted": None,
            "description": None,
            "date": 1728931200,
            "size": len(b"document data"),
            "filetype": filetype,
            "content": "Docling extracted content",
        }

        with mock.patch(
            "focus_feed.producer.get_config",
            return_value=SimpleNamespace(
                docling_api="https://docling.example.com", llm_model="test-model"
            ),
        ):
            metadata = producer.process_file(_make_scanned_file(source.name, str(source)))

        assert metadata is not None
        assert metadata.filetype == filetype
        assert metadata.content == "Docling extracted content"
        producer.docling_extractor.extract.assert_called_once_with(str(source))
        mock_pdf_extractor.extract.assert_not_called()
        mock_summarizer.summarize.assert_called_once_with(
            "Docling extracted content",
            reflection_questions=None,
            filepath=source.name,
            filetype=filetype,
        )

    def test_docling_format_without_api_records_extraction_error(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        source = tmp_path / "document.docx"
        source.write_bytes(b"document data")
        producer.docling_extractor = mock.MagicMock()

        with mock.patch(
            "focus_feed.producer.get_config",
            return_value=SimpleNamespace(docling_api=None, llm_model="test-model"),
        ):
            metadata = producer.process_file(_make_scanned_file(source.name, str(source)))

        assert metadata is None
        producer.docling_extractor.extract.assert_not_called()
        mock_pdf_extractor.extract.assert_not_called()
        error_data = json.loads((tmp_path / "document.docx.json").read_text())
        assert error_data["extract"]["error"] == "DOCLING_API is not configured"


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
        assert metadata.extract.status == "complete"
        assert metadata.final_title == "Generated Title"  # LLM-generated title
        assert metadata.description == "HTML Description"
        assert metadata.filename == "article.html"
        assert metadata.source_filepath == "article.html"
        assert metadata.filetype == "html"
        assert metadata.size == 512
        # HTML should have content from the extractor
        assert metadata.content == "HTML plaintext content"
        assert metadata.feed_eligible is True

    def test_html_calls_llm_for_title_when_content_available(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """HTML should call LLM for title generation when content is available."""
        mock_html_extractor.extract.return_value = {
            "title": "",
            "extracted": None,
            "description": "Desc",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
            "content": "HTML plaintext content",
        }
        source = tmp_path / "page.html"
        source.write_text("<html>content</html>")
        scanned = _make_scanned_file(
            "page.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        # LLM should be called for title generation on HTML files with content
        mock_summarizer.generate_title.assert_called_once()
        assert metadata.final_title == "Generated Title"

    def test_html_does_not_call_llm_for_title_when_no_content(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """HTML should NOT call LLM for title when no content is available."""
        mock_html_extractor.extract.return_value = {
            "title": "",
            "extracted": None,
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

        # LLM should not be called when no content is available for title generation
        mock_summarizer.generate_title.assert_not_called()
        assert metadata.final_title == "page"

    def test_html_always_calls_llm_when_title_present(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """HTML should call LLM for title generation even when extractor provides a title."""
        mock_html_extractor.extract.return_value = {
            "title": "page",
            "extracted": "Existing Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
            "content": "Some content here",
        }
        source = tmp_path / "page.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "page.html",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once()

    def test_html_content_none_when_extractor_returns_no_content(
        self, producer, mock_html_extractor, tmp_path
    ):
        """HTML should have content=None when extractor doesn't return content."""
        mock_html_extractor.extract.return_value = {
            "title": "article",
            "extracted": None,
            "description": "HTML Description",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
        }
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.content is None


class TestProcessAll:
    """Tests for process_all() batch processing."""

    def test_process_all_raises_without_scanner(self, producer):
        """process_all should raise RuntimeError when scanner is None."""
        producer.scanner = None
        with pytest.raises(RuntimeError, match="Cannot call process_all"):
            producer.process_all()

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
