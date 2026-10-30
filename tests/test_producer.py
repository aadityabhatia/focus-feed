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
from unittest import mock

import pytest

from focus_feed.errors import ProcessingError
from focus_feed.producer import Producer
from focus_feed.scanner import ScannedFile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _set_valid_env(monkeypatch):
    """Set valid environment variables for config."""
    monkeypatch.setenv("BASE_URL", "https://example.com/feeds")
    monkeypatch.setenv("LLM_ENDPOINT", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("PROMPT_ARTICLE", "Summarize key takeaways from this document.")
    monkeypatch.setenv(
        "PROMPT_ARTICLE_REFLECTION",
        "Summarize key takeaways from this document with reflection.",
    )
    monkeypatch.setenv("PROMPT_PODCAST", "Summarize key takeaways from this podcast.")
    monkeypatch.setenv(
        "PROMPT_PODCAST_REFLECTION",
        "Summarize key takeaways from this podcast with reflection.",
    )
    monkeypatch.setenv("PROMPT_TITLE", "Generate a concise, descriptive title.")
    monkeypatch.setenv("TRANSCRIPTION_ENDPOINT", "https://transcribe.example.com/v1")
    monkeypatch.setenv("TRANSCRIPTION_API_KEY", "transcribe-test-key")
    monkeypatch.setenv("TRANSCRIPTION_REMOTE_MODEL", "")
    import focus_feed.config as config_module

    config_module._config_instance = None


@pytest.fixture
def valid_env(monkeypatch):
    """Provide valid environment variables for all required config fields."""
    _set_valid_env(monkeypatch)
    yield
    import focus_feed.config as config_module

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
        "title": "document",
        "extracted_title": "PDF Title",
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
        "extracted_title": "HTML Title",
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

    def test_init_scanner_defaults_to_none(self, valid_env):
        """Producer scanner should be None when no scanner is provided."""
        p = Producer()
        assert p.scanner is None

    def test_init_with_scanner(self, valid_env):
        """Producer should accept a pre-configured Scanner."""
        from focus_feed.scanner import Scanner

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

        mock_summarizer.summarize.assert_called_once_with(
            "This is a test transcript.",
            reflection_questions=None,
            filepath="lecture.mp4",
            filetype="mp4",
        )

    def test_mp4_generates_title_from_transcript(self, producer, mock_summarizer, tmp_path):
        """process_file should call summarizer.generate_title for MP4 files."""
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(tmp_path / "lecture.mp4"),
        )
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        producer.process_file(scanned)

        mock_summarizer.generate_title.assert_called_once_with(
            "This is a test transcript.",
            filename="lecture",
            extracted_title=None,
            filepath="lecture.mp4",
        )

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
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"
        assert metadata.final_title == "Generated Title"
        assert metadata.description == ""  # MP4: description is blank
        assert metadata.summary == "Key takeaways: test summary."  # summary holds the LLM output
        assert metadata.filename == "lecture.mp4"
        assert metadata.mtime == source.stat().st_mtime
        assert metadata.filetype == "mp4"
        assert metadata.content == "This is a test transcript."
        assert metadata.size == len(b"video data")
        assert metadata.feed_eligible is True

    def test_mp4_title_dict_has_all_three_keys(self, producer, tmp_path):
        """MP4 title dict should have filename, extracted, and generated keys."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        title = metadata.title
        assert isinstance(title, dict)
        assert "filename" in title
        assert "extracted" in title
        assert "generated" in title
        # For MP4 without audio metadata title, extracted should be None
        assert title["extracted"] is None
        # filename should be sanitized (no extension, no underscores)
        assert title["filename"] == "lecture"
        # generated should be the LLM-generated title
        assert title["generated"] == "Generated Title"

    def test_mp4_title_dict_extracted_from_audio_metadata(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """MP4 title dict should include extracted title from audio metadata."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 3600.5,
            "creation_time": "2024-01-15T10:30:00.000000Z",
            "bit_rate": 128000,
            "title": "Audio Title",
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        title = metadata.title
        assert isinstance(title, dict)
        assert title["extracted"] == "Audio Title"
        assert metadata.title.get("extracted") == "Audio Title"

    def test_mp4_filename_sanitized_no_extension_no_underscores(self, producer, tmp_path):
        """MP4 title filename key should have no extension and no underscores."""
        source = tmp_path / "my_lecture_video.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "my_lecture_video.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        title = metadata.title
        assert isinstance(title, dict)
        assert title["filename"] == "my lecture video"

    def test_pdf_title_dict_has_all_three_keys(self, producer, tmp_path):
        """PDF title dict should have filename, extracted, and generated keys."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        title = metadata.title
        assert isinstance(title, dict)
        assert "filename" in title
        assert "extracted" in title
        assert "generated" in title

    def test_html_title_dict_has_all_three_keys(self, producer, tmp_path):
        """HTML title dict should have filename, extracted, and generated keys."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        title = metadata.title
        assert isinstance(title, dict)
        assert "filename" in title
        assert "extracted" in title
        assert "generated" in title

    def test_mp4_metadata_date_from_file_mtime(self, producer, tmp_path):
        """MP4 metadata date should match the file modification time (epoch float)."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        expected_mtime = int(source.stat().st_mtime * 1000) / 1000.0
        assert isinstance(metadata.date, float)
        assert metadata.date == expected_mtime

    def test_mp4_skips_transcription_when_already_complete(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """_process_mp4 should skip transcription when transcribe step is already 'complete'."""
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
                    "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                    "summarize": {"status": None, "input_size": 0, "output_size": 0},
                    "content": existing_transcript,
                    "filename": "lecture.mp4",
                    "filetype": "mp4",
                    "mtime": source.stat().st_mtime,
                }
            )
        )

        metadata = producer.process_file(scanned)

        # Transcriber should NOT have been called
        mock_transcriber.transcribe.assert_not_called()
        # Summarizer SHOULD have been called with the existing transcript
        mock_summarizer.summarize.assert_called_once_with(
            existing_transcript,
            reflection_questions=None,
            filepath="lecture.mp4",
            filetype="mp4",
        )
        # Metadata should have the existing transcript
        assert metadata is not None
        assert metadata.content == existing_transcript
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"


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
            "extracted_title": "Custom PDF Title",
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
            "extracted_title": "PDF Title",
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
            "extracted_title": "PDF Title" if content_present else None,
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
            "extracted_title": "Existing Title",
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
            "extracted_title": None,
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
            "extracted_title": None,
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
            "extracted_title": None,
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
            "extracted_title": "Existing Title",
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
            "extracted_title": None,
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


# ---------------------------------------------------------------------------
# process_all() tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# generate_title() tests
# ---------------------------------------------------------------------------


class TestGenerateTitle:
    """Tests for the generate_title() helper method."""

    def test_generate_title_calls_summarizer(self, producer, mock_summarizer):
        """generate_title should delegate to summarizer.generate_title."""
        producer.generate_title("Some content here")

        mock_summarizer.generate_title.assert_called_once_with(
            "Some content here", None, None, filepath=None
        )

    def test_generate_title_returns_title(self, producer, mock_summarizer):
        """generate_title should return the title from the summarizer."""
        result = producer.generate_title("Some content here")

        assert result == ("Generated Title", 30, 5)


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
        from focus_feed.services.extractors import PDFExtractor

        producer.pdf_extractor = PDFExtractor()

        metadata = producer.process_file(scanned)

        assert metadata is not None
        # Blank page has no extractable text, so generated falls back to filename
        assert metadata.final_title == "real"
        assert metadata.filetype == "pdf"
        assert metadata.extract.status == "complete"
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
        from focus_feed.services.extractors import HTMLExtractor

        producer.html_extractor = HTMLExtractor()

        metadata = producer.process_file(scanned)

        assert metadata is not None
        # HTML has content "Content", so LLM is called and returns "Generated Title"
        assert metadata.final_title == "Generated Title"
        assert metadata.description == "Real description"
        assert metadata.filetype == "html"
        assert metadata.extract.status == "complete"
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
        from focus_feed.services.extractors import PDFExtractor

        producer.pdf_extractor = PDFExtractor()

        metadata = producer.process_file(scanned)

        assert metadata is not None
        # PDFExtractor falls back to filename stem "untitled"
        assert metadata.final_title == "untitled"


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
        # Final state should have transcribe and summarize step maps complete
        assert data["transcribe"]["status"] == "complete"
        assert data["summarize"]["status"] == "complete"

    def test_mp4_final_state_has_transcription_model(self, producer, tmp_path):
        """MP4 final JSON state should include model in transcribe step map."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcribe"]["model"] == "remote"

    def test_mp4_transcription_model_remote_with_remote_model(self, monkeypatch, tmp_path):
        """When transcription_endpoint is set and transcription_remote_model is configured,
        transcription_model should be the remote model name."""
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("TRANSCRIPTION_ENDPOINT", "https://whisper.example.com/v1")
        monkeypatch.setenv("TRANSCRIPTION_REMOTE_MODEL", "whisper-1")
        import focus_feed.config as config_module

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
        producer.transcriber.transcribe.return_value = "Test transcript."  # type: ignore[attr-defined]

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.transcribe is not None
        assert metadata.transcribe.model == "whisper-1"

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcribe"]["model"] == "whisper-1"

        config_module._config_instance = None

    def test_mp4_transcription_model_remote_without_remote_model(self, monkeypatch, tmp_path):
        """When transcription_endpoint is set but transcription_remote_model is not configured,
        transcription model should be 'remote'."""
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("TRANSCRIPTION_ENDPOINT", "https://whisper.example.com/v1")
        monkeypatch.setenv("TRANSCRIPTION_REMOTE_MODEL", "")
        import focus_feed.config as config_module

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
        producer.transcriber.transcribe.return_value = "Test transcript."  # type: ignore[attr-defined]

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.transcribe is not None
        assert metadata.transcribe.model == "remote"

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcribe"]["model"] == "remote"

        config_module._config_instance = None

    def test_mp4_transcription_model_remote_default(self, producer, tmp_path):
        """When no remote model override is set, model is recorded as remote."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata.transcribe.model == "remote"

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["transcribe"]["model"] == "remote"

    @pytest.mark.parametrize(
        "filetype,content",
        [
            ("pdf", b"pdf data"),
            ("html", "<html>test</html>"),
        ],
    )
    def test_final_state_no_stage_field(self, producer, filetype, content, tmp_path):
        """Final JSON state should not have stage field for PDF/HTML."""
        source = tmp_path / f"file.{filetype}"
        if filetype == "html":
            source.write_text(content)
        else:
            source.write_bytes(content)
        scanned = _make_scanned_file(
            f"file.{filetype}",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / f"file.{filetype}.json"
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
        assert data.get("transcribe", {}).get("status") == "in_progress"
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
        assert data.get("transcribe", {}).get("status") == "complete"
        assert data.get("summarize", {}).get("status") == "in_progress"
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
        assert data.get("extract", {}).get("status") == "in_progress"
        assert data["error"] is not None

    def test_pdf_no_llm_title_generation_needed(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """PDF should NOT call LLM for title generation when no content is available."""
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
        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        # PDF should complete successfully without calling LLM for title
        assert data["extract"]["status"] == "complete"
        assert "error" not in data
        # LLM should NOT have been called for title generation (no content)
        mock_summarizer.generate_title.assert_not_called()

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
        assert data["extract"]["status"] == "complete"

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
        assert data.get("extract", {}).get("status") == "in_progress"
        assert data["error"] is not None

    def test_error_metadata_preserves_step_maps(self, producer, mock_transcriber, tmp_path):
        """Error metadata should preserve the step maps from the in-progress state."""
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
        assert data.get("transcribe", {}).get("status") == "in_progress"
        assert "transcribe" in data

    def test_error_metadata_preserves_extract_step(self, producer, mock_pdf_extractor, tmp_path):
        """Error metadata should preserve the extract step from the in-progress state."""
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
        assert "extract" in data

    def test_handle_error_when_source_disappears(self, producer, tmp_path):
        """Error handling should write metadata even if source stat fails."""
        source = tmp_path / "removed.pdf"

        producer._handle_error(
            RuntimeError("source removed"),
            filepath="removed.pdf",
            filename="removed.pdf",
            filetype=None,
            absolute_path=str(source),
        )

        metadata = json.loads((tmp_path / "removed.pdf.json").read_text())
        assert metadata["error"] == "source removed"
        assert "filetype" not in metadata
        assert "mtime" in metadata
        assert metadata["mtime"] is not None

    def test_partial_metadata_omits_mtime_when_source_disappears(self, producer, tmp_path):
        """Partial metadata should not serialize a null mtime for a missing source."""
        source = tmp_path / "removed.pdf"

        producer._write_metadata_partial(
            {"filename": "removed.pdf", "mtime": None},
            str(source),
        )

        metadata = json.loads((tmp_path / "removed.pdf.json").read_text())
        assert "mtime" not in metadata

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
        # PDF should NOT have transcribe step map
        assert "transcribe" not in data
        assert "error" not in data
        # extract should be complete
        assert data["extract"]["status"] == "complete"
        # summarize should be complete (PDF files are now always summarized)
        assert data["summarize"]["status"] == "complete"

    def test_final_state_has_step_maps_with_timing(self, producer, tmp_path):
        """Final JSON state should have step maps with start_time and end_time."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert "transcribe" in data
        assert "start_time" in data["transcribe"]
        assert "end_time" in data["transcribe"]
        assert data["transcribe"]["start_time"] is not None
        assert data["transcribe"]["end_time"] is not None
        assert "summarize" in data
        assert "start_time" in data["summarize"]
        assert "end_time" in data["summarize"]

    def test_metadata_object_has_complete_step_maps(self, producer, tmp_path):
        """Returned Metadata object should have all step maps complete."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"
        assert metadata.transcribe.model == "remote"
        assert metadata.feed_eligible is True

    @pytest.mark.parametrize(
        "filetype,step_map,source_content",
        [
            ("mp4", "transcribe", b"video data"),
            ("pdf", "extract", b"pdf data"),
            ("html", "extract", "<html>test</html>"),
        ],
    )
    def test_metadata_object_has_step_map_timing(
        self, producer, filetype, step_map, source_content, tmp_path
    ):
        """Returned Metadata object should have step maps with start_time and end_time."""
        source = tmp_path / f"file.{filetype}"
        if filetype == "html":
            source.write_text(source_content)
        else:
            source.write_bytes(source_content)
        scanned = _make_scanned_file(
            f"file.{filetype}",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        step = getattr(metadata, step_map)
        assert step.start_time is not None
        assert step.end_time is not None


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
        assert data.get("summarize", {}).get("status") == "complete"
        assert data.get("transcribe", {}).get("status") == "complete"
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
            written_statuses.append(metadata_dict.get("extract", {}).get("status"))
            original_write(metadata_dict, source_path)

        with mock.patch.object(producer, "_write_metadata_partial", side_effect=track_write):
            producer.process_file(scanned)

        # Should have written at least: in_progress, complete (intermediate), complete (final)
        assert len(written_statuses) >= 3
        assert "in_progress" in written_statuses
        assert "complete" in written_statuses

    def test_pdf_writes_extract_complete_without_llm_title_generation(
        self, producer, mock_pdf_extractor, mock_summarizer, tmp_path
    ):
        """_process_pdf should write 'extract'='complete' without LLM title generation.

        With the new implementation, PDF/HTML titles come from the filename,
        not from LLM generation. The extract step is completed after extraction
        without any LLM title generation.
        """
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data content")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted_title": "Custom Title",
            "description": "Desc",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
        }

        # Make generate_title raise to verify it's NOT called
        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        # Extract should be complete since no LLM title generation is needed
        assert data.get("extract", {}).get("status") == "complete"
        # No error should be present since LLM was not called
        assert "error" not in data
        # LLM should NOT have been called for title generation
        mock_summarizer.generate_title.assert_not_called()

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
            written_statuses.append(metadata_dict.get("extract", {}).get("status"))
            original_write(metadata_dict, source_path)

        with mock.patch.object(producer, "_write_metadata_partial", side_effect=track_write):
            producer.process_file(scanned)

        # Should have written at least: in_progress, complete (intermediate), complete (final)
        assert len(written_statuses) >= 3
        assert "in_progress" in written_statuses
        assert "complete" in written_statuses

    def test_html_no_llm_title_generation_needed(
        self, producer, mock_html_extractor, mock_summarizer, tmp_path
    ):
        """HTML should NOT call LLM for title generation when no content is available."""
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

        # Make generate_title raise to verify it's NOT called
        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "article.html.json"
        data = json.loads(json_path.read_text())
        # HTML should complete successfully without calling LLM for title
        assert data["extract"]["status"] == "complete"
        assert "error" not in data
        # LLM should NOT have been called for title generation (no content)
        mock_summarizer.generate_title.assert_not_called()

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
        assert data["extract"]["status"] == "complete"
        assert data["title"]["filename"] == "PDF Title"
        assert data["title"]["generated"] == "PDF Title"  # Falls back to filename
        assert "end_time" in data["extract"]

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
        assert data["extract"]["status"] == "complete"
        assert data["title"]["filename"] == "HTML Title"
        assert data["title"]["generated"] == "HTML Title"  # Falls back to filename
        assert "end_time" in data["extract"]


# ---------------------------------------------------------------------------
# Audio metadata extraction tests (MP4 with ffprobe)
# ---------------------------------------------------------------------------


class TestMP4AudioMetadata:
    """Tests for audio metadata extraction in _process_mp4."""

    def test_mp4_uses_creation_time_as_date(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """When ffprobe returns creation_time, it should be used as the date field."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 3600.5,
            "creation_time": "2024-01-15T10:30:00.000000Z",
            "bit_rate": 128000,
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        # creation_time 2024-01-15T10:30:00Z = 1705314600 epoch
        assert metadata.date == 1705314600
        assert metadata.creation_time == "2024-01-15T10:30:00.000000Z"
        assert metadata.duration == 3600.5

    def test_mp4_falls_back_to_mtime_when_no_creation_time(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """When ffprobe returns no creation_time, should fall back to file mtime."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 120.0,
            "creation_time": None,
            "bit_rate": 64000,
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        expected_mtime = int(source.stat().st_mtime * 1000) / 1000.0
        assert isinstance(metadata.date, float)
        assert metadata.date == expected_mtime
        assert metadata.creation_time is None
        assert metadata.duration == 120.0

    def test_mp4_falls_back_to_mtime_when_ffprobe_returns_none(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """When ffprobe returns None, should fall back to file mtime."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=None,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        expected_mtime = int(source.stat().st_mtime * 1000) / 1000.0
        assert isinstance(metadata.date, float)
        assert metadata.date == expected_mtime
        assert metadata.creation_time is None
        assert metadata.duration is None

    def test_mp4_json_includes_duration_and_creation_time(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """JSON metadata should include duration and creation_time fields."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 1800.0,
            "creation_time": "2024-06-01T12:00:00Z",
            "bit_rate": 192000,
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert data["duration"] == 1800.0
        assert data["creation_time"] == "2024-06-01T12:00:00Z"

    def test_mp4_date_from_creation_time_is_epoch_float(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """Date from creation_time should be stored as a float (UNIX epoch)."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 300.0,
            "creation_time": "2024-06-15T08:30:00.000000Z",
            "bit_rate": 128000,
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        assert isinstance(metadata.date, float)
        # 2024-06-15T08:30:00Z = 1718440200 epoch
        assert metadata.date == 1718440200.0

    def test_mp4_invalid_creation_time_falls_back_to_mtime(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """When creation_time cannot be parsed, should fall back to file mtime."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 100.0,
            "creation_time": "not-a-valid-date",
            "bit_rate": 128000,
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        expected_mtime = int(source.stat().st_mtime * 1000) / 1000.0
        assert isinstance(metadata.date, float)
        assert metadata.date == expected_mtime
        # creation_time should still be stored as the raw string
        assert metadata.creation_time == "not-a-valid-date"

    def test_mp4_creation_time_without_timezone_assumes_utc(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """creation_time without timezone info should be treated as UTC."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        audio_meta = {
            "type": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": 200.0,
            "creation_time": "2024-03-20T14:00:00",
            "bit_rate": 128000,
        }

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value=audio_meta,
        ):
            metadata = producer.process_file(scanned)

        assert metadata is not None
        # 2024-03-20T14:00:00 UTC = 1710943200 epoch
        assert metadata.date == 1710943200

    def test_mp4_audio_metadata_called_with_absolute_path(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """get_audio_metadata should be called with the absolute path."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        with mock.patch(
            "focus_feed.producer.get_audio_metadata",
            return_value={"type": None, "duration": 0.0, "creation_time": None, "bit_rate": None},
        ) as mock_get_meta:
            producer.process_file(scanned)

        mock_get_meta.assert_called_once_with(str(source))


# ---------------------------------------------------------------------------
# publication_date tests
# ---------------------------------------------------------------------------


class TestPublicationDate:
    """Tests for publication_date field in producer metadata."""

    def test_pdf_publication_date_from_extractor(self, producer, mock_pdf_extractor, tmp_path):
        """PDF metadata should include publication_date from extractor."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted_title": "PDF Title",
            "description": "PDF Description",
            "date": 1728931200,
            "publication_date": 1719200000.0,
            "size": 1024,
            "filetype": "pdf",
            "content": "PDF plaintext content",
        }
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.publication_date == 1719200000.0

    def test_pdf_publication_date_none_when_extractor_returns_none(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF metadata should have publication_date=None when extractor returns None."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted_title": None,
            "description": "PDF Description",
            "date": 1728931200,
            "publication_date": None,
            "size": 1024,
            "filetype": "pdf",
            "content": "PDF plaintext content",
        }
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.publication_date is None

    def test_html_publication_date_from_extractor(self, producer, mock_html_extractor, tmp_path):
        """HTML metadata should include publication_date from extractor."""
        mock_html_extractor.extract.return_value = {
            "title": "article",
            "extracted_title": "HTML Title",
            "description": "HTML Description",
            "date": 1728931200,
            "publication_date": 1719200000.0,
            "size": 512,
            "filetype": "html",
            "content": "HTML plaintext content",
        }
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.publication_date == 1719200000.0

    def test_html_publication_date_none_when_extractor_returns_none(
        self, producer, mock_html_extractor, tmp_path
    ):
        """HTML metadata should have publication_date=None when extractor returns None."""
        mock_html_extractor.extract.return_value = {
            "title": "article",
            "extracted_title": None,
            "description": "HTML Description",
            "date": 1728931200,
            "publication_date": None,
            "size": 512,
            "filetype": "html",
            "content": "HTML plaintext content",
        }
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.publication_date is None

    def test_pdf_metadata_json_includes_publication_date(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF JSON metadata should include publication_date field."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted_title": "PDF Title",
            "description": "Desc",
            "date": 1728931200,
            "publication_date": 1719200000.0,
            "size": 1024,
            "filetype": "pdf",
            "content": "Content",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert "publication_date" in data
        assert data["publication_date"] == 1719200000.0

    def test_pdf_metadata_json_excludes_none_publication_date(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF JSON metadata should exclude publication_date when None."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted_title": None,
            "description": "Desc",
            "date": 1728931200,
            "publication_date": None,
            "size": 1024,
            "filetype": "pdf",
            "content": "Content",
        }
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert "publication_date" not in data


# ---------------------------------------------------------------------------
# Hash computation tests
# ---------------------------------------------------------------------------


class TestHashComputation:
    """Tests for SHA256 hash computation in the producer."""

    def test_hash_computed_on_first_process_mp4(self, producer, tmp_path):
        """Hash should be computed and stored when processing an MP4 file for the first time."""
        import hashlib

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        expected_hash = hashlib.sha256(b"video data").hexdigest()
        assert metadata is not None
        assert metadata.hash == expected_hash

    def test_hash_computed_on_first_process_pdf(self, producer, tmp_path):
        """Hash should be computed and stored when processing a PDF file for the first time."""
        import hashlib

        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        expected_hash = hashlib.sha256(b"pdf data").hexdigest()
        assert metadata is not None
        assert metadata.hash == expected_hash

    def test_hash_computed_on_first_process_html(self, producer, tmp_path):
        """Hash should be computed and stored when processing an HTML file for the first time."""
        import hashlib

        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        expected_hash = hashlib.sha256(b"<html>test</html>").hexdigest()
        assert metadata is not None
        assert metadata.hash == expected_hash

    def test_hash_stored_in_json_metadata(self, producer, tmp_path):
        """Hash should be written to the JSON metadata file."""
        import hashlib

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        expected_hash = hashlib.sha256(b"video data").hexdigest()
        assert data["hash"] == expected_hash

    def test_hash_preserved_when_not_force(self, producer, tmp_path):
        """Hash should be preserved when reprocessing without --force."""
        import hashlib

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # First processing
        producer.process_file(scanned)

        # Pre-write JSON with existing hash and complete steps
        existing_hash = hashlib.sha256(b"video data").hexdigest()
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "hash": existing_hash,
                    "transcribe": {
                        "status": "complete",
                        "input_size": 1000,
                        "output_size": 500,
                    },
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                    },
                    "content": "Existing transcript.",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "title": {"filename": "lecture", "extracted": None, "generated": None},
                    "description": "",
                    "summary": "Summary",
                    "date": 1728931200,
                    "mtime": source.stat().st_mtime,
                    "size": 10,
                }
            )
        )

        # Unchanged, eligible metadata with a matching mtime is skipped.
        metadata = producer.process_file(scanned)

        assert metadata is None
        data = json.loads(json_path.read_text())
        assert data["hash"] == existing_hash

    def test_hash_recomputed_when_force(self, producer, tmp_path):
        """Hash should be recomputed when reprocessing with --force."""
        import hashlib

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # First processing
        producer.process_file(scanned)

        # Pre-write JSON with a different hash and complete steps
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "hash": "0" * 64,  # Fake hash
                    "transcribe": {
                        "status": "complete",
                        "input_size": 1000,
                        "output_size": 500,
                    },
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                    },
                    "content": "Existing transcript.",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "title": {"filename": "lecture", "extracted": None, "generated": None},
                    "description": "",
                    "summary": "Summary",
                    "date": 1728931200,
                    "size": 10,
                }
            )
        )

        # Reprocess with force
        metadata = producer.process_file(scanned, force=True)

        expected_hash = hashlib.sha256(b"video data").hexdigest()
        assert metadata is not None
        assert metadata.hash == expected_hash

    def test_hash_computed_for_legacy_file_without_hash(self, producer, tmp_path):
        """Hash should be computed when processing a legacy file that has no hash field."""
        import hashlib

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # Pre-write JSON without hash field, with incomplete processing
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {
                        "status": "in_progress",
                        "input_size": 0,
                        "output_size": 0,
                    },
                    "summarize": {
                        "status": None,
                        "input_size": 0,
                        "output_size": 0,
                    },
                    "content": "Partial transcript.",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "title": {"filename": "lecture", "extracted": None, "generated": None},
                    "description": "",
                    "date": 1728931200,
                    "size": 10,
                }
            )
        )

        metadata = producer.process_file(scanned)

        expected_hash = hashlib.sha256(b"video data").hexdigest()
        assert metadata is not None
        assert metadata.hash == expected_hash

    def test_hash_is_not_computed_before_transcription_succeeds(
        self, producer, mock_transcriber, tmp_path
    ):
        """MP4 hashing occurs after transcription, so failures have no new hash."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        # Make transcription fail
        mock_transcriber.transcribe.side_effect = RuntimeError("GPU error")

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert "hash" not in data
        assert data["error"] is not None

    def test_hash_is_64_char_hex_digest(self, producer, tmp_path):
        """Hash should be a 64-character hex string (SHA256)."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert len(metadata.hash) == 64
        # Verify it's a valid hex string
        int(metadata.hash, 16)  # Raises ValueError if not hex

    @pytest.mark.parametrize(
        "filetype,content",
        [
            ("mp4", b"video data"),
            ("pdf", b"pdf data"),
            ("html", b"<html>test</html>"),
        ],
    )
    def test_hash_computed_for_all_file_types(self, producer, filetype, content, tmp_path):
        """Hash should be computed for all supported file types."""
        import hashlib

        source = tmp_path / f"file.{filetype}"
        if filetype == "html":
            source.write_text(content.decode("utf-8"))
        else:
            source.write_bytes(content)
        scanned = _make_scanned_file(
            f"file.{filetype}",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        expected_hash = hashlib.sha256(content).hexdigest()
        assert metadata is not None
        assert metadata.hash == expected_hash

    def test_changed_mtime_reprocesses_and_recomputes_hash(self, producer, tmp_path):
        """Changed sources must not skip, and must get a hash for new bytes."""
        import hashlib
        import os
        import time

        source = tmp_path / "document.pdf"
        source.write_bytes(b"first PDF")
        scanned = _make_scanned_file("document.pdf", absolute_path=str(source))
        first = producer.process_file(scanned)
        assert first is not None

        original_mtime = source.stat().st_mtime
        source.write_bytes(b"changed PDF")
        changed_mtime = max(time.time(), original_mtime + 10)
        os.utime(source, (changed_mtime, changed_mtime))

        second = producer.process_file(scanned)

        assert second is not None
        assert second.hash == hashlib.sha256(b"changed PDF").hexdigest()
        assert second.mtime == changed_mtime


# ---------------------------------------------------------------------------
# generate_title step map tracking tests
# ---------------------------------------------------------------------------


class TestGenerateTitleStepTracking:
    """Tests for the generate_title StepMap tracking across file types."""

    def test_mp4_generate_title_step_complete_with_metrics(self, producer, tmp_path):
        """MP4 metadata should have a complete generate_title step with metrics."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is not None
        assert metadata.generate_title.status == "complete"
        assert metadata.generate_title.start_time is not None
        assert metadata.generate_title.end_time is not None
        assert metadata.generate_title.model == "test-model"
        assert metadata.generate_title.input_tokens == 30
        assert metadata.generate_title.output_tokens == 5
        assert metadata.generate_title.words == 2  # "Generated Title"
        assert metadata.generate_title.output_size > 0

    def test_mp4_json_has_generate_title_step(self, producer, tmp_path):
        """MP4 JSON metadata should include the generate_title step map."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        assert "generate_title" in data
        assert data["generate_title"]["status"] == "complete"
        assert data["generate_title"]["model"] == "test-model"
        assert data["generate_title"]["input_tokens"] == 30
        assert data["generate_title"]["output_tokens"] == 5

    def test_pdf_generate_title_step_when_content_available(self, producer, tmp_path):
        """PDF with content should have a complete generate_title step."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is not None
        assert metadata.generate_title.status == "complete"
        assert metadata.generate_title.model == "test-model"
        assert metadata.generate_title.input_tokens == 30
        assert metadata.generate_title.output_tokens == 5

    def test_pdf_no_generate_title_step_when_no_content(
        self, producer, mock_pdf_extractor, tmp_path
    ):
        """PDF without content should NOT have a generate_title step."""
        source = tmp_path / "document.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "document.pdf",
            absolute_path=str(source),
        )

        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted_title": None,
            "description": "PDF Description",
            "date": 1728931200,
            "size": 1024,
            "filetype": "pdf",
            # No content key
        }

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is None

        json_path = tmp_path / "document.pdf.json"
        data = json.loads(json_path.read_text())
        assert "generate_title" not in data

    def test_html_generate_title_step_when_content_available(self, producer, tmp_path):
        """HTML with content should have a complete generate_title step."""
        source = tmp_path / "article.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "article.html",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is not None
        assert metadata.generate_title.status == "complete"
        assert metadata.generate_title.model == "test-model"
        assert metadata.generate_title.input_tokens == 30
        assert metadata.generate_title.output_tokens == 5

    def test_html_no_generate_title_step_when_no_content(
        self, producer, mock_html_extractor, tmp_path
    ):
        """HTML without content should NOT have a generate_title step."""
        source = tmp_path / "page.html"
        source.write_text("<html>test</html>")
        scanned = _make_scanned_file(
            "page.html",
            absolute_path=str(source),
        )

        mock_html_extractor.extract.return_value = {
            "title": "",
            "extracted_title": None,
            "description": "Desc",
            "date": 1728931200,
            "size": 512,
            "filetype": "html",
            # No content key
        }

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is None

        json_path = tmp_path / "page.html.json"
        data = json.loads(json_path.read_text())
        assert "generate_title" not in data

    def test_mp4_generate_title_step_preserved_on_error(self, producer, mock_summarizer, tmp_path):
        """When title generation fails, the generate_title step should show in_progress."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "lecture.mp4.json"
        data = json.loads(json_path.read_text())
        # generate_title should be in_progress (written before the LLM call)
        assert data.get("generate_title", {}).get("status") == "in_progress"
        assert data["error"] is not None

    def test_pdf_generate_title_step_preserved_on_error(self, producer, mock_summarizer, tmp_path):
        """When title generation fails for PDF, generate_title should show in_progress."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        mock_summarizer.generate_title.side_effect = RuntimeError("title fail")

        producer.process_file(scanned)

        json_path = tmp_path / "doc.pdf.json"
        data = json.loads(json_path.read_text())
        assert data.get("generate_title", {}).get("status") == "in_progress"
        assert data["error"] is not None

    def test_mp4_generate_title_step_input_size_matches_transcript(self, producer, tmp_path):
        """generate_title input_size should match the transcript byte length."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file(
            "lecture.mp4",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is not None
        expected_input_size = len(b"This is a test transcript.")
        assert metadata.generate_title.input_size == expected_input_size

    def test_pdf_generate_title_step_input_size_matches_content(self, producer, tmp_path):
        """generate_title input_size should match the extracted content byte length."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        scanned = _make_scanned_file(
            "doc.pdf",
            absolute_path=str(source),
        )

        metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.generate_title is not None
        expected_input_size = len(b"PDF plaintext content")
        assert metadata.generate_title.input_size == expected_input_size
