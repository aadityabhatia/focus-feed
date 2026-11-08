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

from focus_feed.metadata import Metadata
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


@pytest.mark.slow
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

    def test_mp4_transcription_model_remote_with_remote_model(
        self, monkeypatch, tmp_path, valid_env
    ):
        """When transcription_endpoint is set and transcription_remote_model is configured,
        transcription_model should be the remote model name."""
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

    def test_mp4_transcription_model_remote_without_remote_model(
        self, monkeypatch, tmp_path, valid_env
    ):
        """When transcription_endpoint is set but transcription_remote_model is not configured,
        transcription model should be 'remote'."""
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
        assert data.get("transcribe", {}).get("status") is None
        assert data["transcribe"]["error"] == "fail"
        assert data["transcribe"]["output_size"] == 0
        assert "error" not in data

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
        assert data.get("summarize", {}).get("status") is None
        assert data["summarize"]["error"] == "fail"
        assert data["summarize"]["output_size"] == 0
        assert "error" not in data

    def test_mp4_summarization_failure_is_attributed_to_summarize(
        self, producer, mock_transcriber, mock_summarizer, tmp_path
    ):
        """A summary failure belongs to summarize, not the preceding transcribe step."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        scanned = _make_scanned_file("lecture.mp4", absolute_path=str(source))
        mock_summarizer.summarize.side_effect = RuntimeError("summary unavailable")

        producer.process_file(scanned)

        data = json.loads((tmp_path / "lecture.mp4.json").read_text())
        assert data["summarize"]["status"] is None
        assert data["summarize"]["error"] == "summary unavailable"
        assert data["summarize"]["output_size"] == 0
        assert "error" not in data
        assert "error" not in data["transcribe"]

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
        assert data.get("extract", {}).get("status") is None
        assert data["extract"]["error"] == "fail"
        assert data["extract"]["output_size"] == 0
        assert "error" not in data

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
        assert data.get("extract", {}).get("status") is None
        assert data["extract"]["error"] == "fail"
        assert data["extract"]["output_size"] == 0
        assert "error" not in data

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
        assert "error" not in data
        assert data.get("transcribe", {}).get("status") is None
        assert data["transcribe"]["error"] == "GPU error"
        assert data["transcribe"]["output_size"] == 0
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
        assert "error" not in data
        assert "extract" in data
        assert data["extract"]["error"] == "PDF corrupt"
        assert data["extract"]["output_size"] == 0

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
        assert metadata["filetype"] == "pdf"
        assert "mtime" in metadata
        assert metadata["mtime"] is not None
        assert metadata["title"]["filename"] == "removed"
        assert metadata["size"] == 0
        loaded = Metadata.from_json_file(tmp_path / "removed.pdf.json")
        assert loaded.error == "source removed"

    def test_partial_metadata_with_missing_source_is_loadable(self, producer, tmp_path):
        """Partial metadata includes schema fields and remains loadable without its source."""
        source = tmp_path / "removed.pdf"

        producer._write_metadata_partial(
            {"filename": "removed.pdf", "mtime": None},
            str(source),
        )

        metadata = json.loads((tmp_path / "removed.pdf.json").read_text())
        assert metadata["mtime"] is None
        assert metadata["title"]["filename"] == "removed"
        assert metadata["size"] == 0
        loaded = Metadata.from_json_file(tmp_path / "removed.pdf.json")
        assert loaded.filename == "removed.pdf"

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
        assert "error" not in data
        assert data["generate_title"]["error"] == "title fail"

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
        round_tripped_statuses: list[str | None] = []

        original_write = producer._write_metadata_partial

        def track_write(metadata_dict, source_path):
            written_statuses.append(metadata_dict.get("extract", {}).get("status"))
            original_write(metadata_dict, source_path)
            loaded = Metadata.from_json_file(tmp_path / "document.pdf.json")
            round_tripped_statuses.append(
                loaded.extract.status if loaded.extract is not None else None
            )

        with mock.patch.object(producer, "_write_metadata_partial", side_effect=track_write):
            producer.process_file(scanned)

        # Should have written at least: in_progress, complete (intermediate), complete (final)
        assert len(written_statuses) >= 3
        assert "in_progress" in written_statuses
        assert "complete" in written_statuses
        assert round_tripped_statuses == written_statuses
        final_metadata = Metadata.from_json_file(tmp_path / "document.pdf.json")
        assert final_metadata.extract is not None
        assert final_metadata.extract.status == "complete"

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
            "extracted": "Custom Title",
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
