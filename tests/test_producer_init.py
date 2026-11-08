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
        assert p.docling_extractor is not None

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
            extracted=None,
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


class TestProcessFileMedia:
    """Tests for the shared processing pipeline across media file types."""

    def test_m4a_uses_actual_filetype_through_pipeline(self, producer, mock_summarizer, tmp_path):
        source = tmp_path / "lecture.m4a"
        source.write_bytes(b"audio data")
        scanned = _make_scanned_file("lecture.m4a", absolute_path=str(source))

        with mock.patch(
            "focus_feed.producer.get_audio_metadata", return_value=None
        ) as get_audio_metadata:
            metadata = producer.process_file(scanned)

        assert metadata is not None
        assert metadata.filetype == "m4a"
        assert metadata.feed_eligible is True
        assert metadata.transcribe is not None
        assert metadata.summarize is not None
        mock_summarizer.summarize.assert_called_once_with(
            "This is a test transcript.",
            reflection_questions=None,
            filepath="lecture.m4a",
            filetype="m4a",
        )
        get_audio_metadata.assert_called_once_with(str(source), filetype="m4a")


class TestLegacyTopLevelErrors:
    """Tests for migrating legacy top-level errors to step-map errors."""

    @pytest.mark.parametrize("filetype", ["mp4", "pdf"])
    def test_processes_legacy_error_with_zero_step_output(self, producer, filetype, tmp_path):
        source = tmp_path / f"file.{filetype}"
        source.write_bytes(b"source data")
        step_name = "transcribe" if filetype == "mp4" else "extract"
        (tmp_path / f"file.{filetype}.json").write_text(
            json.dumps(
                {
                    "error": "legacy processing failure",
                    step_name: {"output_size": 0},
                    "mtime": source.stat().st_mtime,
                }
            )
        )

        metadata = producer.process_file(_make_scanned_file(f"file.{filetype}", str(source)))

        assert metadata is not None
        assert metadata.feed_eligible is True

    def test_skips_legacy_error_with_nonzero_step_output(
        self, producer, mock_transcriber, tmp_path
    ):
        source = tmp_path / "file.mp4"
        source.write_bytes(b"source data")
        (tmp_path / "file.mp4.json").write_text(
            json.dumps(
                {
                    "error": "legacy processing failure",
                    "transcribe": {"output_size": 10},
                    "mtime": source.stat().st_mtime,
                }
            )
        )

        metadata = producer.process_file(_make_scanned_file("file.mp4", str(source)))

        assert metadata is None
        mock_transcriber.transcribe.assert_not_called()
