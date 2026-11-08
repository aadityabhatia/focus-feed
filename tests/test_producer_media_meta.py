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

        mock_get_meta.assert_called_once_with(str(source), filetype="mp4")


class TestPublicationDate:
    """Tests for publication_date field in producer metadata."""

    def test_pdf_publication_date_from_extractor(self, producer, mock_pdf_extractor, tmp_path):
        """PDF metadata should include publication_date from extractor."""
        mock_pdf_extractor.extract.return_value = {
            "title": "doc",
            "extracted": "PDF Title",
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
            "extracted": None,
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
            "extracted": "HTML Title",
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
            "extracted": None,
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
            "extracted": "PDF Title",
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
            "extracted": None,
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
        assert data["transcribe"]["error"] == "GPU error"

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
            "extracted": None,
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
            "extracted": None,
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

    def test_mp4_generate_title_failure_is_attributed_to_generate_title(
        self, producer, mock_summarizer, tmp_path
    ):
        """A title-generation failure clears status and records the generate_title error."""
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
        assert data.get("generate_title", {}).get("status") is None
        assert data["generate_title"]["output_size"] == 0
        assert "error" not in data
        assert data["generate_title"]["error"] == "title fail"

    def test_pdf_generate_title_failure_is_attributed_to_generate_title(
        self, producer, mock_summarizer, tmp_path
    ):
        """A PDF title-generation failure clears status and records its step error."""
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
        assert data.get("generate_title", {}).get("status") is None
        assert data["generate_title"]["output_size"] == 0
        assert "error" not in data
        assert data["generate_title"]["error"] == "title fail"

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
