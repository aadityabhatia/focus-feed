"""Integration tests for Phase 2 and Phase 3 modules.

These tests verify that modules work together correctly:
- Config + Metadata integration
- Scanner + Metadata integration
- Producer component with mocked services
- Error handling paths
"""

import os
from pathlib import Path
from typing import Any, Literal, cast
from unittest import mock

from local_feed_generator.config import get_config, reload_config
from local_feed_generator.errors import (
    SummarizationError,
    TranscriptionError,
    create_error_metadata,
)
from local_feed_generator.metadata import Metadata, StepMap
from local_feed_generator.scanner import Scanner
from local_feed_generator.services.extractors import (
    HTMLExtractor,
    PDFExtractor,
    _clean_title,
)

# ---------------------------------------------------------------------------
# Service Mocks
# ---------------------------------------------------------------------------


class MockTranscriber:
    """Mock transcriber that returns deterministic transcript without GPU."""

    def __init__(self, transcript: str = "Mock transcript for testing") -> None:
        self.transcript = transcript
        self.call_count = 0
        self.called_with: list[str] = []

    def transcribe(self, filepath: str | Path) -> str:
        self.call_count += 1
        self.called_with.append(str(filepath))
        return self.transcript


class MockSummarizer:
    """Mock summarizer that returns deterministic summary without LLM."""

    def __init__(self, summary: str = "Mock summary of key takeaways") -> None:
        self.summary = summary
        self.call_count = 0
        self.called_with: list[str] = []

    def summarize(self, transcript: str, **kwargs: Any) -> tuple[str, int, int]:
        self.call_count += 1
        self.called_with.append(transcript)
        return (self.summary, 100, 50)

    def generate_title(
        self,
        content: str,
        filename: str | None = None,
        extracted_title: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        self.call_count += 1
        self.called_with.append(content)
        return ("Mock Generated Title", 30, 5)


class FailingTranscriber:
    """Mock transcriber that always fails."""

    def transcribe(self, filepath: str | Path) -> str:
        raise TranscriptionError("Mock transcription failure", filepath=str(filepath))


class FailingSummarizer:
    """Mock summarizer that always fails."""

    def summarize(self, transcript: str, **kwargs: Any) -> tuple[str, int, int]:
        raise SummarizationError("Mock summarization failure")


# ---------------------------------------------------------------------------
# Config + Metadata Integration
# ---------------------------------------------------------------------------


class TestConfigMetadataIntegration:
    """Tests that config loads correctly and metadata can be created with config values."""

    def _get_valid_env(self) -> dict[str, str]:
        return {
            "BASE_URL": "https://example.com/feeds",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }

    def test_config_loads_and_metadata_created_with_config_values(self) -> None:
        """Config should load and metadata can be created using config-derived paths."""
        env = self._get_valid_env()
        with mock.patch.dict(os.environ, env, clear=False):
            # Reset singleton
            import local_feed_generator.config as config_module

            config_module._config_instance = None

            config = get_config()
            assert config.base_url == "https://example.com/feeds"

            # Create metadata using config values for filepath
            relative_path = "videos/test.mp4"
            metadata = Metadata(
                title="Test Video",
                description="Test description",
                filename="test.mp4",
                filepath=relative_path,
                filetype="mp4",
                date=1728931200,
                size=1234,
                content="Mock transcript",
                summary="Mock summary",
                transcribe=StepMap(status="complete", input_size=1234, output_size=0),
                summarize=StepMap(status="complete", input_size=0, output_size=0),
            )

            # Verify metadata can be serialized and includes config-derived path
            data = metadata.to_dict()
            assert data["filepath"] == relative_path
            assert data["filename"] == "test.mp4"

    def test_metadata_feed_eligible_from_config_context(self) -> None:
        """Metadata feed_eligible should be computed correctly regardless of config."""
        env = self._get_valid_env()
        with mock.patch.dict(os.environ, env, clear=False):
            metadata = Metadata(
                title="Test",
                description="Test",
                filename="test.pdf",
                filepath="docs/test.pdf",
                filetype="pdf",
                date=1728931200,
                size=100,
                extract=StepMap(status="complete", input_size=100, output_size=0),
                summarize=StepMap(status="complete", input_size=100, output_size=0),
            )
            assert metadata.feed_eligible is True

    def test_config_reload_reflects_new_values_for_metadata_creation(self) -> None:
        """Config reload should allow metadata to be created with updated config."""
        env1 = {**self._get_valid_env(), "BASE_URL": "https://first.example.com"}
        env2 = {**self._get_valid_env(), "BASE_URL": "https://second.example.com"}

        import local_feed_generator.config as config_module

        config_module._config_instance = None

        with mock.patch.dict(os.environ, env1, clear=False):
            config1 = get_config()
            assert config1.base_url == "https://first.example.com"

        with mock.patch.dict(os.environ, env2, clear=False):
            config2 = reload_config()
            assert config2.base_url == "https://second.example.com"

            # Metadata can still be created with new config context
            metadata = Metadata(
                title="Test",
                description="Test",
                filename="test.html",
                filepath="docs/test.html",
                filetype="html",
                date=1728931200,
                size=100,
                extract=StepMap(status="complete", input_size=100, output_size=0),
                summarize=StepMap(status="complete", input_size=100, output_size=0),
            )
            assert metadata.feed_eligible is True


# ---------------------------------------------------------------------------
# Scanner + Metadata Integration
# ---------------------------------------------------------------------------


class TestScannerMetadataIntegration:
    """Tests that scanner returns files that can be processed into valid metadata."""

    def test_scanner_finds_files_and_metadata_created(self, tmp_path: Path) -> None:
        """Scanner should find files and they should be convertible to metadata."""
        # Create test files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        videos_dir = data_dir / "videos"
        videos_dir.mkdir()

        test_file = videos_dir / "lecture.mp4"
        test_file.write_bytes(b"fake mp4 content")

        # Configure scanner
        scanner = Scanner(data_dir=str(data_dir))
        files = scanner.scan_directory()

        assert len(files) == 1
        scanned = files[0]
        assert scanned.filename == "lecture.mp4"
        assert scanned.filetype == "mp4"
        assert scanned.filepath == "videos/lecture.mp4"

        # Create metadata from scanned file
        stat = test_file.stat()
        metadata = Metadata(
            title="Lecture",
            description="Test lecture",
            filename=scanned.filename,
            filepath=scanned.filepath,
            filetype=cast(Literal["mp4", "pdf", "html"], scanned.filetype),
            date=int(stat.st_mtime),
            size=stat.st_size,
            transcribe=StepMap(status="complete", input_size=stat.st_size, output_size=0),
            summarize=StepMap(status="complete", input_size=0, output_size=0),
        )

        assert metadata.filename == "lecture.mp4"
        assert metadata.filepath == "videos/lecture.mp4"
        assert metadata.filetype == "mp4"
        assert metadata.feed_eligible is True

    def test_scanner_skips_files_with_existing_metadata(self, tmp_path: Path) -> None:
        """Scanner should skip files that already have metadata."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        docs_dir = data_dir / "docs"
        docs_dir.mkdir()

        pdf_file = docs_dir / "doc.pdf"
        pdf_file.write_bytes(b"fake pdf")

        # Create metadata file
        metadata_file = Path(str(pdf_file) + ".json")
        metadata = Metadata(
            title="Doc",
            description="Test",
            filename="doc.pdf",
            filepath="docs/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
            extract=StepMap(status="complete", input_size=100, output_size=0),
            summarize=StepMap(status="complete", input_size=100, output_size=0),
        )
        metadata.to_json_file(metadata_file)

        scanner = Scanner(data_dir=str(data_dir))
        files = scanner.scan_directory()

        # Should be empty because metadata exists
        assert len(files) == 0

    def test_scanner_finds_outdated_metadata(self, tmp_path: Path) -> None:
        """Scanner should find files where source is newer than metadata."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        html_file = data_dir / "page.html"
        html_file.write_text("<html><title>Test</title></html>")

        # Create old metadata
        metadata_file = Path(str(html_file) + ".json")
        metadata = Metadata(
            title="Old",
            description="Old",
            filename="page.html",
            filepath="page.html",
            filetype="html",
            date=1728931200,
            size=100,
            extract=StepMap(status="complete", input_size=100, output_size=0),
        )
        metadata.to_json_file(metadata_file)

        # Modify source file to be newer
        import time

        time.sleep(0.01)
        html_file.write_text("<html><title>Updated</title></html>")

        scanner = Scanner(data_dir=str(data_dir))
        files = scanner.scan_directory()

        assert len(files) == 1
        assert files[0].filename == "page.html"

    def test_scanner_returns_sorted_results(self, tmp_path: Path) -> None:
        """Scanner should return files sorted by filepath."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create files in non-alphabetical order
        (data_dir / "z_file.pdf").write_bytes(b"z")
        (data_dir / "a_file.pdf").write_bytes(b"a")
        subdir = data_dir / "sub"
        subdir.mkdir()
        (subdir / "b_file.html").write_bytes(b"b")

        scanner = Scanner(data_dir=str(data_dir))
        files = scanner.scan_directory()

        filepaths = [f.filepath for f in files]
        assert filepaths == sorted(filepaths)


# ---------------------------------------------------------------------------
# Producer Component Tests (with mocked services)
# ---------------------------------------------------------------------------


class TestProducerComponent:
    """Tests for producer-like logic with mocked services."""

    def test_file_type_detection(self) -> None:
        """File type should be correctly detected from extension."""
        # Simulate file type detection logic
        test_cases = [
            ("video.mp4", "mp4"),
            ("doc.pdf", "pdf"),
            ("page.html", "html"),
            ("VIDEO.MP4", "mp4"),
            ("DOC.PDF", "pdf"),
        ]

        for filename, expected_type in test_cases:
            ext = Path(filename).suffix.lower().lstrip(".")
            assert ext == expected_type

    def test_producer_dispatch_logic_with_mocks(self, tmp_path: Path) -> None:
        """Producer should dispatch to correct services based on file type."""
        # Setup mock services
        transcriber = MockTranscriber()
        summarizer = MockSummarizer()

        # Simulate processing MP4 file
        mp4_file = tmp_path / "test.mp4"
        mp4_file.write_bytes(b"fake")

        # Producer logic for MP4
        filetype = "mp4"
        if filetype == "mp4":
            transcript = transcriber.transcribe(str(mp4_file))
            summary, _, _ = summarizer.summarize(transcript)

        assert transcriber.call_count == 1
        assert summarizer.call_count == 1
        assert transcript == "Mock transcript for testing"
        assert summary == "Mock summary of key takeaways"

    def test_producer_creates_valid_metadata_for_mp4(self, tmp_path: Path) -> None:
        """Producer should create valid metadata for MP4 files."""
        transcriber = MockTranscriber()
        summarizer = MockSummarizer()

        mp4_file = tmp_path / "lecture.mp4"
        mp4_file.write_bytes(b"fake mp4")

        # Simulate producer processing
        transcript = transcriber.transcribe(str(mp4_file))
        summary, _, _ = summarizer.summarize(transcript)
        stat = mp4_file.stat()

        metadata = Metadata(
            title="Lecture",
            description=summary,
            filename="lecture.mp4",
            filepath="lecture.mp4",
            filetype="mp4",
            date=int(stat.st_mtime),
            size=stat.st_size,
            content=transcript,
            summary=summary,
            transcribe=StepMap(status="complete", input_size=stat.st_size, output_size=0),
            summarize=StepMap(status="complete", input_size=0, output_size=0),
        )

        assert metadata.transcribe is not None
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize is not None
        assert metadata.summarize.status == "complete"
        assert metadata.content == "Mock transcript for testing"
        assert metadata.summary == "Mock summary of key takeaways"
        assert metadata.filetype == "mp4"
        assert metadata.feed_eligible is True

    def test_producer_creates_valid_metadata_for_pdf(self, tmp_path: Path) -> None:
        """Producer should create valid metadata for PDF files using extractor."""
        # Create a simple PDF file (minimal valid PDF)
        pdf_file = tmp_path / "doc.pdf"
        # Write minimal PDF content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        pdf_file.write_bytes(pdf_content)

        extractor = PDFExtractor()
        extracted = extractor.extract(str(pdf_file))

        metadata = Metadata(
            title=extracted["title"],
            description=extracted["description"] or "",
            filename="doc.pdf",
            filepath="doc.pdf",
            filetype="pdf",
            date=extracted["date"],
            size=extracted["size"],
            extract=StepMap(status="complete", input_size=extracted["size"], output_size=0),
            summarize=StepMap(status="complete", input_size=0, output_size=0),
        )

        assert metadata.filetype == "pdf"
        assert metadata.title == "doc"  # Fallback to filename
        assert metadata.content is None
        assert metadata.summary is None
        assert metadata.feed_eligible is True

    def test_producer_creates_valid_metadata_for_html(self, tmp_path: Path) -> None:
        """Producer should create valid metadata for HTML files using extractor."""
        html_file = tmp_path / "page.html"
        html_file.write_text(
            "<html><head><title>Test Page</title></head><body>Content</body></html>"
        )

        extractor = HTMLExtractor()
        extracted = extractor.extract(str(html_file))

        metadata = Metadata(
            title=extracted["title"],
            extracted_title=extracted.get("extracted_title"),
            description=extracted["description"] or "",
            filename="page.html",
            filepath="page.html",
            filetype="html",
            date=extracted["date"],
            size=extracted["size"],
            extract=StepMap(status="complete", input_size=extracted["size"], output_size=0),
            summarize=StepMap(status="complete", input_size=0, output_size=0),
        )

        assert metadata.filetype == "html"
        assert metadata.final_title == "page"
        assert metadata.extracted_title == "Test Page"
        assert metadata.description == ""  # No meta description tag in HTML
        assert metadata.content is None
        assert metadata.summary is None
        assert metadata.feed_eligible is True

    def test_producer_uses_title_generation_fallback(self, tmp_path: Path) -> None:
        """Producer should use LLM title generation when extractor finds no title."""
        summarizer = MockSummarizer()

        # HTML with no title
        html_file = tmp_path / "no_title.html"
        html_file.write_text("<html><body>No title here</body></html>")

        extractor = HTMLExtractor()
        extracted = extractor.extract(str(html_file))

        # If title is just cleaned filename, use LLM to generate better title
        if extracted["title"] == _clean_title(html_file.stem):
            title, _, _ = summarizer.generate_title(html_file.read_text())
        else:
            title = extracted["title"]

        assert title == "Mock Generated Title"
        assert summarizer.call_count == 1


# ---------------------------------------------------------------------------
# Error Path Testing
# ---------------------------------------------------------------------------


class TestErrorPathHandling:
    """Tests that errors result in proper error metadata."""

    def test_transcription_error_creates_error_metadata(self, tmp_path: Path) -> None:
        """Transcription failure should create error metadata."""
        mp4_file = tmp_path / "bad.mp4"
        mp4_file.write_bytes(b"corrupted")

        transcriber = FailingTranscriber()

        try:
            transcriber.transcribe(str(mp4_file))
            raise AssertionError("Should have raised TranscriptionError")
        except TranscriptionError as exc:
            # Create error metadata
            error_meta = create_error_metadata(
                exc=exc,
                filename="bad.mp4",
                filepath="videos/bad.mp4",
                filetype="mp4",
            )

            assert error_meta["error"] == "Mock transcription failure"
            assert error_meta["filename"] == "bad.mp4"
            assert error_meta["filetype"] == "mp4"

            # Verify it can be validated as Metadata
            metadata = Metadata.from_dict(error_meta)
            assert metadata.error == "Mock transcription failure"
            assert metadata.feed_eligible is False

    def test_summarization_error_creates_error_metadata(self, tmp_path: Path) -> None:
        """Summarization failure should create error metadata."""
        summarizer = FailingSummarizer()

        try:
            summarizer.summarize("Some transcript")
            raise AssertionError("Should have raised SummarizationError")
        except SummarizationError as exc:
            error_meta = create_error_metadata(
                exc=exc,
                filename="lecture.mp4",
                filepath="videos/lecture.mp4",
                filetype="mp4",
            )

            assert error_meta["error"]
            assert "Mock summarization failure" in error_meta["error"]

    def test_error_metadata_excluded_from_feeds(self) -> None:
        """Error metadata should have error set and feed_eligible=False."""
        error_metadata = Metadata(
            title="Failed",
            description="",
            filename="bad.mp4",
            filepath="videos/bad.mp4",
            filetype="mp4",
            date=1728931200,
            size=0,
            error="Processing failed",
        )

        success_metadata = Metadata(
            title="Good",
            description="",
            filename="good.mp4",
            filepath="videos/good.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            content="Transcript",
            summary="Summary",
            transcribe=StepMap(status="complete", input_size=100, output_size=0),
            summarize=StepMap(status="complete", input_size=0, output_size=0),
        )

        all_metadata = [error_metadata, success_metadata]
        # Simulate feed filtering
        valid_for_feed = [m for m in all_metadata if m.feed_eligible]

        assert len(valid_for_feed) == 1
        assert valid_for_feed[0].filename == "good.mp4"

    def test_error_metadata_has_required_fields(self) -> None:
        """Error metadata should have all required fields with correct values."""
        exc = Exception("Test error")
        error_dict = create_error_metadata(
            exc=exc,
            filename="test.pdf",
            filepath="docs/test.pdf",
            filetype="pdf",
        )

        required_fields = [
            "title",
            "description",
            "filename",
            "filepath",
            "filetype",
            "date",
            "size",
            "error",
        ]
        for field in required_fields:
            assert field in error_dict

        assert error_dict["error"] == "Test error"
        assert error_dict["title"] == {"filename": "test", "extracted": None, "generated": None}
        assert error_dict["size"] == 0

    def test_producer_handles_error_gracefully(self, tmp_path):
        """Producer should handle errors and create error metadata without crashing."""
        transcriber = FailingTranscriber()
        mp4_file = tmp_path / "bad.mp4"
        mp4_file.write_bytes(b"bad")

        try:
            transcript = transcriber.transcribe(str(mp4_file))
            # Should not reach here
            Metadata(
                title="Test",
                description="Test",
                filename="bad.mp4",
                filepath="bad.mp4",
                filetype="mp4",
                date=1728931200,
                size=100,
                content=transcript,
                summary="Summary",
                transcribe=StepMap(status="complete", input_size=100, output_size=0),
                summarize=StepMap(status="complete", input_size=0, output_size=0),
            )
        except TranscriptionError as exc:
            # Create error metadata as producer would
            error_metadata = Metadata(
                title="bad.mp4",
                description="",
                filename="bad.mp4",
                filepath="bad.mp4",
                filetype="mp4",
                date=1728931200,
                size=0,
                error=str(exc),
            )

        assert error_metadata.error is not None
        assert error_metadata.feed_eligible is False
        # Ensure no transcript/summary for error metadata
        assert error_metadata.content is None
        assert error_metadata.summary is None
