"""Tests for the Metadata model."""

import json

from local_feed_generator.metadata import Metadata, StepMap
from pydantic import ValidationError
import pytest


class TestStepMap:
    """Tests for the StepMap model."""

    def test_step_map_defaults(self):
        """StepMap should have correct defaults for optional fields."""
        step = StepMap(input_size=100, output_size=200)
        assert step.status is None
        assert step.start_time is None
        assert step.end_time is None
        assert step.words is None
        assert step.model is None
        assert step.input_tokens is None
        assert step.output_tokens is None
        assert step.input_size == 100
        assert step.output_size == 200

    def test_step_map_complete(self):
        """StepMap should accept all fields."""
        step = StepMap(
            status="complete",
            start_time=1792096505,
            end_time=1792096515,
            input_size=12345678,
            output_size=456789,
            words=452,
            model="large-v3-turbo",
            input_tokens=1234,
            output_tokens=567,
        )
        assert step.status == "complete"
        assert step.start_time == 1792096505
        assert step.end_time == 1792096515
        assert step.input_size == 12345678
        assert step.output_size == 456789
        assert step.words == 452
        assert step.model == "large-v3-turbo"
        assert step.input_tokens == 1234
        assert step.output_tokens == 567

    def test_step_map_in_progress(self):
        """StepMap should accept 'in_progress' status."""
        step = StepMap(
            status="in_progress",
            input_size=100,
            output_size=0,
        )
        assert step.status == "in_progress"

    def test_step_map_negative_input_size_rejected(self):
        """StepMap should reject negative input_size."""
        with pytest.raises(ValidationError):
            StepMap(input_size=-1, output_size=0)

    def test_step_map_negative_output_size_rejected(self):
        """StepMap should reject negative output_size."""
        with pytest.raises(ValidationError):
            StepMap(input_size=0, output_size=-1)

    def test_step_map_negative_words_rejected(self):
        """StepMap should reject negative words."""
        with pytest.raises(ValidationError):
            StepMap(input_size=0, output_size=0, words=-1)

    def test_step_map_extra_field_rejected(self):
        """StepMap should reject extra fields."""
        with pytest.raises(ValidationError):
            StepMap(input_size=0, output_size=0, extra_field="bad")


class TestValidMetadata:
    """Tests for valid metadata creation."""

    def test_create_complete_metadata_mp4(self):
        """A valid MP4 complete metadata should be created without errors."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                start_time=1792096505,
                end_time=1792096515,
                input_size=12345678,
                output_size=456789,
                words=452,
                model="large-v3-turbo",
                input_tokens=1234,
                output_tokens=567,
            ),
            summarize=StepMap(
                status="complete",
                start_time=1792096515,
                end_time=1792096520,
                input_size=456789,
                output_size=1997,
                words=345,
                model="mistralai/Mistral-7B-Instruct",
                input_tokens=890,
                output_tokens=123,
            ),
            title="Lecture 1",
            description="Key takeaways: Introduction to ILE concepts.",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            content="# Lecture 1\n\nWelcome to the course.",
        )
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"
        assert metadata.filetype == "mp4"
        assert metadata.content is not None
        assert metadata.description is not None
        assert metadata.error is None
        assert metadata.feed_eligible is True

    def test_create_complete_metadata_pdf(self):
        """A valid PDF complete metadata should be created without errors."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                start_time=1792096500,
                end_time=1792096505,
                input_size=1790891,
                output_size=12345,
                words=543,
            ),
            title="Document 1",
            description="A PDF document",
            filename="doc.pdf",
            filepath="documents/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=500000,
            content="Extracted plaintext from PDF.",
        )
        assert metadata.extract.status == "complete"
        assert metadata.filetype == "pdf"
        assert metadata.content is not None
        assert metadata.error is None
        assert metadata.feed_eligible is True

    def test_create_complete_metadata_html(self):
        """A valid HTML complete metadata should be created without errors."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                start_time=1792096500,
                end_time=1792096505,
                input_size=1790891,
                output_size=12345,
                words=543,
            ),
            title="Web Article",
            description="An HTML article",
            filename="article.html",
            filepath="documents/article.html",
            filetype="html",
            date=1728931200,
            size=150000,
            content="Plaintext from HTML body.",
        )
        assert metadata.extract.status == "complete"
        assert metadata.filetype == "html"
        assert metadata.feed_eligible is True

    def test_create_in_progress_metadata_mp4(self):
        """An in_progress metadata (transcription started) should be created."""
        metadata = Metadata(
            transcribe=StepMap(
                status="in_progress",
                start_time=1728931200,
                input_size=12345678,
                output_size=0,
                model="large-v3-turbo",
            ),
            summarize=StepMap(
                input_size=0,
                output_size=0,
            ),
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
        )
        assert metadata.transcribe.status == "in_progress"
        assert metadata.summarize.status is None
        assert metadata.error is None
        assert metadata.feed_eligible is False

    def test_create_in_progress_metadata_pdf(self):
        """An in_progress metadata (extraction started) should be created."""
        metadata = Metadata(
            extract=StepMap(
                status="in_progress",
                start_time=1728931200,
                input_size=500000,
                output_size=0,
            ),
            title="Document 1",
            description="A PDF document",
            filename="doc.pdf",
            filepath="documents/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=500000,
        )
        assert metadata.extract.status == "in_progress"
        assert metadata.feed_eligible is False

    def test_create_complete_metadata_with_all_fields(self):
        """A complete metadata with all new fields should be created."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                start_time=1728931200,
                end_time=1728931200,
                input_size=12345678,
                output_size=456789,
                words=452,
                model="large-v3-turbo",
                input_tokens=1234,
                output_tokens=567,
            ),
            summarize=StepMap(
                status="complete",
                start_time=1728931200,
                end_time=1728931200,
                input_size=456789,
                output_size=1997,
                words=345,
                model="mistralai/Mistral-7B-Instruct",
                input_tokens=890,
                output_tokens=123,
            ),
            title="Lecture 1",
            description="Key takeaways.",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            content="# Lecture 1\n\nWelcome.",
        )
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"
        assert metadata.transcribe.model == "large-v3-turbo"
        assert metadata.feed_eligible is True

    def test_create_error_metadata(self):
        """A valid error metadata should include the error field."""
        metadata = Metadata(
            title="Failed File",
            description="Processing failed",
            filename="bad.mp4",
            filepath="videos/bad.mp4",
            filetype="mp4",
            date=1728931200,
            size=0,
            error="Transcription failed: corrupted audio",
        )
        assert metadata.error == "Transcription failed: corrupted audio"
        assert metadata.feed_eligible is False

    def test_version_field_not_in_model(self):
        """version field should not exist in the Metadata model."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert not hasattr(metadata, "version")
        assert "version" not in metadata.to_dict()
        assert "version" not in json.loads(metadata.to_json())

    def test_new_fields_default_to_none(self):
        """New fields should default to None when not provided."""
        metadata = Metadata(
            transcribe=StepMap(
                status="in_progress",
                input_size=12345678,
                output_size=0,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.transcribe.status == "in_progress"
        assert metadata.summarize is None
        assert metadata.extract is None
        assert metadata.content is None
        assert metadata.error is None


class TestFeedEligible:
    """Tests for the feed_eligible computed property."""

    def test_mp4_feed_eligible_when_both_complete(self):
        """MP4 should be feed_eligible when transcribe and summarize are complete."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is True

    def test_mp4_not_feed_eligible_when_transcribe_incomplete(self):
        """MP4 should not be feed_eligible when transcribe is not complete."""
        metadata = Metadata(
            transcribe=StepMap(
                status="in_progress",
                input_size=12345678,
                output_size=0,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is False

    def test_mp4_not_feed_eligible_when_summarize_incomplete(self):
        """MP4 should not be feed_eligible when summarize is not complete."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="in_progress",
                input_size=456789,
                output_size=0,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is False

    def test_pdf_feed_eligible_when_extract_complete(self):
        """PDF should be feed_eligible when extract is complete."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=500000,
                output_size=12345,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is True

    def test_html_feed_eligible_when_extract_complete(self):
        """HTML should be feed_eligible when extract is complete."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=150000,
                output_size=12345,
            ),
            title="Test",
            description="Test",
            filename="test.html",
            filepath="test.html",
            filetype="html",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is True

    def test_pdf_not_feed_eligible_when_extract_incomplete(self):
        """PDF should not be feed_eligible when extract is not complete."""
        metadata = Metadata(
            extract=StepMap(
                status="in_progress",
                input_size=500000,
                output_size=0,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is False

    def test_not_feed_eligible_when_error_set(self):
        """File should not be feed_eligible when error is set."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            error="Some error",
        )
        assert metadata.feed_eligible is False

    def test_mp4_not_feed_eligible_when_no_step_maps(self):
        """MP4 should not be feed_eligible when step maps are None."""
        metadata = Metadata(
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is False

    def test_mp4_not_feed_eligible_when_summarize_missing(self):
        """MP4 should not be feed_eligible when summarize step map is missing."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.feed_eligible is False


class TestInvalidFields:
    """Tests for invalid field values."""

    def test_invalid_filetype(self):
        """filetype must be 'mp4', 'pdf', or 'html'."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                title="Test",
                description="Test",
                filename="test.txt",
                filepath="test.txt",
                filetype="txt",
                date=1728931200,
                size=100,
            )
        assert "filetype" in str(exc_info.value).lower()

    def test_invalid_filetype_docx(self):
        """filetype 'docx' should be rejected."""
        with pytest.raises(ValidationError):
            Metadata(
                title="Test",
                description="Test",
                filename="test.docx",
                filepath="test.docx",
                filetype="docx",
                date=1728931200,
                size=100,
            )


class TestErrorFieldValidation:
    """Tests for error field validation."""

    def test_empty_error_string_rejected(self):
        """error field must not be an empty string."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                title="Test",
                description="Test",
                filename="test.mp4",
                filepath="test.mp4",
                filetype="mp4",
                date=1728931200,
                size=100,
                error="",
            )
        assert "error" in str(exc_info.value).lower()

    def test_whitespace_error_string_rejected(self):
        """error field must not be whitespace-only."""
        with pytest.raises(ValidationError):
            Metadata(
                title="Test",
                description="Test",
                filename="test.mp4",
                filepath="test.mp4",
                filetype="mp4",
                date=1728931200,
                size=100,
                error="   ",
            )


class TestSerialization:
    """Tests for serialization methods."""

    def test_to_json_complete_mp4(self):
        """to_json should produce valid JSON matching the schema."""
        transcribe_step = StepMap(
            status="complete",
            start_time=1792096505,
            end_time=1792096515,
            input_size=12345678,
            output_size=456789,
            words=452,
            model="large-v3-turbo",
            input_tokens=1234,
            output_tokens=567,
        )
        summarize_step = StepMap(
            status="complete",
            start_time=1792096515,
            end_time=1792096520,
            input_size=456789,
            output_size=1997,
            words=345,
            model="mistralai/Mistral-7B-Instruct",
            input_tokens=890,
            output_tokens=123,
        )
        metadata = Metadata(
            transcribe=transcribe_step,
            summarize=summarize_step,
            title="Lecture 1",
            description="Key takeaways.",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            content="# Lecture 1\n\nWelcome.",
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["transcribe"]["status"] == "complete"
        assert data["transcribe"]["model"] == "large-v3-turbo"
        assert data["summarize"]["status"] == "complete"
        assert data["summarize"]["model"] == "mistralai/Mistral-7B-Instruct"
        assert data["title"] == "Lecture 1"
        assert data["filename"] == "lecture.mp4"
        assert data["filepath"] == "videos/lecture.mp4"
        assert data["filetype"] == "mp4"
        assert data["size"] == 12345678
        assert data["date"] == 1728931200
        assert data["content"] == "# Lecture 1\n\nWelcome."
        assert "error" not in data
        assert "feed_eligible" not in data

    def test_to_json_error(self):
        """to_json should include error field for error metadata."""
        metadata = Metadata(
            title="Failed",
            description="Processing failed",
            filename="bad.mp4",
            filepath="videos/bad.mp4",
            filetype="mp4",
            date=1728931200,
            size=0,
            error="Transcription failed",
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["error"] == "Transcription failed"
        assert "transcribe" not in data
        assert "summarize" not in data

    def test_to_json_in_progress_mp4(self):
        """to_json should include step maps when in progress."""
        metadata = Metadata(
            transcribe=StepMap(
                status="in_progress",
                start_time=1728931200,
                input_size=12345678,
                output_size=0,
                model="large-v3-turbo",
            ),
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["transcribe"]["status"] == "in_progress"
        assert data["transcribe"]["model"] == "large-v3-turbo"
        assert data["transcribe"]["start_time"] == 1728931200
        assert "error" not in data

    def test_to_json_excludes_none_fields(self):
        """to_json should exclude None fields when not set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="docs/test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert "transcribe" not in data
        assert "summarize" not in data
        assert "error" not in data
        assert "content" not in data
        assert "duration" not in data
        assert "creation_time" not in data

    def test_to_dict_complete(self):
        """to_dict should produce a dict matching the schema."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="docs/test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = metadata.to_dict()
        assert data["extract"]["status"] == "complete"
        assert data["title"] == "Test"
        assert data["filetype"] == "pdf"
        assert "error" not in data
        assert "transcribe" not in data
        assert "summarize" not in data

    def test_to_json_pdf_with_content(self):
        """to_json should include content field for PDF with content."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="docs/test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            content="This is the extracted plaintext.",
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["content"] == "This is the extracted plaintext."

    def test_to_json_html_with_content(self):
        """to_json should include content field for HTML with content."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=3000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.html",
            filepath="docs/test.html",
            filetype="html",
            date=1728931200,
            size=5000,
            content="Plaintext from HTML.",
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["content"] == "Plaintext from HTML."

    def test_to_json_mp4_excludes_transcribe_and_summarize_for_pdf(self):
        """to_json should not include transcribe/summarize for PDF."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert "transcribe" not in data
        assert "summarize" not in data


class TestDeserialization:
    """Tests for deserialization methods."""

    def test_from_json_pdf_with_content(self):
        """from_json should deserialize a PDF with content field."""
        json_str = json.dumps(
            {
                "extract": {
                    "status": "complete",
                    "input_size": 5000,
                    "output_size": 1234,
                },
                "title": "Test Doc",
                "description": "A PDF document",
                "filename": "doc.pdf",
                "filepath": "docs/doc.pdf",
                "filetype": "pdf",
                "date": 1728931200,
                "size": 5000,
                "content": "Extracted plaintext from PDF.",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.content == "Extracted plaintext from PDF."
        assert metadata.feed_eligible is True

    def test_from_json_html_with_content(self):
        """from_json should deserialize an HTML with content field."""
        json_str = json.dumps(
            {
                "extract": {
                    "status": "complete",
                    "input_size": 3000,
                    "output_size": 1234,
                },
                "title": "Test Page",
                "description": "An HTML page",
                "filename": "page.html",
                "filepath": "docs/page.html",
                "filetype": "html",
                "date": 1728931200,
                "size": 3000,
                "content": "Plaintext from HTML body.",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.content == "Plaintext from HTML body."
        assert metadata.feed_eligible is True

    def test_from_json_pdf_without_content_defaults_none(self):
        """from_json should default content to None when not present."""
        json_str = json.dumps(
            {
                "extract": {
                    "status": "complete",
                    "input_size": 5000,
                    "output_size": 0,
                },
                "title": "Test Doc",
                "description": "A PDF document",
                "filename": "doc.pdf",
                "filepath": "docs/doc.pdf",
                "filetype": "pdf",
                "date": 1728931200,
                "size": 5000,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.content is None

    def test_from_json_complete_mp4(self):
        """from_json should deserialize a valid JSON string."""
        json_str = json.dumps(
            {
                "transcribe": {
                    "status": "complete",
                    "input_size": 12345678,
                    "output_size": 456789,
                },
                "summarize": {
                    "status": "complete",
                    "input_size": 456789,
                    "output_size": 1997,
                },
                "title": "Lecture 1",
                "description": "Introduction",
                "filename": "lecture.mp4",
                "filepath": "videos/lecture.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 12345678,
                "content": "Transcript here",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"
        assert metadata.title == "Lecture 1"
        assert metadata.filetype == "mp4"
        assert metadata.content == "Transcript here"
        assert metadata.error is None
        assert metadata.feed_eligible is True

    def test_from_json_error(self):
        """from_json should deserialize error metadata."""
        json_str = json.dumps(
            {
                "title": "Failed",
                "description": "Error",
                "filename": "bad.mp4",
                "filepath": "videos/bad.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 0,
                "error": "Transcription failed",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.error == "Transcription failed"
        assert metadata.feed_eligible is False

    def test_from_json_in_progress_with_step_maps(self):
        """from_json should deserialize in_progress metadata with step maps."""
        json_str = json.dumps(
            {
                "transcribe": {
                    "status": "in_progress",
                    "start_time": 1728931200,
                    "input_size": 12345678,
                    "output_size": 0,
                },
                "title": "Processing",
                "description": "In progress",
                "filename": "proc.mp4",
                "filepath": "videos/proc.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.transcribe.status == "in_progress"
        assert metadata.feed_eligible is False

    def test_from_dict_complete(self):
        """from_dict should deserialize a valid dictionary."""
        data = {
            "extract": {
                "status": "complete",
                "input_size": 5000,
                "output_size": 1234,
            },
            "title": "Test",
            "description": "Test",
            "filename": "test.pdf",
            "filepath": "docs/test.pdf",
            "filetype": "pdf",
            "date": 1728931200,
            "size": 5000,
        }
        metadata = Metadata.from_dict(data)
        assert metadata.extract.status == "complete"
        assert metadata.filetype == "pdf"
        assert metadata.feed_eligible is True

    def test_from_json_missing_required_field(self):
        """from_json should reject missing required fields."""
        json_str = json.dumps(
            {
                "title": "Test",
                # missing description, filename, filepath, filetype, size
            }
        )
        with pytest.raises(ValidationError):
            Metadata.from_json(json_str)


class TestExtraFields:
    """Tests for extra field rejection."""

    def test_extra_field_ignored(self):
        """Extra fields should be ignored (extra='ignore')."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            extra_field="should be ignored",
        )
        # Extra fields are ignored, not stored
        assert not hasattr(metadata, "extra_field")
        assert "extra_field" not in metadata.to_dict()


class TestSizeValidation:
    """Tests for size field validation."""

    def test_negative_size_rejected(self):
        """Negative file size should be rejected."""
        with pytest.raises(ValidationError):
            Metadata(
                extract=StepMap(
                    status="complete",
                    input_size=5000,
                    output_size=1234,
                ),
                title="Test",
                description="Test",
                filename="test.mp4",
                filepath="test.mp4",
                filetype="mp4",
                date=1728931200,
                size=-1,
            )

    def test_zero_size_accepted(self):
        """Zero file size should be accepted (e.g., for error cases)."""
        metadata = Metadata(
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=0,
            error="File not found",
        )
        assert metadata.size == 0


class TestJsonFileIO:
    """Tests for JSON file write/read methods."""

    def test_to_json_file_complete(self, tmp_path):
        """to_json_file should write valid JSON to a file."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Lecture 1",
            description="Key takeaways.",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            content="Transcript here",
        )
        output_path = tmp_path / "lecture.mp4.json"
        metadata.to_json_file(output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["transcribe"]["status"] == "complete"
        assert data["summarize"]["status"] == "complete"
        assert data["title"] == "Lecture 1"
        assert data["filetype"] == "mp4"

    def test_to_json_file_creates_parent_dirs(self, tmp_path):
        """to_json_file should create parent directories if needed."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="doc.pdf",
            filepath="subdir/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        output_path = tmp_path / "subdir" / "doc.pdf.json"
        metadata.to_json_file(output_path)

        assert output_path.exists()

    def test_from_json_file_complete(self, tmp_path):
        """from_json_file should read and parse a JSON metadata file."""
        json_content = json.dumps(
            {
                "transcribe": {
                    "status": "complete",
                    "input_size": 12345678,
                    "output_size": 456789,
                },
                "summarize": {
                    "status": "complete",
                    "input_size": 456789,
                    "output_size": 1997,
                },
                "title": "Test",
                "description": "Test",
                "filename": "test.mp4",
                "filepath": "test.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "content": "Transcript",
            }
        )
        json_path = tmp_path / "test.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize.status == "complete"
        assert metadata.title == "Test"
        assert metadata.filetype == "mp4"
        assert metadata.content == "Transcript"
        assert metadata.feed_eligible is True

    def test_from_json_file_error_metadata(self, tmp_path):
        """from_json_file should read error metadata correctly."""
        json_content = json.dumps(
            {
                "title": "Failed",
                "description": "Error",
                "filename": "bad.mp4",
                "filepath": "bad.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 0,
                "error": "Transcription failed",
            }
        )
        json_path = tmp_path / "bad.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        assert metadata.error == "Transcription failed"
        assert metadata.feed_eligible is False

    def test_roundtrip_complete_metadata(self, tmp_path):
        """Writing and reading back should produce equivalent metadata."""
        original = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Round Trip",
            description="Test round trip",
            filename="roundtrip.mp4",
            filepath="videos/roundtrip.mp4",
            filetype="mp4",
            date=1728931200,
            size=999999,
            content="Original transcript",
        )
        json_path = tmp_path / "roundtrip.mp4.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.transcribe.status == original.transcribe.status
        assert loaded.summarize.status == original.summarize.status
        assert loaded.title == original.title
        assert loaded.description == original.description
        assert loaded.filename == original.filename
        assert loaded.filepath == original.filepath
        assert loaded.filetype == original.filetype
        assert loaded.size == original.size
        assert loaded.content == original.content
        assert loaded.error == original.error
        assert loaded.feed_eligible == original.feed_eligible

    def test_roundtrip_pdf_with_content(self, tmp_path):
        """Writing and reading back PDF metadata with content should be equivalent."""
        original = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="PDF Round Trip",
            description="Test round trip",
            filename="doc.pdf",
            filepath="docs/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            content="This is the extracted plaintext content.",
        )
        json_path = tmp_path / "doc.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.content == original.content
        assert loaded.content == "This is the extracted plaintext content."
        assert loaded.feed_eligible is True

    def test_roundtrip_error_metadata(self, tmp_path):
        """Writing and reading back error metadata should produce equivalent metadata."""
        original = Metadata(
            title="Error Round Trip",
            description="Error test",
            filename="error.mp4",
            filepath="videos/error.mp4",
            filetype="mp4",
            date=1728931200,
            size=0,
            error="Something went wrong",
        )
        json_path = tmp_path / "error.mp4.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.error == "Something went wrong"
        assert loaded.content is None
        assert loaded.feed_eligible is False

    def test_roundtrip_in_progress_mp4(self, tmp_path):
        """Writing and reading back in_progress metadata with step maps."""
        original = Metadata(
            transcribe=StepMap(
                status="in_progress",
                start_time=1728931200,
                input_size=12345678,
                output_size=0,
                model="large-v3-turbo",
            ),
            title="In Progress",
            description="Processing",
            filename="proc.mp4",
            filepath="videos/proc.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        json_path = tmp_path / "proc.mp4.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.transcribe.status == "in_progress"
        assert loaded.transcribe.model == "large-v3-turbo"
        assert loaded.transcribe.start_time == 1728931200
        assert loaded.feed_eligible is False


class TestSchemaCompliance:
    """Tests to verify the model matches design doc section 3.3 schema."""

    def test_schema_has_all_required_fields(self):
        """Model should have all fields from design doc section 3.3."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        expected_fields = {
            "transcribe",
            "summarize",
            "extract",
            "title",
            "description",
            "filename",
            "filepath",
            "filetype",
            "date",
            "size",
            "content",
            "duration",
            "creation_time",
            "error",
        }
        assert expected_fields.issubset(set(properties.keys()))

    def test_schema_does_not_have_old_fields(self):
        """Model should not have old status dict or removed fields."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        assert "status" not in properties
        assert "transcript" not in properties
        assert "processing_date" not in properties
        assert "transcription_model" not in properties
        assert "started_at" not in properties
        assert "completed_at" not in properties

    def test_feed_eligible_is_computed(self):
        """feed_eligible should be a computed property that appears in serialization."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        # The computed field should appear in model_dump
        data = metadata.model_dump()
        assert "feed_eligible" in data
        assert data["feed_eligible"] is True

    def test_date_field_is_integer(self):
        """date should be an integer type (UNIX epoch), nullable."""
        json_schema = Metadata.model_json_schema()
        date_schema = json_schema["properties"]["date"]
        # date is int | None, so it uses anyOf with integer and null
        assert "anyOf" in date_schema
        types = {t.get("type") for t in date_schema["anyOf"]}
        assert "integer" in types
        assert "null" in types

    def test_size_is_integer(self):
        """size should be an integer type."""
        json_schema = Metadata.model_json_schema()
        assert json_schema["properties"]["size"]["type"] == "integer"

    def test_transcribe_is_optional(self):
        """transcribe field should be optional."""
        json_schema = Metadata.model_json_schema()
        assert "transcribe" in json_schema["properties"]
        assert "transcribe" not in json_schema.get("required", [])

    def test_summarize_is_optional(self):
        """summarize field should be optional."""
        json_schema = Metadata.model_json_schema()
        assert "summarize" in json_schema["properties"]
        assert "summarize" not in json_schema.get("required", [])

    def test_extract_is_optional(self):
        """extract field should be optional."""
        json_schema = Metadata.model_json_schema()
        assert "extract" in json_schema["properties"]
        assert "extract" not in json_schema.get("required", [])

    def test_description_is_optional(self):
        """description should be an optional string field."""
        json_schema = Metadata.model_json_schema()
        assert "description" in json_schema["properties"]
        description_schema = json_schema["properties"]["description"]
        assert "anyOf" in description_schema
        types = {t.get("type") for t in description_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
        assert "description" not in json_schema.get("required", [])

    def test_summary_is_optional(self):
        """summary should be an optional string field."""
        json_schema = Metadata.model_json_schema()
        assert "summary" in json_schema["properties"]
        summary_schema = json_schema["properties"]["summary"]
        assert "anyOf" in summary_schema
        types = {t.get("type") for t in summary_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
        assert "summary" not in json_schema.get("required", [])

    def test_error_is_optional(self):
        """error should be an optional string field."""
        json_schema = Metadata.model_json_schema()
        assert "error" in json_schema["properties"]
        error_schema = json_schema["properties"]["error"]
        assert "anyOf" in error_schema
        types = {t.get("type") for t in error_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
        assert "error" not in json_schema.get("required", [])

    def test_step_map_schema_has_required_fields(self):
        """StepMap should have input_size and output_size as required."""
        json_schema = StepMap.model_json_schema()
        required = json_schema.get("required", [])
        assert "input_size" in required
        assert "output_size" in required
