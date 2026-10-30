"""Tests for the Metadata model."""

import json
from typing import Any

from pydantic import ValidationError
import pytest

from focus_feed.metadata import Metadata, StepMap


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

    @pytest.mark.parametrize(
        "input_size,output_size,words,extra_field",
        [
            (-1, 0, None, None),  # negative input_size
            (0, -1, None, None),  # negative output_size
            (0, 0, -1, None),  # negative words
            (0, 0, None, "bad"),  # extra field
        ],
    )
    def test_step_map_validation_errors(self, input_size, output_size, words, extra_field):
        """StepMap should reject negative sizes, words, and extra fields."""
        kwargs = {"input_size": input_size, "output_size": output_size}
        if words is not None:
            kwargs["words"] = words
        if extra_field:
            kwargs["extra_field"] = extra_field
        with pytest.raises(ValidationError):
            StepMap(**kwargs)


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
            filetype="mp4",
            date=1728931200,
            size=12345678,
            content="# Lecture 1\n\nWelcome to the course.",
        )
        assert metadata.transcribe is not None
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize is not None
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
            summarize=StepMap(
                status="complete",
                start_time=1792096505,
                end_time=1792096510,
                input_size=12345,
                output_size=500,
                words=75,
                model="mistralai/Mistral-7B-Instruct",
                input_tokens=200,
                output_tokens=50,
            ),
            title="Document 1",
            description="A PDF document",
            filename="doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=500000,
            content="Extracted plaintext from PDF.",
        )
        assert metadata.extract is not None
        assert metadata.extract.status == "complete"
        assert metadata.summarize is not None
        assert metadata.summarize.status == "complete"
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
            summarize=StepMap(
                status="complete",
                start_time=1792096505,
                end_time=1792096510,
                input_size=12345,
                output_size=500,
                words=75,
                model="mistralai/Mistral-7B-Instruct",
                input_tokens=200,
                output_tokens=50,
            ),
            title="Web Article",
            description="An HTML article",
            filename="article.html",
            filetype="html",
            date=1728931200,
            size=150000,
            content="Plaintext from HTML body.",
        )
        assert metadata.extract is not None
        assert metadata.extract.status == "complete"
        assert metadata.summarize is not None
        assert metadata.summarize.status == "complete"
        assert metadata.filetype == "html"
        assert metadata.feed_eligible is True

    @pytest.mark.parametrize("filetype", ["mp4", "pdf"])
    def test_create_in_progress_metadata(self, filetype):
        """An in_progress metadata should be created with correct status."""
        if filetype == "mp4":
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
                filetype="mp4",
                date=1728931200,
                size=12345678,
            )
            assert metadata.transcribe is not None
            assert metadata.transcribe.status == "in_progress"
            assert metadata.summarize is not None
            assert metadata.summarize.status is None
        else:
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
                filetype="pdf",
                date=1728931200,
                size=500000,
            )
            assert metadata.extract is not None
            assert metadata.extract.status == "in_progress"
        assert metadata.error is None
        assert metadata.feed_eligible is False

    def test_create_error_metadata(self):
        """A valid error metadata should include the error field."""
        metadata = Metadata(
            title="Failed File",
            description="Processing failed",
            filename="bad.mp4",
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
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        assert metadata.transcribe is not None
        assert metadata.transcribe.status == "in_progress"
        assert metadata.summarize is None
        assert metadata.extract is None
        assert metadata.content is None
        assert metadata.error is None


class TestFeedEligible:
    """Tests for the feed_eligible computed property."""

    @pytest.mark.parametrize(
        "filetype,step_map,expected",
        [
            ("mp4", {"transcribe": "complete", "summarize": "complete"}, True),
            ("mp4", {"transcribe": "in_progress", "summarize": "complete"}, False),
            ("mp4", {"transcribe": "complete", "summarize": "in_progress"}, False),
            ("mp4", {"transcribe": None, "summarize": None}, False),
            ("mp4", {"transcribe": "complete", "summarize": None}, False),  # missing summarize
            ("pdf", {"extract": "complete", "summarize": "complete"}, True),
            ("html", {"extract": "complete", "summarize": "complete"}, True),
            ("pdf", {"extract": "in_progress", "summarize": "complete"}, False),
            ("pdf", {"extract": "complete", "summarize": "in_progress"}, False),
            ("pdf", {"extract": "complete", "summarize": None}, False),  # missing summarize
            ("pdf", {"extract": None, "summarize": "complete"}, False),  # missing extract
        ],
    )
    def test_feed_eligible_computation(self, filetype, step_map, expected):
        """feed_eligible should be computed correctly based on step map statuses."""
        kwargs = {
            "title": "Test",
            "description": "Test",
            "filename": f"test.{filetype}",
            "filepath": f"test.{filetype}",
            "filetype": filetype,
            "date": 1728931200,
            "size": 100,
        }
        if filetype == "mp4":
            if step_map["transcribe"] is not None:
                kwargs["transcribe"] = StepMap(
                    status=step_map["transcribe"], input_size=100, output_size=0
                )
            if step_map.get("summarize") is not None:
                kwargs["summarize"] = StepMap(
                    status=step_map["summarize"], input_size=100, output_size=0
                )
        else:
            if step_map["extract"] is not None:
                kwargs["extract"] = StepMap(
                    status=step_map["extract"], input_size=100, output_size=0
                )
            if step_map.get("summarize") is not None:
                kwargs["summarize"] = StepMap(
                    status=step_map["summarize"], input_size=100, output_size=0
                )
        metadata = Metadata(**kwargs)
        assert metadata.feed_eligible is expected

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
            filetype="mp4",
            date=1728931200,
            size=100,
            error="Some error",
        )
        assert metadata.feed_eligible is False


class TestInvalidFields:
    """Tests for invalid field values."""

    @pytest.mark.parametrize("filetype", ["txt", "docx"])
    def test_invalid_filetype_rejected(self, filetype):
        """filetype must be 'mp4', 'pdf', or 'html'."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                title="Test",
                description="Test",
                filename=f"test.{filetype}",
                filetype=filetype,
                date=1728931200,
                size=100,
            )
        assert "filetype" in str(exc_info.value).lower()


class TestErrorFieldValidation:
    """Tests for error field validation."""

    @pytest.mark.parametrize("error_value", ["", "   "])
    def test_empty_or_whitespace_error_rejected(self, error_value):
        """error field must not be empty or whitespace-only."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                title="Test",
                description="Test",
                filename="test.mp4",
                filetype="mp4",
                date=1728931200,
                size=100,
                error=error_value,
            )
        assert "error" in str(exc_info.value).lower()


class TestSerialization:
    """Tests for serialization methods."""

    @pytest.mark.parametrize(
        "status,include_error",
        [
            ("complete", False),  # Complete MP4 with all fields
            ("in_progress", False),  # In-progress MP4
            ("error", True),  # Error metadata (no step maps)
        ],
    )
    def test_to_json_serialization(self, status, include_error):
        """to_json should correctly serialize metadata with various states."""
        kwargs: dict[str, Any] = {
            "title": "Test",
            "description": "Test desc",
            "filename": "test.mp4",
            "filepath": "test.mp4",
            "filetype": "mp4",
            "date": 1728931200,
            "size": 100,
        }
        if status == "complete":
            kwargs["transcribe"] = StepMap(
                status="complete", input_size=12345678, output_size=456789
            )
            kwargs["summarize"] = StepMap(status="complete", input_size=456789, output_size=1997)
            kwargs["content"] = "Transcript here"
        elif status == "in_progress":
            kwargs["transcribe"] = StepMap(
                status="in_progress",
                start_time=1728931200,
                input_size=12345678,
                output_size=0,
                model="large-v3-turbo",
            )
        elif status == "error":
            kwargs["error"] = "Transcription failed"

        metadata = Metadata(**kwargs)
        json_str = metadata.to_json()
        data = json.loads(json_str)

        if status == "complete":
            assert data["transcribe"]["status"] == "complete"
            assert data["summarize"]["status"] == "complete"
            assert data["content"] == "Transcript here"
        elif status == "in_progress":
            assert data["transcribe"]["status"] == "in_progress"
            assert data["transcribe"]["model"] == "large-v3-turbo"
            assert "error" not in data
        elif status == "error":
            assert data["error"] == "Transcription failed"
            assert "transcribe" not in data
            assert "summarize" not in data

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

    @pytest.mark.parametrize(
        "filetype,expected_content",
        [
            ("pdf", "This is the extracted plaintext."),
            ("html", "Plaintext from HTML."),
        ],
    )
    def test_to_json_with_content(self, filetype, expected_content):
        """to_json should include content field for PDF/HTML with content."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000 if filetype == "pdf" else 3000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename=f"test.{filetype}",
            filetype=filetype,
            date=1728931200,
            size=5000,
            content=expected_content,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["content"] == expected_content

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
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert "transcribe" not in data
        assert "summarize" not in data

    def test_to_dict_with_generate_title(self):
        """to_dict should include generate_title when set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            generate_title=StepMap(
                status="complete",
                start_time=1792096520,
                end_time=1792096525,
                input_size=456789,
                output_size=24,
                words=4,
                model="test-model",
                input_tokens=456,
                output_tokens=8,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = metadata.to_dict()
        assert data["generate_title"]["status"] == "complete"
        assert data["generate_title"]["model"] == "test-model"
        assert data["generate_title"]["input_tokens"] == 456
        assert data["generate_title"]["output_tokens"] == 8

    def test_to_dict_excludes_generate_title_when_none(self):
        """to_dict should exclude generate_title when it is None."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = metadata.to_dict()
        assert "generate_title" not in data


class TestDeserialization:
    """Tests for deserialization methods."""

    @pytest.mark.parametrize(
        "filetype,expected_content",
        [
            ("pdf", "This is the extracted plaintext."),
            ("html", "Plaintext from HTML."),
        ],
    )
    def test_from_json_with_content(self, filetype, expected_content):
        """from_json should deserialize a PDF/HTML with content field."""
        json_str = json.dumps(
            {
                "extract": {
                    "status": "complete",
                    "input_size": 5000,
                    "output_size": 1234,
                },
                "summarize": {
                    "status": "complete",
                    "input_size": 1234,
                    "output_size": 500,
                },
                "title": "Test Doc",
                "description": "A document",
                "filename": f"doc.{filetype}",
                "filepath": f"docs/doc.{filetype}",
                "filetype": filetype,
                "date": 1728931200,
                "size": 5000,
                "content": expected_content,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.content == expected_content
        assert metadata.feed_eligible is True

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
        assert metadata.transcribe is not None
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize is not None
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
        assert metadata.transcribe is not None
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
            "summarize": {
                "status": "complete",
                "input_size": 1234,
                "output_size": 500,
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
        assert metadata.extract is not None
        assert metadata.extract.status == "complete"
        assert metadata.summarize is not None
        assert metadata.summarize.status == "complete"
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
            filetype="mp4",
            date=1728931200,
            size=100,
            extra_field="should be ignored",  # type: ignore[call-arg]
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
            filetype="mp4",
            date=1728931200,
            size=0,
            error="File not found",
        )
        assert metadata.size == 0


class TestJsonFileIO:
    """Tests for JSON file write/read methods."""

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
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        output_path = tmp_path / "subdir" / "doc.pdf.json"
        metadata.to_json_file(output_path)

        assert output_path.exists()

    @pytest.mark.parametrize("filetype", ["mp4", "pdf"])
    def test_roundtrip_complete_metadata(self, tmp_path, filetype):
        """Writing and reading back should produce equivalent metadata."""
        if filetype == "mp4":
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
                filetype="mp4",
                date=1728931200,
                mtime=1728931201,
                size=999999,
                content="Original transcript",
            )
            json_path = tmp_path / "roundtrip.mp4.json"
        else:
            original = Metadata(
                extract=StepMap(
                    status="complete",
                    input_size=5000,
                    output_size=1234,
                ),
                summarize=StepMap(
                    status="complete",
                    input_size=1234,
                    output_size=500,
                ),
                title="PDF Round Trip",
                description="Test round trip",
                filename="doc.pdf",
                filetype="pdf",
                date=1728931200,
                mtime=1728931201,
                size=5000,
                content="This is the extracted plaintext content.",
            )
            json_path = tmp_path / "doc.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.title == original.title
        assert loaded.description == original.description
        assert loaded.filename == original.filename
        assert loaded.filetype == original.filetype
        assert loaded.size == original.size
        assert loaded.content == original.content
        assert loaded.error == original.error
        assert loaded.feed_eligible == original.feed_eligible

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
            "filetype",
            "date",
            "mtime",
            "size",
            "content",
            "duration",
            "creation_time",
            "hash",
            "error",
        }
        assert expected_fields.issubset(set(properties.keys()))
        assert "filepath" not in properties

    def test_schema_does_not_have_old_fields(self):
        """Model should not have old status dict or removed fields."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        assert "status" not in properties
        assert "transcript" not in properties
        assert "processing_date" not in properties
        assert "transcription_remote_model" not in properties
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
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        # The computed field should appear in model_dump
        data = metadata.model_dump()
        assert "feed_eligible" in data
        assert data["feed_eligible"] is True

    def test_date_field_is_number(self):
        """date should be a number type (UNIX epoch float), nullable."""
        json_schema = Metadata.model_json_schema()
        date_schema = json_schema["properties"]["date"]
        # date is float | None, so it uses anyOf with number and null
        assert "anyOf" in date_schema
        types = {t.get("type") for t in date_schema["anyOf"]}
        assert "number" in types
        assert "null" in types

    def test_size_is_integer(self):
        """size should be an integer type."""
        json_schema = Metadata.model_json_schema()
        assert json_schema["properties"]["size"]["type"] == "integer"

    @pytest.mark.parametrize("field_name", ["transcribe", "summarize", "extract"])
    def test_step_fields_are_optional(self, field_name):
        """Step map fields should be optional."""
        json_schema = Metadata.model_json_schema()
        assert field_name in json_schema["properties"]
        assert field_name not in json_schema.get("required", [])

    @pytest.mark.parametrize("field_name", ["description", "summary", "error"])
    def test_optional_string_fields(self, field_name):
        """Optional string fields should be nullable strings."""
        json_schema = Metadata.model_json_schema()
        assert field_name in json_schema["properties"]
        field_schema = json_schema["properties"][field_name]
        assert "anyOf" in field_schema
        types = {t.get("type") for t in field_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
        assert field_name not in json_schema.get("required", [])

    def test_step_map_schema_has_required_fields(self):
        """StepMap should have input_size and output_size as required."""
        json_schema = StepMap.model_json_schema()
        required = json_schema.get("required", [])
        assert "input_size" in required
        assert "output_size" in required

    def test_publication_date_field_exists(self):
        """publication_date field should exist in the Metadata model."""
        json_schema = Metadata.model_json_schema()
        assert "publication_date" in json_schema["properties"]

    def test_publication_date_defaults_to_none(self):
        """publication_date should default to None when not provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        assert metadata.publication_date is None

    def test_publication_date_stored_when_provided(self):
        """publication_date should be stored when provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            publication_date=1728931200.0,
            size=5000,
        )
        assert metadata.publication_date == 1728931200.0

    def test_publication_date_serialized(self):
        """publication_date should be included in JSON serialization when set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            publication_date=1728931200.0,
            size=5000,
        )
        data = json.loads(metadata.to_json())
        assert "publication_date" in data
        assert data["publication_date"] == 1728931200.0

    def test_publication_date_excluded_when_none(self):
        """publication_date should be excluded from JSON when None."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = json.loads(metadata.to_json())
        assert "publication_date" not in data

    def test_hash_field_exists(self):
        """hash field should exist in the Metadata model."""
        json_schema = Metadata.model_json_schema()
        assert "hash" in json_schema["properties"]

    def test_hash_defaults_to_none(self):
        """hash should default to None when not provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        assert metadata.hash is None

    def test_hash_stored_when_provided(self):
        """hash should be stored when provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        assert metadata.hash == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_hash_serialized_when_set(self):
        """hash should be included in JSON serialization when set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        data = json.loads(metadata.to_json())
        assert "hash" in data
        assert data["hash"] == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_hash_excluded_when_none(self):
        """hash should be excluded from JSON serialization when None."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = json.loads(metadata.to_json())
        assert "hash" not in data

    def test_hash_roundtrip(self, tmp_path):
        """hash should survive a write/read roundtrip."""
        original = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        json_path = tmp_path / "test.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.hash == original.hash

    def test_mtime_roundtrip(self, tmp_path):
        """mtime should survive a write/read roundtrip."""
        original = Metadata(
            extract=StepMap(status="complete", input_size=5000, output_size=1234),
            summarize=StepMap(status="complete", input_size=1234, output_size=500),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            mtime=1728931234.5,
            size=5000,
        )
        json_path = tmp_path / "test.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.mtime == original.mtime


# ---------------------------------------------------------------------------
# Title dict tests
# ---------------------------------------------------------------------------


class TestTitleDict:
    """Tests for the title dict structure and final_title precedence."""

    def _make_base_metadata(self, title: dict[str, str | None] | str) -> Metadata:
        """Create a Metadata instance with the given title."""
        metadata = Metadata(
            extract=StepMap(status="complete", input_size=100, output_size=50),
            summarize=StepMap(status="complete", input_size=50, output_size=20),
            title=title,
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        metadata._source_filepath = "test.pdf"
        return metadata

    def test_title_dict_has_all_three_keys(self):
        """Title dict should have filename, extracted, and generated keys."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": "Generated Title"}
        )
        title = metadata.title
        assert isinstance(title, dict)
        assert "filename" in title
        assert "extracted" in title
        assert "generated" in title

    def test_title_dict_missing_keys_are_added(self):
        """Title dict missing keys should have them added with empty strings."""
        metadata = self._make_base_metadata({"filename": "my file"})
        title = metadata.title
        assert isinstance(title, dict)
        assert title["filename"] == "my file"
        assert title["extracted"] == ""
        assert title["generated"] == ""

    def test_title_dict_none_values_preserved(self):
        """Title dict with None values should preserve them."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": None, "generated": None}
        )
        title = metadata.title
        assert isinstance(title, dict)
        assert title["extracted"] is None
        assert title["generated"] is None

    def test_final_title_precedence_generated_over_extracted(self):
        """final_title should prefer generated over extracted."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": "Generated Title"}
        )
        assert metadata.final_title == "Generated Title"

    def test_final_title_precedence_extracted_over_filename(self):
        """final_title should prefer extracted over filename when generated is empty."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": ""}
        )
        assert metadata.final_title == "Extracted Title"

    def test_final_title_precedence_extracted_over_filename_when_generated_none(self):
        """final_title should prefer extracted over filename when generated is None."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": None}
        )
        assert metadata.final_title == "Extracted Title"

    def test_final_title_falls_back_to_filename(self):
        """final_title should fall back to filename when generated and extracted are empty."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "", "generated": ""}
        )
        assert metadata.final_title == "my file"

    def test_final_title_falls_back_to_filename_when_extracted_none(self):
        """final_title should fall back to filename when extracted is None."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": None, "generated": None}
        )
        assert metadata.final_title == "my file"

    def test_final_title_empty_when_all_empty(self):
        """final_title should return empty string when all values are empty."""
        metadata = self._make_base_metadata({"filename": "", "extracted": "", "generated": ""})
        assert metadata.final_title == ""

    def test_final_title_empty_when_all_none(self):
        """final_title should return empty string when all values are None."""
        metadata = self._make_base_metadata(
            {"filename": None, "extracted": None, "generated": None}
        )
        assert metadata.final_title == ""

    def test_final_title_strips_whitespace(self):
        """final_title should treat whitespace-only strings as empty."""
        metadata = self._make_base_metadata(
            {"filename": "  ", "extracted": "  ", "generated": "  "}
        )
        assert metadata.final_title == ""

    def test_final_title_string_title_backward_compat(self):
        """final_title should return string title as-is for backward compat."""
        metadata = self._make_base_metadata("Plain String Title")
        assert metadata.final_title == "Plain String Title"

    def test_final_title_string_title_empty(self):
        """final_title should return empty string for empty string title."""
        metadata = self._make_base_metadata("")
        assert metadata.final_title == ""

    def test_string_title_emits_warning(self, caplog):
        """String title should emit a deprecation warning."""
        with caplog.at_level("WARNING"):
            self._make_base_metadata("Plain String Title")
        assert any(
            "deprecated" in record.message.lower()
            for record in caplog.records
            if record.levelno >= 30
        )

    def test_string_title_warning_includes_filename(self, caplog):
        """String title deprecation warning should include the source filename."""
        with caplog.at_level("WARNING"):
            metadata = self._make_base_metadata("Plain String Title")
        string_title_warnings = [
            record
            for record in caplog.records
            if record.levelno >= 30
            and "deprecated" in record.message.lower()
            and "string titles are deprecated" in record.message.lower()
        ]
        assert string_title_warnings, "Expected a string title deprecation warning"
        assert metadata.filename in string_title_warnings[0].message

    def test_title_dict_missing_keys_emits_warning(self, caplog):
        """Title dict missing keys should emit a warning."""
        with caplog.at_level("WARNING"):
            self._make_base_metadata({"filename": "my file"})
        assert any(
            "missing keys" in record.message.lower()
            for record in caplog.records
            if record.levelno >= 30
        )

    def test_title_dict_serialization_excludes_computed(self):
        """to_json should not include final_title or feed_eligible."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Ext", "generated": "Gen"}
        )
        data = json.loads(metadata.to_json())
        assert "final_title" not in data
        assert "feed_eligible" not in data

    def test_title_dict_serialization_includes_all_keys(self):
        """to_json should include all three title dict keys."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Ext", "generated": "Gen"}
        )
        data = json.loads(metadata.to_json())
        assert data["title"]["filename"] == "my file"
        assert data["title"]["extracted"] == "Ext"
        assert data["title"]["generated"] == "Gen"

    def test_title_dict_serialization_excludes_none_values(self):
        """to_json should exclude None values from title dict."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": None, "generated": None}
        )
        data = json.loads(metadata.to_json())
        # None values are excluded by exclude_none=True
        assert data["title"]["filename"] == "my file"
        assert "extracted" not in data["title"]
        assert "generated" not in data["title"]

    def test_final_title_precedence_full_chain(self):
        """final_title should follow generated > extracted > filename precedence."""
        # Test: generated wins over everything
        metadata = self._make_base_metadata(
            {"filename": "fname", "extracted": "ext", "generated": "gen"}
        )
        assert metadata.final_title == "gen"

        # Test: extracted wins over filename when generated is empty
        metadata = self._make_base_metadata(
            {"filename": "fname", "extracted": "ext", "generated": ""}
        )
        assert metadata.final_title == "ext"

        # Test: filename is the fallback
        metadata = self._make_base_metadata({"filename": "fname", "extracted": "", "generated": ""})
        assert metadata.final_title == "fname"

    def test_title_dict_with_extra_keys_keeps_required(self):
        """Title dict with extra keys should still have the three required keys."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Ext", "generated": "Gen", "extra": "ignored"}
        )
        title = metadata.title
        assert isinstance(title, dict)
        assert title["filename"] == "my file"
        assert title["extracted"] == "Ext"
        assert title["generated"] == "Gen"


class TestModulePrefix:
    """Tests for module prefix application in final_title.

    The module prefix (parent directory name) is prepended to the resolved
    title when the title itself does not already start with the module name.
    This prevents duplicate prefixes when the LLM-generated or extracted
    title already includes the module name.
    """

    def _make_metadata(
        self,
        title: dict[str, str | None] | str,
        filename: str,
        filepath: str = "",
    ) -> Metadata:
        """Create a Metadata instance with the given title and filepath."""
        metadata = Metadata(
            extract=StepMap(status="complete", input_size=100, output_size=50),
            title=title,
            description="Test",
            filename=filename,
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        metadata._source_filepath = filepath
        return metadata

    def test_module_prefix_applied_when_title_does_not_start_with_module(self):
        """Module prefix should be applied when title doesn't start with folder name."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
        )
        assert metadata.final_title == "videos: My Lecture"

    def test_module_prefix_not_applied_when_title_starts_with_module(self):
        """Module prefix should NOT be applied when title already starts with folder name."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "Videos My Lecture"},
            filename="lecture.mp4",
        )
        # Title "Videos My Lecture" starts with "Videos" (case-insensitive),
        # so no prefix is applied to avoid duplication.
        assert metadata.final_title == "Videos My Lecture"

    def test_module_prefix_applied_to_generated_title(self):
        """Module prefix should be applied after precedence resolution (generated)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": "Extracted", "generated": "Generated"},
            filename="lecture.mp4",
            filepath="course1/lecture.mp4",
        )
        assert metadata.final_title == "course1: Generated"

    def test_module_prefix_applied_to_extracted_title(self):
        """Module prefix should be applied after precedence resolution (extracted)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": "Extracted", "generated": ""},
            filename="lecture.mp4",
            filepath="course1/lecture.mp4",
        )
        assert metadata.final_title == "course1: Extracted"

    def test_module_prefix_applied_to_filename_title(self):
        """Module prefix should be applied after precedence resolution (filename)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": "", "generated": ""},
            filename="lecture.mp4",
            filepath="course1/lecture.mp4",
        )
        assert metadata.final_title == "course1: lecture"

    def test_module_prefix_not_applied_at_root_level(self):
        """Module prefix should not be applied when file is at root level (no parent dir)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            filename="lecture.mp4",
        )
        assert metadata.final_title == "My Lecture"

    def test_module_prefix_handles_underscores_in_folder_name(self):
        """Module prefix should handle underscores in folder name (replaced with spaces)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            filename="lecture.mp4",
            filepath="my_course/lecture.mp4",
        )
        assert metadata.final_title == "my course: My Lecture"

    def test_module_prefix_case_insensitive_title_match(self):
        """Module prefix not applied when title starts with module name (case-insensitive).

        Title "Videos My Lecture" starts with "Videos" (case-insensitive),
        so no prefix is applied to avoid duplication.
        """
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "Videos My Lecture"},
            filename="lecture.mp4",
            filepath="Videos/lecture.mp4",
        )
        # Title "Videos My Lecture" starts with "Videos" (case-insensitive),
        # so no prefix is applied to avoid duplication.
        assert metadata.final_title == "Videos My Lecture"

    def test_module_prefix_not_applied_when_title_starts_with_module_underscored(self):
        """Module prefix not applied when title starts with module name (underscores in title).

        Title "Videos_my_topic" → normalized "videos my topic" starts with
        "videos", so no prefix is applied to avoid duplication.
        """
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "Videos_my_topic"},
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
        )
        # Title "Videos_my_topic" → normalized "videos my topic" starts with "videos",
        # so no prefix is applied to avoid duplication.
        assert metadata.final_title == "Videos_my_topic"

    def test_module_prefix_with_string_title(self):
        """Module prefix should be applied to string titles (backward compat)."""
        metadata = self._make_metadata(
            "Plain Title",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
        )
        assert metadata.final_title == "videos: Plain Title"

    def test_module_prefix_not_applied_when_string_title_starts_with_module(self):
        """Module prefix should NOT be applied when string title starts with module name."""
        metadata = self._make_metadata(
            "Videos Lecture",
            filename="lecture.mp4",
        )
        assert metadata.final_title == "Videos Lecture"
