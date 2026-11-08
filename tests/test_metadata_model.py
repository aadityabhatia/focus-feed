"""Tests for the Metadata model."""

import json

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
        assert step.error is None
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
            error="Step processing failed",
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
        assert step.error == "Step processing failed"

    def test_step_map_in_progress(self):
        """StepMap should accept 'in_progress' status."""
        step = StepMap(
            status="in_progress",
            input_size=100,
            output_size=0,
        )
        assert step.status == "in_progress"

    @pytest.mark.parametrize(
        "input_size,output_size,words",
        [
            (-1, 0, None),  # negative input_size
            (0, -1, None),  # negative output_size
            (0, 0, -1),  # negative words
        ],
    )
    def test_step_map_validation_errors(self, input_size, output_size, words):
        """StepMap should reject negative sizes and words."""
        kwargs = {"input_size": input_size, "output_size": output_size}
        if words is not None:
            kwargs["words"] = words
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
        "filetype",
        [
            "mp3",
            "m4a",
            "wav",
            "webm",
            "aac",
            "flac",
            "ogg",
            "opus",
            "mp4",
            "mpeg",
            "mpga",
            "mkv",
            "mov",
            "avi",
            "wmv",
        ],
    )
    def test_media_filetypes_require_transcribe_and_summarize(self, filetype):
        """All media filetypes use the transcription and summarization steps."""
        metadata = Metadata(
            transcribe=StepMap(status="complete", input_size=100, output_size=100),
            summarize=StepMap(status="complete", input_size=100, output_size=50),
            title="Test",
            filename=f"test.{filetype}",
            filetype=filetype,
            size=100,
        )
        assert metadata.feed_eligible is True

        metadata.transcribe = StepMap(status="in_progress", input_size=100, output_size=0)
        assert metadata.feed_eligible is False

        metadata.transcribe = StepMap(status="complete", input_size=100, output_size=100)
        metadata.summarize = StepMap(status="in_progress", input_size=100, output_size=0)
        assert metadata.feed_eligible is False

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
            ("docx", {"extract": "complete", "summarize": "complete"}, True),
            ("doc", {"extract": "complete", "summarize": "complete"}, True),
            ("pptx", {"extract": "complete", "summarize": "complete"}, True),
            ("ppt", {"extract": "complete", "summarize": "complete"}, True),
            ("xlsx", {"extract": "complete", "summarize": "complete"}, True),
            ("xls", {"extract": "complete", "summarize": "complete"}, True),
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

    @pytest.mark.parametrize(
        "filetype, errored_step",
        [
            ("mp4", "transcribe"),
            ("mp4", "summarize"),
            ("pdf", "extract"),
            ("pdf", "summarize"),
        ],
    )
    def test_not_feed_eligible_when_applicable_step_has_error(self, filetype, errored_step):
        """An errored applicable step prevents feed eligibility."""
        metadata = Metadata(
            transcribe=StepMap(status="complete", input_size=100, output_size=100)
            if filetype == "mp4"
            else None,
            extract=StepMap(status="complete", input_size=100, output_size=100)
            if filetype == "pdf"
            else None,
            summarize=StepMap(status="complete", input_size=100, output_size=50),
            title="Test",
            filename=f"test.{filetype}",
            filetype=filetype,
            size=100,
        )
        setattr(
            metadata,
            errored_step,
            StepMap(status="in_progress", input_size=100, output_size=0, error="Failed"),
        )

        assert metadata.feed_eligible is False

    @pytest.mark.parametrize(
        "filetype, errored_step",
        [
            ("mp4", "transcribe"),
            ("mp4", "summarize"),
            ("pdf", "extract"),
            ("pdf", "summarize"),
        ],
    )
    def test_complete_step_error_does_not_override_eligibility(self, filetype, errored_step):
        """Step errors are ignored when the corresponding step is complete."""
        metadata = Metadata(
            transcribe=StepMap(status="complete", input_size=100, output_size=100)
            if filetype == "mp4"
            else None,
            extract=StepMap(status="complete", input_size=100, output_size=100)
            if filetype == "pdf"
            else None,
            summarize=StepMap(status="complete", input_size=100, output_size=50),
            title="Test",
            filename=f"test.{filetype}",
            filetype=filetype,
            size=100,
        )
        setattr(
            metadata,
            errored_step,
            StepMap(status="complete", input_size=100, output_size=50, error="Stale"),
        )

        assert metadata.feed_eligible is True


class TestInvalidFields:
    """Tests for fields that accept arbitrary filetype strings."""

    @pytest.mark.parametrize("filetype", ["txt", "docx"])
    def test_unrecognized_filetype_is_stored_but_not_eligible(self, filetype):
        """Unknown filetype strings are preserved and remain feed-ineligible."""
        metadata = Metadata(
            title="Test",
            description="Test",
            filename=f"test.{filetype}",
            filetype=filetype,
            date=1728931200,
            size=100,
        )
        assert metadata.filetype == filetype
        assert metadata.feed_eligible is False


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
