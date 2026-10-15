"""Tests for the Metadata model."""

import json

from pydantic import ValidationError
import pytest

from local_feed_generator.metadata import Metadata


class TestValidMetadata:
    """Tests for valid metadata creation."""

    def test_create_complete_metadata_mp4(self):
        """A valid MP4 complete metadata should be created without errors."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            transcript="# Lecture 1\n\nWelcome to the course.",
            summary="Key takeaways: Introduction to ILE concepts.",
            processing_date=1728931200,
        )
        assert metadata.status == {"transcribe": "complete", "summarize": "complete"}
        assert metadata.filetype == "mp4"
        assert metadata.transcript is not None
        assert metadata.summary is not None
        assert metadata.error is None
        assert metadata.feed_eligible is True

    def test_create_complete_metadata_pdf(self):
        """A valid PDF complete metadata should be created without errors."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Document 1",
            description="A PDF document",
            filename="doc.pdf",
            filepath="documents/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=500000,
            processing_date=1728931200,
        )
        assert metadata.status == {"extract": "complete"}
        assert metadata.filetype == "pdf"
        assert metadata.transcript is None
        assert metadata.summary is None
        assert metadata.error is None
        assert metadata.feed_eligible is True

    def test_create_complete_metadata_html(self):
        """A valid HTML complete metadata should be created without errors."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Web Article",
            description="An HTML article",
            filename="article.html",
            filepath="documents/article.html",
            filetype="html",
            date=1728931200,
            size=150000,
            processing_date=1728931200,
        )
        assert metadata.status == {"extract": "complete"}
        assert metadata.filetype == "html"
        assert metadata.feed_eligible is True

    def test_create_in_progress_metadata_mp4(self):
        """An in_progress metadata (transcription started) should be created."""
        metadata = Metadata(
            status={"transcribe": "in_progress", "summarize": None},
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            processing_date=1728931200,
            started_at=1728931200,
            transcription_model="large-v3-turbo",
        )
        assert metadata.status["transcribe"] == "in_progress"
        assert metadata.status["summarize"] is None
        assert metadata.started_at is not None
        assert metadata.transcription_model == "large-v3-turbo"
        assert metadata.completed_at is None
        assert metadata.error is None
        assert metadata.feed_eligible is False

    def test_create_in_progress_metadata_pdf(self):
        """An in_progress metadata (extraction started) should be created."""
        metadata = Metadata(
            status={"extract": "in_progress"},
            title="Document 1",
            description="A PDF document",
            filename="doc.pdf",
            filepath="documents/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=500000,
            processing_date=1728931200,
            started_at=1728931200,
        )
        assert metadata.status["extract"] == "in_progress"
        assert metadata.feed_eligible is False

    def test_create_complete_metadata_with_all_fields(self):
        """A complete metadata with all new fields should be created."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            transcript="# Lecture 1\n\nWelcome.",
            summary="Key takeaways.",
            processing_date=1728931200,
            transcription_model="large-v3-turbo",
            started_at=1728931200,
            completed_at=1728931200,
        )
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"
        assert metadata.transcription_model == "large-v3-turbo"
        assert metadata.started_at is not None
        assert metadata.completed_at is not None
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
            processing_date=1728931200,
            error="Transcription failed: corrupted audio",
        )
        assert metadata.error == "Transcription failed: corrupted audio"
        assert metadata.feed_eligible is False

    def test_version_field_not_in_model(self):
        """version field should not exist in the Metadata model."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert not hasattr(metadata, "version")
        assert "version" not in metadata.to_dict()
        assert "version" not in json.loads(metadata.to_json())

    def test_new_fields_default_to_none(self):
        """New fields should default to None when not provided."""
        metadata = Metadata(
            status={"transcribe": "in_progress"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.status["transcribe"] == "in_progress"
        assert metadata.status.get("summarize") is None
        assert metadata.transcription_model is None
        assert metadata.started_at is None
        assert metadata.completed_at is None


class TestFeedEligible:
    """Tests for the feed_eligible computed property."""

    def test_mp4_feed_eligible_when_both_complete(self):
        """MP4 should be feed_eligible when transcribe and summarize are complete."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is True

    def test_mp4_not_feed_eligible_when_transcribe_incomplete(self):
        """MP4 should not be feed_eligible when transcribe is not complete."""
        metadata = Metadata(
            status={"transcribe": "in_progress", "summarize": "complete"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is False

    def test_mp4_not_feed_eligible_when_summarize_incomplete(self):
        """MP4 should not be feed_eligible when summarize is not complete."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": None},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is False

    def test_pdf_feed_eligible_when_extract_complete(self):
        """PDF should be feed_eligible when extract is complete."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is True

    def test_html_feed_eligible_when_extract_complete(self):
        """HTML should be feed_eligible when extract is complete."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Test",
            description="Test",
            filename="test.html",
            filepath="test.html",
            filetype="html",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is True

    def test_pdf_not_feed_eligible_when_extract_incomplete(self):
        """PDF should not be feed_eligible when extract is not complete."""
        metadata = Metadata(
            status={"extract": "in_progress"},
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is False

    def test_not_feed_eligible_when_error_set(self):
        """File should not be feed_eligible when error is set."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
            error="Some error",
        )
        assert metadata.feed_eligible is False

    def test_mp4_not_feed_eligible_when_status_none(self):
        """MP4 should not be feed_eligible when status is None."""
        metadata = Metadata(
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        assert metadata.feed_eligible is False

    def test_mp4_not_feed_eligible_when_status_missing_keys(self):
        """MP4 should not be feed_eligible when status dict is missing keys."""
        metadata = Metadata(
            status={"transcribe": "complete"},  # missing summarize
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
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
                processing_date=1728931200,
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
                processing_date=1728931200,
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
                processing_date=1728931200,
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
                processing_date=1728931200,
                error="   ",
            )


class TestConditionalFieldsValidation:
    """Tests for conditional field validation (transcript/summary only for MP4)."""

    def test_transcript_not_allowed_for_pdf(self):
        """transcript should not be allowed for filetype='pdf'."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                status={"extract": "complete"},
                title="Test",
                description="Test",
                filename="test.pdf",
                filepath="test.pdf",
                filetype="pdf",
                date=1728931200,
                size=100,
                processing_date=1728931200,
                transcript="Some transcript",
            )
        assert "transcript" in str(exc_info.value).lower()

    def test_summary_not_allowed_for_html(self):
        """summary should not be allowed for filetype='html'."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                status={"extract": "complete"},
                title="Test",
                description="Test",
                filename="test.html",
                filepath="test.html",
                filetype="html",
                date=1728931200,
                size=100,
                processing_date=1728931200,
                summary="Some summary",
            )
        assert "summary" in str(exc_info.value).lower()

    def test_transcript_allowed_for_mp4(self):
        """transcript should be allowed for filetype='mp4'."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
            transcript="Transcript here",
            summary="Summary here",
        )
        assert metadata.transcript == "Transcript here"
        assert metadata.summary == "Summary here"


class TestExtraFields:
    """Tests for extra field rejection."""

    def test_extra_field_rejected(self):
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(
                status={"extract": "complete"},
                title="Test",
                description="Test",
                filename="test.mp4",
                filepath="test.mp4",
                filetype="mp4",
                date=1728931200,
                size=100,
                processing_date=1728931200,
                extra_field="should fail",
            )
        assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


class TestSizeValidation:
    """Tests for size field validation."""

    def test_negative_size_rejected(self):
        """Negative file size should be rejected."""
        with pytest.raises(ValidationError):
            Metadata(
                status={"extract": "complete"},
                title="Test",
                description="Test",
                filename="test.mp4",
                filepath="test.mp4",
                filetype="mp4",
                date=1728931200,
                size=-1,
                processing_date=1728931200,
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
            processing_date=1728931200,
            error="File not found",
        )
        assert metadata.size == 0


class TestSerialization:
    """Tests for serialization methods."""

    def test_to_json_complete_mp4(self):
        """to_json should produce valid JSON matching the schema."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            transcript="# Lecture 1\n\nWelcome.",
            summary="Key takeaways.",
            processing_date=1728931200,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["status"]["transcribe"] == "complete"
        assert data["status"]["summarize"] == "complete"
        assert data["title"] == "Lecture 1"
        assert data["filename"] == "lecture.mp4"
        assert data["filepath"] == "videos/lecture.mp4"
        assert data["filetype"] == "mp4"
        assert data["size"] == 12345678
        assert data["date"] == 1728931200
        assert data["transcript"] == "# Lecture 1\n\nWelcome."
        assert data["summary"] == "Key takeaways."
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
            processing_date=1728931200,
            error="Transcription failed",
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["error"] == "Transcription failed"
        assert "transcript" not in data
        assert "summary" not in data

    def test_to_json_in_progress_mp4(self):
        """to_json should include status dict when in progress."""
        metadata = Metadata(
            status={"transcribe": "in_progress", "summarize": None},
            title="Lecture 1",
            description="Introduction to ILE",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            processing_date=1728931200,
            started_at=1728931200,
            transcription_model="large-v3-turbo",
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["status"]["transcribe"] == "in_progress"
        assert data["transcription_model"] == "large-v3-turbo"
        assert data["started_at"] == 1728931200
        assert "completed_at" not in data
        assert "error" not in data

    def test_to_json_excludes_none_fields(self):
        """to_json should exclude None fields when not set."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="docs/test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            processing_date=1728931200,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert "transcript" not in data
        assert "summary" not in data
        assert "error" not in data
        assert "transcription_model" not in data
        assert "started_at" not in data
        assert "completed_at" not in data

    def test_to_dict_complete(self):
        """to_dict should produce a dict matching the schema."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Test",
            description="Test",
            filename="test.pdf",
            filepath="docs/test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            processing_date=1728931200,
        )
        data = metadata.to_dict()
        assert data["status"]["extract"] == "complete"
        assert data["title"] == "Test"
        assert data["filetype"] == "pdf"
        assert "error" not in data
        assert "transcript" not in data
        assert "summary" not in data


class TestDeserialization:
    """Tests for deserialization methods."""

    def test_from_json_complete_mp4(self):
        """from_json should deserialize a valid JSON string."""
        json_str = json.dumps(
            {
                "status": {"transcribe": "complete", "summarize": "complete"},
                "title": "Lecture 1",
                "description": "Introduction",
                "filename": "lecture.mp4",
                "filepath": "videos/lecture.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 12345678,
                "transcript": "Transcript here",
                "summary": "Summary here",
                "processing_date": 1728931200,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"
        assert metadata.title == "Lecture 1"
        assert metadata.filetype == "mp4"
        assert metadata.transcript == "Transcript here"
        assert metadata.summary == "Summary here"
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
                "processing_date": 1728931200,
                "error": "Transcription failed",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.error == "Transcription failed"
        assert metadata.feed_eligible is False

    def test_from_json_in_progress_with_new_fields(self):
        """from_json should deserialize in_progress metadata with status dict."""
        json_str = json.dumps(
            {
                "status": {"transcribe": "in_progress", "summarize": None},
                "title": "Processing",
                "description": "In progress",
                "filename": "proc.mp4",
                "filepath": "videos/proc.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
                "started_at": 1728931200,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.status["transcribe"] == "in_progress"
        assert metadata.status["summarize"] is None
        assert metadata.started_at is not None
        assert metadata.feed_eligible is False

    def test_from_dict_complete(self):
        """from_dict should deserialize a valid dictionary."""
        data = {
            "status": {"extract": "complete"},
            "title": "Test",
            "description": "Test",
            "filename": "test.pdf",
            "filepath": "docs/test.pdf",
            "filetype": "pdf",
            "date": 1728931200,
            "size": 5000,
            "processing_date": 1728931200,
        }
        metadata = Metadata.from_dict(data)
        assert metadata.status["extract"] == "complete"
        assert metadata.filetype == "pdf"
        assert metadata.feed_eligible is True

    def test_from_dict_backward_compatible_without_new_fields(self):
        """Old metadata files without status dict should load successfully."""
        # Simulate an old-style metadata file (pre-status-dict)
        old_data = {
            "title": "Old File",
            "description": "Old description",
            "filename": "old.mp4",
            "filepath": "videos/old.mp4",
            "filetype": "mp4",
            "date": 1728931200,
            "size": 5000,
            "transcript": "Old transcript",
            "summary": "Old summary",
            "processing_date": 1728931200,
        }
        metadata = Metadata.from_dict(old_data)
        assert metadata.status is None
        assert metadata.transcript == "Old transcript"
        assert metadata.summary == "Old summary"
        assert metadata.feed_eligible is False  # No status dict set

    def test_from_json_missing_required_field(self):
        """from_json should reject missing required fields."""
        json_str = json.dumps(
            {
                "title": "Test",
                # missing description, filename, filepath, filetype, size, processing_date
            }
        )
        with pytest.raises(ValidationError):
            Metadata.from_json(json_str)

    def test_from_json_backward_compatible_with_legacy_status_fields(self):
        """Old JSON metadata files with individual status fields should load successfully."""
        json_str = json.dumps(
            {
                "transcribe_status": "complete",
                "summarize_status": "complete",
                "title": "Lecture 1",
                "description": "Introduction",
                "filename": "lecture.mp4",
                "filepath": "videos/lecture.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 12345678,
                "transcript": "Transcript here",
                "summary": "Summary here",
                "processing_date": 1728931200,
            }
        )
        metadata = Metadata.from_json(json_str)
        # Old status fields are ignored (extra='ignore'), status is None
        assert metadata.status is None
        # feed_eligible is False because status dict is not populated
        assert metadata.feed_eligible is False


class TestOldFieldWarnings:
    """Tests for warnings when old deprecated fields are found during metadata loading."""

    def test_from_json_file_warns_old_fields(self, tmp_path, capsys):
        """from_json_file should print warnings for old deprecated fields."""
        json_content = json.dumps(
            {
                "transcribe_status": "complete",
                "summarize_status": "complete",
                "extract_status": "complete",
                "stage": "processing",
                "title": "Old File",
                "description": "Old description",
                "filename": "old.mp4",
                "filepath": "videos/old.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "old.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        # Metadata should still be created successfully
        assert metadata.title == "Old File"
        # Old fields are ignored, status is None
        assert metadata.status is None

        # Check warnings were printed to stderr
        captured = capsys.readouterr()
        assert "transcribe_status" in captured.err
        assert "summarize_status" in captured.err
        assert "extract_status" in captured.err
        assert "stage" in captured.err
        assert "deprecated" in captured.err
        assert "old.mp4.json" in captured.err  # filename included

    def test_from_json_warns_old_fields(self, capsys):
        """from_json should print warnings for old deprecated fields."""
        json_str = json.dumps(
            {
                "transcribe_status": "complete",
                "title": "Old File",
                "description": "Old description",
                "filename": "old.mp4",
                "filepath": "videos/old.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.title == "Old File"

        captured = capsys.readouterr()
        assert "transcribe_status" in captured.err
        assert "deprecated" in captured.err

    def test_from_dict_warns_old_fields(self, capsys):
        """from_dict should print warnings for old deprecated fields."""
        data = {
            "summarize_status": "complete",
            "stage": "done",
            "title": "Old File",
            "description": "Old description",
            "filename": "old.mp4",
            "filepath": "videos/old.mp4",
            "filetype": "mp4",
            "date": 1728931200,
            "size": 100,
            "processing_date": 1728931200,
        }
        metadata = Metadata.from_dict(data)
        assert metadata.title == "Old File"

        captured = capsys.readouterr()
        assert "summarize_status" in captured.err
        assert "stage" in captured.err

    def test_warns_old_status_string_field(self, tmp_path, capsys):
        """A string status field (old format) should warn and be removed."""
        json_content = json.dumps(
            {
                "status": "complete",  # old string format, not dict
                "title": "Old File",
                "description": "Old description",
                "filename": "old.mp4",
                "filepath": "videos/old.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "old.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        # Old string status is removed, status defaults to None
        assert metadata.status is None
        assert metadata.feed_eligible is False

        captured = capsys.readouterr()
        assert "status" in captured.err
        assert "deprecated" in captured.err

    def test_no_warning_for_new_format_status(self, tmp_path, capsys):
        """A dict status field (new format) should NOT trigger a warning."""
        json_content = json.dumps(
            {
                "status": {"transcribe": "complete", "summarize": "complete"},
                "title": "New File",
                "description": "New description",
                "filename": "new.mp4",
                "filepath": "videos/new.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "new.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        assert metadata.status == {"transcribe": "complete", "summarize": "complete"}

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_no_warning_for_clean_metadata(self, tmp_path, capsys):
        """Clean new-format metadata should not produce any warnings."""
        json_content = json.dumps(
            {
                "status": {"extract": "complete"},
                "title": "Clean File",
                "description": "Clean description",
                "filename": "clean.pdf",
                "filepath": "docs/clean.pdf",
                "filetype": "pdf",
                "date": 1728931200,
                "size": 5000,
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "clean.pdf.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        assert metadata.feed_eligible is True

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_warning_message_format(self, tmp_path, capsys):
        """Warning message should match the expected format."""
        json_content = json.dumps(
            {
                "transcribe_status": "complete",
                "title": "Test",
                "description": "Test",
                "filename": "test.mp4",
                "filepath": "test.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "test.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        Metadata.from_json_file(json_path)

        captured = capsys.readouterr()
        expected = (
            "Warning: Old JSON field 'transcribe_status' found in test.mp4.json. "
            "This field is deprecated. Please regenerate metadata using the new schema."
        )
        assert expected in captured.err

    def test_multiple_old_fields_each_warned(self, tmp_path, capsys):
        """Each old field should produce its own warning line."""
        json_content = json.dumps(
            {
                "transcribe_status": "complete",
                "summarize_status": "in_progress",
                "extract_status": "complete",
                "stage": "done",
                "title": "Test",
                "description": "Test",
                "filename": "test.mp4",
                "filepath": "test.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "test.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        Metadata.from_json_file(json_path)

        captured = capsys.readouterr()
        # Each old field should have its own warning
        assert captured.err.count("Warning:") == 4
        assert "transcribe_status" in captured.err
        assert "summarize_status" in captured.err
        assert "extract_status" in captured.err
        assert "stage" in captured.err


class TestJsonFileIO:
    """Tests for JSON file write/read methods."""

    def test_to_json_file_complete(self, tmp_path):
        """to_json_file should write valid JSON to a file."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Lecture 1",
            description="Introduction",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            size=12345678,
            transcript="Transcript here",
            summary="Summary here",
            processing_date=1728931200,
        )
        output_path = tmp_path / "lecture.mp4.json"
        metadata.to_json_file(output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["status"]["transcribe"] == "complete"
        assert data["status"]["summarize"] == "complete"
        assert data["title"] == "Lecture 1"
        assert data["filetype"] == "mp4"

    def test_to_json_file_creates_parent_dirs(self, tmp_path):
        """to_json_file should create parent directories if needed."""
        metadata = Metadata(
            status={"extract": "complete"},
            title="Test",
            description="Test",
            filename="doc.pdf",
            filepath="subdir/doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            processing_date=1728931200,
        )
        output_path = tmp_path / "subdir" / "doc.pdf.json"
        metadata.to_json_file(output_path)

        assert output_path.exists()

    def test_from_json_file_complete(self, tmp_path):
        """from_json_file should read and parse a JSON metadata file."""
        json_content = json.dumps(
            {
                "status": {"transcribe": "complete", "summarize": "complete"},
                "title": "Test",
                "description": "Test",
                "filename": "test.mp4",
                "filepath": "test.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
                "transcript": "Transcript",
                "summary": "Summary",
                "processing_date": 1728931200,
            }
        )
        json_path = tmp_path / "test.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        assert metadata.status["transcribe"] == "complete"
        assert metadata.status["summarize"] == "complete"
        assert metadata.title == "Test"
        assert metadata.filetype == "mp4"
        assert metadata.transcript == "Transcript"
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
                "processing_date": 1728931200,
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
            status={"transcribe": "complete", "summarize": "complete"},
            title="Round Trip",
            description="Test round trip",
            filename="roundtrip.mp4",
            filepath="videos/roundtrip.mp4",
            filetype="mp4",
            date=1728931200,
            size=999999,
            transcript="Original transcript",
            summary="Original summary",
            processing_date=1728931200,
        )
        json_path = tmp_path / "roundtrip.mp4.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.status == original.status
        assert loaded.title == original.title
        assert loaded.description == original.description
        assert loaded.filename == original.filename
        assert loaded.filepath == original.filepath
        assert loaded.filetype == original.filetype
        assert loaded.size == original.size
        assert loaded.transcript == original.transcript
        assert loaded.summary == original.summary
        assert loaded.error == original.error
        assert loaded.feed_eligible == original.feed_eligible

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
            processing_date=1728931200,
            error="Something went wrong",
        )
        json_path = tmp_path / "error.mp4.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.error == "Something went wrong"
        assert loaded.transcript is None
        assert loaded.summary is None
        assert loaded.feed_eligible is False

    def test_roundtrip_in_progress_mp4(self, tmp_path):
        """Writing and reading back in_progress metadata with status dict."""
        original = Metadata(
            status={"transcribe": "in_progress", "summarize": None},
            title="In Progress",
            description="Processing",
            filename="proc.mp4",
            filepath="videos/proc.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
            started_at=1728931200,
            transcription_model="large-v3-turbo",
        )
        json_path = tmp_path / "proc.mp4.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.status["transcribe"] == "in_progress"
        assert loaded.status["summarize"] is None
        assert loaded.transcription_model == "large-v3-turbo"
        assert loaded.started_at == original.started_at
        assert loaded.completed_at is None
        assert loaded.feed_eligible is False


class TestSchemaCompliance:
    """Tests to verify the model matches design doc section 3.3 schema."""

    def test_schema_has_all_required_fields(self):
        """Model should have all fields from design doc section 3.3."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        expected_fields = {
            "status",
            "title",
            "description",
            "filename",
            "filepath",
            "filetype",
            "date",
            "size",
            "transcript",
            "summary",
            "processing_date",
            "transcription_model",
            "started_at",
            "completed_at",
            "error",
        }
        assert expected_fields.issubset(set(properties.keys()))

    def test_schema_does_not_have_old_status_fields(self):
        """Model should not have individual status fields."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        assert "transcribe_status" not in properties
        assert "summarize_status" not in properties
        assert "extract_status" not in properties
        assert "stage" not in properties

    def test_feed_eligible_is_computed(self):
        """feed_eligible should be a computed property that appears in serialization."""
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title="Test",
            description="Test",
            filename="test.mp4",
            filepath="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            processing_date=1728931200,
        )
        # The computed field should appear in model_dump
        data = metadata.model_dump()
        assert "feed_eligible" in data
        assert data["feed_eligible"] is True

    def test_date_fields_are_integer(self):
        """date and processing_date should be integer types (UNIX epoch)."""
        json_schema = Metadata.model_json_schema()
        assert json_schema["properties"]["date"]["type"] == "integer"
        assert json_schema["properties"]["processing_date"]["type"] == "integer"

    def test_size_is_integer(self):
        """size should be an integer type."""
        json_schema = Metadata.model_json_schema()
        assert json_schema["properties"]["size"]["type"] == "integer"

    def test_status_field_is_optional(self):
        """status field should be optional."""
        json_schema = Metadata.model_json_schema()
        assert "status" in json_schema["properties"]
        assert "status" not in json_schema.get("required", [])

    def test_new_datetime_fields_are_optional(self):
        """started_at and completed_at should be optional integer fields."""
        json_schema = Metadata.model_json_schema()
        assert "started_at" in json_schema["properties"]
        assert "completed_at" in json_schema["properties"]
        started_at_schema = json_schema["properties"]["started_at"]
        assert "anyOf" in started_at_schema
        types = {t.get("type") for t in started_at_schema["anyOf"]}
        assert "integer" in types
        assert "null" in types
        assert "started_at" not in json_schema.get("required", [])
        assert "completed_at" not in json_schema.get("required", [])

    def test_transcription_model_is_optional(self):
        """transcription_model should be an optional string field."""
        json_schema = Metadata.model_json_schema()
        assert "transcription_model" in json_schema["properties"]
        transcription_model_schema = json_schema["properties"]["transcription_model"]
        assert "anyOf" in transcription_model_schema
        types = {t.get("type") for t in transcription_model_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
        assert "transcription_model" not in json_schema.get("required", [])

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