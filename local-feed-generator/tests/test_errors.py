"""Tests for the error handling utilities."""

from io import StringIO
import json
import logging

import pytest

from local_feed_generator.errors import (
    ProcessingError,
    SummarizationError,
    TranscriptionError,
    ValidationError,
    create_error_metadata,
    log_error,
)


class TestExceptionHierarchy:
    """Tests for custom exception class hierarchy and raisability."""

    def test_processing_error_can_be_raised(self):
        """ProcessingError should be raisable and catchable."""
        with pytest.raises(ProcessingError):
            raise ProcessingError("something went wrong")

    def test_transcription_error_is_subclass_of_processing_error(self):
        """TranscriptionError should inherit from ProcessingError."""
        assert issubclass(TranscriptionError, ProcessingError)

    def test_summarization_error_is_subclass_of_processing_error(self):
        """SummarizationError should inherit from ProcessingError."""
        assert issubclass(SummarizationError, ProcessingError)

    def test_validation_error_is_not_subclass_of_processing_error(self):
        """ValidationError should be a separate exception hierarchy."""
        assert not issubclass(ValidationError, ProcessingError)

    def test_transcription_error_can_be_raised(self):
        """TranscriptionError should be raisable and catchable."""
        with pytest.raises(TranscriptionError):
            raise TranscriptionError("audio decode failed")

    def test_summarization_error_can_be_raised(self):
        """SummarizationError should be raisable and catchable."""
        with pytest.raises(SummarizationError):
            raise SummarizationError("LLM request failed")

    def test_validation_error_can_be_raised(self):
        """ValidationError should be raisable and catchable."""
        with pytest.raises(ValidationError):
            raise ValidationError("field is required")

    def test_transcription_error_caught_as_processing_error(self):
        """TranscriptionError should be catchable via ProcessingError."""
        with pytest.raises(ProcessingError):
            raise TranscriptionError("gpu out of memory")

    def test_summarization_error_caught_as_processing_error(self):
        """SummarizationError should be catchable via ProcessingError."""
        with pytest.raises(ProcessingError):
            raise SummarizationError("rate limited")

    def test_processing_error_stores_message(self):
        """ProcessingError should store the message attribute."""
        exc = ProcessingError("disk full")
        assert exc.message == "disk full"

    def test_processing_error_stores_filepath(self):
        """ProcessingError should store the filepath attribute."""
        exc = ProcessingError("disk full", filepath="videos/lecture.mp4")
        assert exc.filepath == "videos/lecture.mp4"

    def test_processing_error_filepath_defaults_none(self):
        """ProcessingError filepath should default to None."""
        exc = ProcessingError("disk full")
        assert exc.filepath is None

    def test_validation_error_stores_field(self):
        """ValidationError should store the field attribute."""
        exc = ValidationError("invalid value", field="email")
        assert exc.field == "email"

    def test_validation_error_field_defaults_none(self):
        """ValidationError field should default to None."""
        exc = ValidationError("invalid value")
        assert exc.field is None


class TestCreateErrorMetadata:
    """Tests for the create_error_metadata utility function."""

    def test_returns_dict_with_error_field(self):
        """Error metadata should have error field set."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == "failed"

    def test_includes_filename(self):
        """Error metadata should include the filename."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["filename"] == "lecture.mp4"

    def test_includes_filepath(self):
        """Error metadata should include the filepath."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["filepath"] == "videos/lecture.mp4"

    def test_includes_filetype(self):
        """Error metadata should include the filetype."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["filetype"] == "mp4"

    def test_includes_error_message(self):
        """Error metadata should include the exception message in the error field."""
        exc = ProcessingError("disk full")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == "disk full"

    def test_includes_processing_date(self):
        """Error metadata should include a processing_date timestamp."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert "processing_date" in metadata
        assert metadata["processing_date"] is not None

    def test_includes_date(self):
        """Error metadata should include a date timestamp."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert "date" in metadata
        assert metadata["date"] is not None

    def test_filetype_defaults_to_unknown(self):
        """Error metadata filetype should default to 'unknown' when not provided."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4")
        assert metadata["filetype"] == "unknown"

    def test_works_with_transcription_error(self):
        """create_error_metadata should work with TranscriptionError."""
        exc = TranscriptionError("gpu error")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == "gpu error"

    def test_works_with_summarization_error(self):
        """create_error_metadata should work with SummarizationError."""
        exc = SummarizationError("api timeout")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == "api timeout"

    def test_works_with_validation_error(self):
        """create_error_metadata should work with ValidationError."""
        exc = ValidationError("bad schema")
        metadata = create_error_metadata(exc, "doc.pdf", "documents/doc.pdf", "pdf")
        assert metadata["error"] == "bad schema"

    def test_works_with_generic_exception(self):
        """create_error_metadata should work with any Exception subclass."""
        exc = RuntimeError("unexpected failure")
        metadata = create_error_metadata(exc, "doc.html", "documents/doc.html", "html")
        assert metadata["error"] == "unexpected failure"

    def test_empty_message_uses_class_name(self):
        """Error metadata should use exception class name when message is empty."""
        exc = ProcessingError("")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == "ProcessingError"

    def test_metadata_is_json_serializable(self):
        """Error metadata should be serializable to JSON."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        json_str = json.dumps(metadata)
        restored = json.loads(json_str)
        assert restored["error"] == "failed"

    def test_metadata_has_all_schema_fields(self):
        """Error metadata should include all fields from the JSON schema."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        expected_fields = {
            "status",
            "title",
            "description",
            "filename",
            "filepath",
            "filetype",
            "date",
            "size",
            "processing_date",
            "error",
        }
        assert expected_fields.issubset(set(metadata.keys()))


class TestLogError:
    """Tests for the log_error helper function."""

    @pytest.fixture
    def capture_logger(self):
        """Create a logger with a StringIO handler to capture output."""
        log = logging.getLogger("test_log_error")
        log.handlers = []
        log.setLevel(logging.DEBUG)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handler.setLevel(logging.DEBUG)
        log.addHandler(handler)
        return log, stream

    def test_logs_error_message(self, capture_logger):
        """log_error should include the error message in the log output."""
        log, stream = capture_logger
        exc = ProcessingError("disk full")
        log_error(exc, filepath="videos/lecture.mp4", logger=log)
        output = stream.getvalue()
        assert "disk full" in output

    def test_logs_exception_class_name(self, capture_logger):
        """log_error should include the exception class name in the log output."""
        log, stream = capture_logger
        exc = ProcessingError("disk full")
        log_error(exc, filepath="videos/lecture.mp4", logger=log)
        output = stream.getvalue()
        assert "ProcessingError" in output

    def test_logs_filepath(self, capture_logger):
        """log_error should include the filepath in the log output."""
        log, stream = capture_logger
        exc = ProcessingError("disk full")
        log_error(exc, filepath="videos/lecture.mp4", logger=log)
        output = stream.getvalue()
        assert "videos/lecture.mp4" in output

    def test_logs_timestamp(self, capture_logger):
        """log_error should include a timestamp in the log output."""
        log, stream = capture_logger
        exc = ProcessingError("disk full")
        log_error(exc, filepath="videos/lecture.mp4", logger=log)
        output = stream.getvalue()
        # The formatter includes %(asctime)s which produces a date-like string
        assert "|" in output
        # Should contain a date pattern (YYYY-MM-DD)
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", output)

    def test_log_level_is_error(self, capture_logger):
        """log_error should log at the ERROR level."""
        log, stream = capture_logger
        exc = ProcessingError("disk full")
        log_error(exc, filepath="videos/lecture.mp4", logger=log)
        output = stream.getvalue()
        assert "ERROR" in output

    def test_logs_without_filepath(self, capture_logger):
        """log_error should work without a filepath argument."""
        log, stream = capture_logger
        exc = ProcessingError("disk full")
        log_error(exc, logger=log)
        output = stream.getvalue()
        assert "disk full" in output
        assert "ERROR" in output

    def test_logs_without_logger(self):
        """log_error should use the module logger when no logger is provided."""
        exc = ProcessingError("disk full")
        # Should not raise an exception
        log_error(exc, filepath="videos/lecture.mp4")

    def test_logs_transcription_error(self, capture_logger):
        """log_error should correctly log TranscriptionError."""
        log, stream = capture_logger
        exc = TranscriptionError("gpu error")
        log_error(exc, filepath="lecture.mp4", logger=log)
        output = stream.getvalue()
        assert "TranscriptionError" in output
        assert "gpu error" in output

    def test_logs_summarization_error(self, capture_logger):
        """log_error should correctly log SummarizationError."""
        log, stream = capture_logger
        exc = SummarizationError("api timeout")
        log_error(exc, filepath="lecture.mp4", logger=log)
        output = stream.getvalue()
        assert "SummarizationError" in output
        assert "api timeout" in output

    def test_logs_validation_error(self, capture_logger):
        """log_error should correctly log ValidationError."""
        log, stream = capture_logger
        exc = ValidationError("bad schema")
        log_error(exc, filepath="doc.pdf", logger=log)
        output = stream.getvalue()
        assert "ValidationError" in output
        assert "bad schema" in output

    def test_consistent_format(self, capture_logger):
        """log_error should produce a consistent format across calls."""
        log, stream = capture_logger
        exc1 = ProcessingError("error one")
        exc2 = ProcessingError("error two")
        log_error(exc1, filepath="file1.mp4", logger=log)
        log_error(exc2, filepath="file2.pdf", logger=log)
        output = stream.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 2
        # Both lines should have the same format structure
        for line in lines:
            assert "|" in line
            assert "ERROR" in line

    def test_logs_stage_when_provided(self, capture_logger):
        """log_error should include the stage in the log output when provided."""
        log, stream = capture_logger
        exc = ProcessingError("transcription failed")
        log_error(exc, filepath="lecture.mp4", logger=log, stage="transcription")
        output = stream.getvalue()
        assert "stage=transcription" in output

    def test_omits_stage_when_not_provided(self, capture_logger):
        """log_error should not include stage info when not provided."""
        log, stream = capture_logger
        exc = ProcessingError("processing failed")
        log_error(exc, filepath="doc.pdf", logger=log)
        output = stream.getvalue()
        assert "stage=" not in output
