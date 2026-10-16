"""Tests for the error handling utilities."""

from io import StringIO
import json
import logging

from local_feed_generator.errors import (
    ProcessingError,
    SummarizationError,
    TranscriptionError,
    ValidationError,
    create_error_metadata,
    log_error,
)
import pytest


class TestExceptionHierarchy:
    """Tests for custom exception class hierarchy and raisability."""

    @pytest.mark.parametrize("exc_class,exc_msg", [
        (ProcessingError, "something went wrong"),
        (TranscriptionError, "audio decode failed"),
        (SummarizationError, "LLM request failed"),
        (ValidationError, "field is required"),
    ])
    def test_exception_can_be_raised(self, exc_class, exc_msg):
        """All exception classes should be raisable and catchable."""
        with pytest.raises(exc_class):
            raise exc_class(exc_msg)

    def test_transcription_error_is_subclass_of_processing_error(self):
        """TranscriptionError should inherit from ProcessingError."""
        assert issubclass(TranscriptionError, ProcessingError)

    def test_summarization_error_is_subclass_of_processing_error(self):
        """SummarizationError should inherit from ProcessingError."""
        assert issubclass(SummarizationError, ProcessingError)

    def test_validation_error_is_not_subclass_of_processing_error(self):
        """ValidationError should be a separate exception hierarchy."""
        assert not issubclass(ValidationError, ProcessingError)

    @pytest.mark.parametrize("exc_class", [TranscriptionError, SummarizationError])
    def test_subclass_caught_as_processing_error(self, exc_class):
        """Subclasses should be catchable via ProcessingError."""
        with pytest.raises(ProcessingError):
            raise exc_class("test error")

    @pytest.mark.parametrize("exc_class,exc_msg,attr_name", [
        (ProcessingError, "disk full", "message"),
        (ProcessingError, None, "filepath"),
        (ValidationError, "invalid value", "field"),
    ])
    def test_attribute_storage(self, exc_class, exc_msg, attr_name):
        """Exceptions should store their specific attributes."""
        kwargs = {}
        if exc_msg is not None:
            kwargs["message"] = exc_msg
        if attr_name == "filepath":
            exc = ProcessingError("disk full", filepath="videos/lecture.mp4")
            assert exc.filepath == "videos/lecture.mp4"
        elif attr_name == "field":
            exc = ValidationError("invalid value", field="email")
            assert exc.field == "email"
        elif attr_name == "message":
            exc = ProcessingError("disk full")
            assert exc.message == "disk full"

    def test_processing_error_filepath_defaults_none(self):
        """ProcessingError filepath should default to None."""
        exc = ProcessingError("disk full")
        assert exc.filepath is None

    def test_validation_error_field_defaults_none(self):
        """ValidationError field should default to None."""
        exc = ValidationError("invalid value")
        assert exc.field is None


class TestCreateErrorMetadata:
    """Tests for the create_error_metadata utility function."""

    def test_error_metadata_has_all_fields(self):
        """Error metadata should have all required fields with correct values."""
        exc = ProcessingError("disk full")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == "disk full"
        assert metadata["filename"] == "lecture.mp4"
        assert metadata["filepath"] == "videos/lecture.mp4"
        assert metadata["filetype"] == "mp4"
        assert metadata["title"] == {"filename": "lecture", "extracted": None, "generated": None}
        assert "date" in metadata
        assert metadata["date"] is not None
        assert "processing_date" not in metadata
        expected_fields = {"title", "description", "filename", "filepath", "filetype", "date", "size", "error"}
        assert expected_fields.issubset(set(metadata.keys()))
        assert "status" not in metadata

    @pytest.mark.parametrize("exc_class,exc_msg", [
        (ProcessingError, "failed"),
        (TranscriptionError, "gpu error"),
        (SummarizationError, "api timeout"),
        (ValidationError, "bad schema"),
        (RuntimeError, "unexpected failure"),
    ])
    def test_works_with_various_exception_types(self, exc_class, exc_msg):
        """create_error_metadata should work with various exception types."""
        exc = exc_class(exc_msg)
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4", "mp4")
        assert metadata["error"] == exc_msg

    def test_filetype_defaults_to_unknown(self):
        """Error metadata filetype should default to 'unknown' when not provided."""
        exc = ProcessingError("failed")
        metadata = create_error_metadata(exc, "lecture.mp4", "videos/lecture.mp4")
        assert metadata["filetype"] == "unknown"

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

    @pytest.mark.parametrize("exc,exc_name,exc_msg", [
        (ProcessingError("disk full"), "ProcessingError", "disk full"),
        (TranscriptionError("gpu error"), "TranscriptionError", "gpu error"),
        (SummarizationError("api timeout"), "SummarizationError", "api timeout"),
        (ValidationError("bad schema"), "ValidationError", "bad schema"),
    ])
    def test_logs_exception_details(self, capture_logger, exc, exc_name, exc_msg):
        """log_error should include both the exception class name and message."""
        log, stream = capture_logger
        log_error(exc, filepath="lecture.mp4", logger=log)
        output = stream.getvalue()
        assert exc_name in output
        assert exc_msg in output

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

    @pytest.mark.parametrize("stage,expected_in_output", [
        ("transcription", "stage=transcription"),
        (None, None),
    ])
    def test_stage_logging(self, capture_logger, stage, expected_in_output):
        """log_error should include stage when provided, omit when not."""
        log, stream = capture_logger
        exc = ProcessingError("processing failed")
        if stage:
            log_error(exc, filepath="lecture.mp4", logger=log, stage=stage)
        else:
            log_error(exc, filepath="doc.pdf", logger=log)
        output = stream.getvalue()
        if expected_in_output:
            assert expected_in_output in output
        else:
            assert "stage=" not in output
