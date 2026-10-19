"""Tests for the error handling utilities."""

from io import StringIO
import json
import logging

from local_feed_generator.errors import (
    ProcessingError,
    SummarizationError,
    TranscriptionError,
    ValidationError,
    _configure_error_log,
    _print_error_to_stdout,
    _sanitize_endpoint,
    create_error_metadata,
    log_error,
)
import pytest


class TestExceptionHierarchy:
    """Tests for custom exception class hierarchy and raisability."""

    @pytest.mark.parametrize(
        "exc_class,exc_msg",
        [
            (ProcessingError, "something went wrong"),
            (TranscriptionError, "audio decode failed"),
            (SummarizationError, "LLM request failed"),
            (ValidationError, "field is required"),
        ],
    )
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

    @pytest.mark.parametrize(
        "exc_class,exc_msg,attr_name",
        [
            (ProcessingError, "disk full", "message"),
            (ProcessingError, None, "filepath"),
            (ValidationError, "invalid value", "field"),
        ],
    )
    def test_attribute_storage(self, exc_class, exc_msg, attr_name):
        """Exceptions should store their specific attributes."""
        kwargs = {}
        if exc_msg is not None:
            kwargs["message"] = exc_msg
        exc: Exception
        if attr_name == "filepath":
            exc = ProcessingError("disk full", filepath="videos/lecture.mp4")
            assert isinstance(exc, ProcessingError)
            assert exc.filepath == "videos/lecture.mp4"
        elif attr_name == "field":
            exc = ValidationError("invalid value", field="email")
            assert isinstance(exc, ValidationError)
            assert exc.field == "email"
        elif attr_name == "message":
            exc = ProcessingError("disk full")
            assert isinstance(exc, ProcessingError)
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
        expected_fields = {
            "title",
            "description",
            "filename",
            "filepath",
            "filetype",
            "date",
            "size",
            "error",
        }
        assert expected_fields.issubset(set(metadata.keys()))
        assert "status" not in metadata

    @pytest.mark.parametrize(
        "exc_class,exc_msg",
        [
            (ProcessingError, "failed"),
            (TranscriptionError, "gpu error"),
            (SummarizationError, "api timeout"),
            (ValidationError, "bad schema"),
            (RuntimeError, "unexpected failure"),
        ],
    )
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

    @pytest.mark.parametrize(
        "exc,exc_name,exc_msg",
        [
            (ProcessingError("disk full"), "ProcessingError", "disk full"),
            (TranscriptionError("gpu error"), "TranscriptionError", "gpu error"),
            (SummarizationError("api timeout"), "SummarizationError", "api timeout"),
            (ValidationError("bad schema"), "ValidationError", "bad schema"),
        ],
    )
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

    @pytest.mark.parametrize(
        "stage,expected_in_output",
        [
            ("transcription", "stage=transcription"),
            (None, None),
        ],
    )
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


# ---------------------------------------------------------------------------
# Tests for _sanitize_endpoint
# ---------------------------------------------------------------------------


class TestSanitizeEndpoint:
    """Tests for the _sanitize_endpoint helper function."""

    def test_plain_url_unchanged(self):
        """A plain URL without credentials or query params should be unchanged."""
        result = _sanitize_endpoint("https://llm.example.com/v1")
        assert result == "https://llm.example.com/v1"

    def test_removes_query_params(self):
        """Query parameters (e.g. API keys) should be removed."""
        result = _sanitize_endpoint("https://llm.example.com/v1?api_key=secret123")
        assert result == "https://llm.example.com/v1"

    def test_removes_credentials(self):
        """Username/password in the URL should be removed."""
        result = _sanitize_endpoint("https://user:pass@llm.example.com/v1")
        assert result == "https://llm.example.com/v1"

    def test_removes_credentials_and_query(self):
        """Both credentials and query parameters should be removed."""
        result = _sanitize_endpoint("https://key@host.com/v1?token=abc")
        assert result == "https://host.com/v1"

    def test_preserves_port(self):
        """Port number should be preserved."""
        result = _sanitize_endpoint("https://llm.example.com:8080/v1")
        assert result == "https://llm.example.com:8080/v1"

    def test_none_returns_none(self):
        """None input should return None."""
        assert _sanitize_endpoint(None) is None

    def test_empty_string_returns_empty(self):
        """Empty string should return empty string."""
        assert _sanitize_endpoint("") == ""

    def test_malformed_url_returns_original(self):
        """A malformed URL should return the original value."""
        result = _sanitize_endpoint("not-a-url")
        assert result == "not-a-url"


# ---------------------------------------------------------------------------
# Tests for _configure_error_log
# ---------------------------------------------------------------------------


class TestConfigureErrorLog:
    """Tests for the _configure_error_log function."""

    def test_creates_error_log_file(self, tmp_path):
        """_configure_error_log should create an error.log file."""
        handler = _configure_error_log(data_dir=tmp_path, force=True)
        assert handler is not None
        assert (tmp_path / "error.log").exists()

    def test_returns_existing_handler_when_not_forced(self, tmp_path):
        """Without force=True, should return the existing handler."""
        handler1 = _configure_error_log(data_dir=tmp_path, force=True)
        handler2 = _configure_error_log(data_dir=tmp_path, force=False)
        assert handler1 is handler2

    def test_force_reconfigures(self, tmp_path):
        """With force=True, should create a new handler."""
        handler1 = _configure_error_log(data_dir=tmp_path, force=True)
        handler2 = _configure_error_log(data_dir=tmp_path, force=True)
        assert handler1 is not handler2

    def test_handler_logs_at_error_level(self, tmp_path):
        """The file handler should be configured at ERROR level."""
        handler = _configure_error_log(data_dir=tmp_path, force=True)
        assert handler is not None
        assert handler.level == logging.ERROR

    def test_handler_has_correct_formatter(self, tmp_path):
        """The file handler should use the expected format."""
        handler = _configure_error_log(data_dir=tmp_path, force=True)
        assert handler is not None
        formatter = handler.formatter
        assert formatter is not None
        assert formatter._fmt is not None
        # The format should include asctime, levelname, and message
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_log_error_writes_to_error_log(self, tmp_path):
        """log_error should write to the error.log file."""
        _configure_error_log(data_dir=tmp_path, force=True)
        exc = ProcessingError("test error for error.log")
        log_error(exc, filepath="test.txt", stage="test_stage")

        error_log_path = tmp_path / "error.log"
        assert error_log_path.exists()
        content = error_log_path.read_text(encoding="utf-8")
        assert "test error for error.log" in content
        assert "ProcessingError" in content
        assert "stage=test_stage" in content
        assert "ERROR" in content

    def test_log_error_writes_all_fields_to_error_log(self, tmp_path):
        """log_error should include all detail fields in error.log."""
        _configure_error_log(data_dir=tmp_path, force=True)
        exc = ProcessingError("detailed error")
        log_error(
            exc,
            filepath="videos/lecture.mp4",
            stage="summarization",
            retry_info="attempt=1/4",
            endpoint="https://llm.example.com/v1",
            model="test-model",
        )

        content = (tmp_path / "error.log").read_text(encoding="utf-8")
        assert "ProcessingError" in content
        assert "detailed error" in content
        assert "file=videos/lecture.mp4" in content
        assert "stage=summarization" in content
        assert "attempt=1/4" in content
        assert "endpoint=https://llm.example.com/v1" in content
        assert "model=test-model" in content

    def test_log_error_sanitizes_endpoint_in_error_log(self, tmp_path):
        """Endpoint should be sanitized in error.log."""
        _configure_error_log(data_dir=tmp_path, force=True)
        exc = ProcessingError("auth error")
        log_error(
            exc,
            endpoint="https://key:secret@llm.example.com/v1?api_key=hidden",
        )

        content = (tmp_path / "error.log").read_text(encoding="utf-8")
        assert "secret" not in content
        assert "api_key=hidden" not in content
        assert "key:" not in content  # credentials removed
        assert "https://llm.example.com/v1" in content

    def test_uses_cwd_when_data_dir_not_provided(self, tmp_path, monkeypatch):
        """_configure_error_log should use the CWD when data_dir is not provided.

        The current working directory is used to determine where to create
        ``error.log`` when no *data_dir* is given.
        """
        # Change to tmp_path so CWD is a known, writable location.
        monkeypatch.chdir(tmp_path)

        handler = _configure_error_log(force=True)
        assert handler is not None

        # error.log should be written to the current working directory (tmp_path)
        assert (tmp_path / "error.log").exists()

    def test_returns_none_on_permission_error(self, monkeypatch, tmp_path):
        """_configure_error_log should return None on PermissionError during mkdir."""
        import local_feed_generator.errors as errors_module

        # Mock Path.mkdir to raise PermissionError, while keeping Path.cwd()
        # and other Path methods functional.
        original_path = errors_module.Path

        def mock_mkdir(self, *args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(original_path, "mkdir", mock_mkdir)

        handler = _configure_error_log(force=True)
        assert handler is None

    def test_logs_critical_on_permission_error(self, monkeypatch, tmp_path, caplog):
        """_configure_error_log should log CRITICAL when error.log cannot be created."""
        import local_feed_generator.errors as errors_module

        original_path = errors_module.Path

        def mock_mkdir(self, *args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(original_path, "mkdir", mock_mkdir)

        with caplog.at_level(logging.CRITICAL, logger="local_feed_generator"):
            handler = _configure_error_log(force=True)
        assert handler is None
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) == 1
        assert "Failed to create error.log" in critical_records[0].message


# ---------------------------------------------------------------------------
# Tests for log_error stdout printing
# ---------------------------------------------------------------------------


class TestLogErrorStdout:
    """Tests for stdout printing in log_error."""

    @pytest.fixture
    def capture_logger(self):
        """Create a logger with a StringIO handler to capture stderr output."""
        log = logging.getLogger("test_log_error_stdout")
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

    def test_print_stdout_outputs_to_stdout(self, capture_logger, capsys):
        """log_error with print_stdout=True should print to stdout."""
        log, _ = capture_logger
        exc = ProcessingError("connection failed")
        log_error(
            exc,
            filepath="test.mp4",
            stage="summarization",
            retry_info="attempt=1/4",
            endpoint="https://llm.example.com/v1",
            model="test-model",
            logger=log,
            print_stdout=True,
        )
        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "ProcessingError" in captured.out
        assert "connection failed" in captured.out
        assert "attempt=1/4" in captured.out
        assert "https://llm.example.com/v1" in captured.out
        assert "test-model" in captured.out
        assert "test.mp4" in captured.out
        assert "summarization" in captured.out

    def test_print_stdout_false_no_stdout(self, capture_logger, capsys):
        """log_error with print_stdout=False should not print to stdout."""
        log, _ = capture_logger
        exc = ProcessingError("some error")
        log_error(exc, logger=log, print_stdout=False)
        captured = capsys.readouterr()
        assert "Error:" not in captured.out
        assert "Error type" not in captured.out

    def test_print_stdout_includes_error_type(self, capture_logger, capsys):
        """stdout output should include the error type (class name)."""
        log, _ = capture_logger
        exc = SummarizationError("api failure")
        log_error(exc, logger=log, print_stdout=True)
        captured = capsys.readouterr()
        assert "SummarizationError" in captured.out

    def test_print_stdout_includes_endpoint(self, capture_logger, capsys):
        """stdout output should include the sanitized endpoint."""
        log, _ = capture_logger
        exc = ProcessingError("connection refused")
        log_error(
            exc,
            endpoint="https://key:secret@llm.example.com/v1?token=abc",
            logger=log,
            print_stdout=True,
        )
        captured = capsys.readouterr()
        assert "https://llm.example.com/v1" in captured.out
        assert "secret" not in captured.out
        assert "token=abc" not in captured.out

    def test_print_stdout_includes_retry_info(self, capture_logger, capsys):
        """stdout output should include retry attempt information."""
        log, _ = capture_logger
        exc = ProcessingError("timeout")
        log_error(exc, retry_info="attempt=3/4", logger=log, print_stdout=True)
        captured = capsys.readouterr()
        assert "attempt=3/4" in captured.out

    def test_print_stdout_without_optional_fields(self, capture_logger, capsys):
        """stdout output should work without optional fields."""
        log, _ = capture_logger
        exc = ProcessingError("simple error")
        log_error(exc, logger=log, print_stdout=True)
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Error type: ProcessingError" in captured.out
        assert "simple error" in captured.out


# ---------------------------------------------------------------------------
# Tests for _print_error_to_stdout
# ---------------------------------------------------------------------------


class TestPrintErrorToStdout:
    """Tests for the _print_error_to_stdout helper function."""

    def test_prints_error_type_and_message(self, capsys):
        """Should print error type and message."""
        exc = ProcessingError("something broke")
        _print_error_to_stdout(exc)
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "ProcessingError" in captured.out
        assert "something broke" in captured.out

    def test_includes_all_fields(self, capsys):
        """Should include all provided fields in stdout output."""
        exc = ProcessingError("fail")
        _print_error_to_stdout(
            exc,
            filepath="videos/lecture.mp4",
            stage="summarization",
            retry_info="attempt=2/4",
            endpoint="https://llm.example.com/v1",
            model="mistral",
        )
        captured = capsys.readouterr()
        assert "videos/lecture.mp4" in captured.out
        assert "summarization" in captured.out
        assert "attempt=2/4" in captured.out
        assert "https://llm.example.com/v1" in captured.out
        assert "mistral" in captured.out

    def test_omits_none_fields(self, capsys):
        """Should omit fields that are None."""
        exc = ProcessingError("fail")
        _print_error_to_stdout(exc)
        captured = capsys.readouterr()
        assert "File:" not in captured.out
        assert "Stage:" not in captured.out
        assert "Retry" not in captured.out
        assert "Endpoint:" not in captured.out
        assert "Model:" not in captured.out
