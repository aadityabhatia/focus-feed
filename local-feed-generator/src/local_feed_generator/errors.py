"""Error handling utilities for local-feed-generator.

Provides custom exception classes for the processing pipeline, a helper to
create error metadata dictionaries matching the JSON schema (design doc
section 3.3), and a consistent error-logging helper that writes to stderr.

Error handling flow (design doc section 7.1):
1. Catch the exception.
2. Create metadata with the ``error`` field set to the exception message.
3. Write the ``.json`` file to disk.
4. Exclude the entry from feeds (``feed_eligible`` returns ``False``).
5. Log the error to stderr.
"""

from datetime import UTC, datetime
import logging
import sys

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger("local_feed_generator")


def _configure_logger(logger: logging.Logger) -> None:
    """Attach a stderr handler with a consistent format if none exists.

    The format includes a timestamp, log level, and message, satisfying the
    requirement that error logging includes timestamp, file, and error message.
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


_configure_logger(_logger)


# ---------------------------------------------------------------------------
# Custom exception classes
# ---------------------------------------------------------------------------


class ProcessingError(Exception):
    """Base exception for errors during file processing.

    All processing-related exceptions inherit from this class so that
    callers can catch a single base type when they want to handle any
    processing failure.

    Attributes:
        message: Human-readable error description.
        filepath: Path of the file being processed when the error occurred,
            or ``None`` if not applicable.
    """

    def __init__(self, message: str, filepath: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.filepath = filepath


class TranscriptionError(ProcessingError):
    """Exception raised when audio transcription fails.

    Covers corrupted audio files, GPU/CUDA errors, and model loading failures.
    """


class SummarizationError(ProcessingError):
    """Exception raised when LLM summarization fails.

    Covers API connection errors, rate limiting, and response parsing failures.
    """


class ValidationError(Exception):
    """Exception raised when data validation fails.

    Attributes:
        message: Human-readable error description.
        field: Name of the field that failed validation, or ``None`` if not
            specific to a single field.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field


# ---------------------------------------------------------------------------
# Error metadata creation
# ---------------------------------------------------------------------------


def create_error_metadata(
    exc: Exception,
    filename: str,
    filepath: str,
    filetype: str | None = None,
) -> dict:
    """Create an error metadata dictionary from an exception.

    Produces a dictionary matching the JSON metadata schema (design doc
    section 3.3) with the ``error`` field set to the exception message.
    Step maps are set to ``None`` so that :attr:`feed_eligible`
    returns ``False`` and the entry is excluded from feeds.  The returned
    dict is ready for JSON serialization and writing to a ``.json`` file
    adjacent to the source file.

    Date field (``date``) is stored as a UNIX epoch integer (seconds since
    1970-01-01 UTC).

    Args:
        exc: The exception that occurred during processing.
        filename: Name of the source file being processed (e.g. ``"lecture.mp4"``).
        filepath: Relative path of the file from the data directory root
            (e.g. ``"videos/lecture.mp4"``).
        filetype: File type (``"mp4"``, ``"pdf"``, ``"html"``), if known.
            Defaults to ``"unknown"`` when not provided.

    Returns:
        A dictionary suitable for JSON serialization as an error metadata file.
    """
    timestamp = int(datetime.now(UTC).timestamp())
    error_message = str(exc) if str(exc) else exc.__class__.__name__

    return {
        "title": filename,
        "description": "",
        "filename": filename,
        "filepath": filepath,
        "filetype": filetype or "unknown",
        "date": timestamp,
        "size": 0,
        "error": error_message,
    }


# ---------------------------------------------------------------------------
# Error logging helper
# ---------------------------------------------------------------------------


def log_error(
    exc: Exception,
    filepath: str | None = None,
    logger: logging.Logger | None = None,
    stage: str | None = None,
) -> None:
    """Log an error with timestamp, file, error type, and optional stage.

    Uses a consistent format that includes:
    - **Timestamp** — provided by the logging formatter.
    - **File** — the ``filepath`` argument, if provided.
    - **Error type** — the exception class name.
    - **Error message** — the exception message.
    - **Stage** — the processing stage where the error occurred, if provided
        (e.g. ``"transcription"``, ``"summarization"``, ``"metadata"``).

    Output goes to stderr, matching design doc section 7.1.

    Args:
        exc: The exception to log.
        filepath: Path of the file being processed when the error occurred.
            When ``None``, the file information is omitted from the log line.
        logger: Optional logger instance. Uses the module-level logger if not
            provided.
        stage: Optional processing stage label for additional context about
            where in the pipeline the error occurred.
    """
    log = logger or _logger
    file_info = f" | file={filepath}" if filepath else ""
    stage_info = f" | stage={stage}" if stage else ""
    log.error(
        "%s: %s%s%s",
        exc.__class__.__name__,
        exc,
        file_info,
        stage_info,
    )
