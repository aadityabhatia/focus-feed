"""Error handling utilities for local-feed-generator.

Provides custom exception classes for the processing pipeline, a helper to
create error metadata dictionaries matching the JSON schema (design doc
section 3.3), and a consistent error-logging helper that writes to stderr,
stdout (for connection errors), and a dedicated error.log file.

Error handling flow (design doc section 7.1):
1. Catch the exception.
2. Create metadata with the ``error`` field set to the exception message.
3. Write the ``.json`` file to disk.
4. Exclude the entry from feeds (``feed_eligible`` returns ``False``).
5. Log the error to stderr.
6. Write the error to error.log with full details.
7. Print connection errors to stdout with detailed information.
"""

from datetime import UTC, datetime
import logging
from pathlib import Path
import sys
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Module-level loggers
# ---------------------------------------------------------------------------

_logger = logging.getLogger("local_feed_generator")

# Dedicated logger for error.log file handler.  Uses propagate=False so that
# records do not also appear on stderr via the parent logger.
_error_log_logger = logging.getLogger("local_feed_generator.error_log")
_error_log_logger.propagate = False
_error_log_logger.addHandler(logging.NullHandler())

# Module-level error log file handler (set by _configure_error_log).
_error_log_handler: logging.FileHandler | None = None


def _sanitize_error_filename(filename: str) -> str:
    """Sanitize a filename for use as the 'filename' key in error title dicts.

    Removes the file extension and replaces underscores with spaces,
    matching the sanitization used in the Producer's title dict.

    Args:
        filename: The original filename with extension.

    Returns:
        The sanitized filename (no extension, no underscores).
    """
    return Path(filename).stem.replace("_", " ")


def _sanitize_endpoint(endpoint: str | None) -> str | None:
    """Sanitize an endpoint URL to remove sensitive information.

    Removes credentials (username/password) and query parameters from the
    URL to prevent logging API keys or other sensitive data.

    Args:
        endpoint: The endpoint URL to sanitize.

    Returns:
        The sanitized URL, or the original value if it is ``None`` or
        cannot be parsed.
    """
    if not endpoint:
        return endpoint
    try:
        parsed = urlparse(endpoint)
        # If there's no scheme or netloc, the URL is malformed — return as-is.
        if not parsed.scheme or not parsed.netloc:
            return endpoint
        # Reconstruct URL without credentials and query parameters.
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc += f":{parsed.port}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except Exception:
        return endpoint


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


def _configure_error_log(
    data_dir: str | Path | None = None, force: bool = False
) -> logging.FileHandler | None:
    """Configure a FileHandler for the dedicated ``error.log`` file.

    Creates or appends to an ``error.log`` file in the specified *data_dir*
    or the ``DATA_DIR`` from configuration (falling back to the current
    working directory when configuration is unavailable).  The handler
    captures ``ERROR``-level messages with full details including timestamp,
    error type, message, file path, stage, retry info, and endpoint.

    Args:
        data_dir: Directory in which to create ``error.log``.  When
            ``None``, uses ``DATA_DIR`` from configuration, falling back to
            the current working directory.
        force: If ``True``, reconfigure even when a handler is already
            attached (useful in tests).

    Returns:
        The newly created :class:`~logging.FileHandler`, or ``None`` if the
        handler could not be created (e.g. permission error).
    """
    global _error_log_handler

    if _error_log_handler is not None and not force:
        return _error_log_handler

    # Remove existing handler if reconfiguring.
    if _error_log_handler is not None:
        _error_log_logger.removeHandler(_error_log_handler)
        _error_log_handler = None

    # Determine log directory.
    if data_dir is not None:
        log_dir = Path(data_dir).resolve()
    else:
        try:
            from local_feed_generator.config import get_config

            config = get_config()
            log_dir = Path(config.data_dir).resolve()
        except Exception:
            log_dir = Path.cwd()

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        error_log_path = log_dir / "error.log"

        handler = logging.FileHandler(error_log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handler.setLevel(logging.ERROR)

        _error_log_logger.addHandler(handler)
        _error_log_handler = handler
    except (OSError, PermissionError):
        pass

    return _error_log_handler


_configure_logger(_logger)
_configure_error_log()


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
    timestamp = int(datetime.now(UTC).timestamp() * 1000) / 1000.0
    error_message = str(exc) if str(exc) else exc.__class__.__name__

    # Convert string title to dict for backward compatibility.
    title_dict: dict[str, str | None] = {
        "filename": _sanitize_error_filename(filename),
        "extracted": None,
        "generated": None,
    }

    return {
        "title": title_dict,
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
    retry_info: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    print_stdout: bool = False,
) -> None:
    """Log an error with timestamp, file, error type, and optional context.

    Uses a consistent format that includes:
    - **Timestamp** — provided by the logging formatter.
    - **File** — the ``filepath`` argument, if provided.
    - **Error type** — the exception class name.
    - **Error message** — the exception message.
    - **Stage** — the processing stage where the error occurred, if provided
        (e.g. ``"transcription"``, ``"summarization"``, ``"title_generation"``).
    - **Retry info** — retry attempt information, if provided (e.g.
        ``"attempt=1/4"``).
    - **Endpoint** — the LLM endpoint URL, sanitized to remove credentials
        and query parameters.
    - **Model** — the model name used for the request, if provided.

    Output goes to stderr (via the logger) and to ``error.log`` (via the
    dedicated file handler).  When *print_stdout* is ``True``, a detailed
    multi-line summary is also printed to stdout.

    Args:
        exc: The exception to log.
        filepath: Path of the file being processed when the error occurred.
            When ``None``, the file information is omitted from the log line.
        logger: Optional logger instance. Uses the module-level logger if not
            provided.
        stage: Optional processing stage label for additional context about
            where in the pipeline the error occurred.
        retry_info: Optional retry attempt information (e.g.
            ``"attempt=1/4"``).
        endpoint: Optional LLM endpoint URL. Will be sanitized to remove
            credentials and query parameters before logging.
        model: Optional model name used for the request.
        print_stdout: If ``True``, print a detailed multi-line error summary
            to stdout in addition to stderr and error.log.
    """
    log = logger or _logger
    file_info = f" | file={filepath}" if filepath else ""
    stage_info = f" | stage={stage}" if stage else ""
    retry_info_str = f" | {retry_info}" if retry_info else ""
    endpoint_info = f" | endpoint={_sanitize_endpoint(endpoint)}" if endpoint else ""
    model_info = f" | model={model}" if model else ""

    message = (
        f"{exc.__class__.__name__}: {exc}"
        f"{file_info}{stage_info}{retry_info_str}{endpoint_info}{model_info}"
    )

    # Log to stderr (existing behaviour, unchanged).
    log.error(message)

    # Also write to error.log with full details.
    _error_log_logger.error(message)

    # Print to stdout when requested (e.g. for connection errors).
    if print_stdout:
        _print_error_to_stdout(
            exc, filepath, stage, retry_info, endpoint, model
        )


def _print_error_to_stdout(
    exc: Exception,
    filepath: str | None = None,
    stage: str | None = None,
    retry_info: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
) -> None:
    """Print a detailed, human-readable error summary to stdout.

    Produces a multi-line format suitable for direct user consumption,
    including error type, message, retry information, endpoint, and other
    context where available.

    Args:
        exc: The exception to print.
        filepath: Optional file path context.
        stage: Optional processing stage.
        retry_info: Optional retry attempt information.
        endpoint: Optional endpoint URL (will be sanitized).
        model: Optional model name.
    """
    lines = [
        f"Error: {exc}",
        f"Error type: {exc.__class__.__name__}",
    ]
    if retry_info:
        lines.append(f"Retry: {retry_info}")
    if endpoint:
        lines.append(f"Endpoint: {_sanitize_endpoint(endpoint)}")
    if filepath:
        lines.append(f"File: {filepath}")
    if stage:
        lines.append(f"Stage: {stage}")
    if model:
        lines.append(f"Model: {model}")
    print("\n".join(lines))
