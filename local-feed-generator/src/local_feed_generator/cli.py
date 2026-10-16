"""CLI for local-feed-generator.

Provides a Click-based command-line interface with granular subcommands for
each step of the processing pipeline:

- ``process``          — Batch process all files in the data directory.
- ``watch``            — Watch the data directory for changes (incremental mode).
- ``feeds``            — Generate RSS feeds from existing JSON metadata.
- ``validate``         — Validate all JSON metadata files.
- ``transcribe``       — Transcribe MP4 files to markdown text.
- ``summarize``        — Generate summaries from existing transcripts.
- ``extract``          — Extract metadata from PDF/HTML files.
- ``title``            — Generate titles for files that lack proper titles.

Shared options (``--data-dir``, ``--config-file``, ``--file``, ``--force``,
and ``--dry-run``) are available on every subcommand via decorators so that
the CLI usage matches the design-doc examples, e.g.::

    uv run local-feed-generator process --data-dir ./data
    uv run local-feed-generator transcribe --data-dir ./data --file videos/lecture.mp4 --force
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from typing import Any

import click
from watchdog.events import FileSystemEventHandler

from local_feed_generator.config import reload_config
from local_feed_generator.errors import log_error
from local_feed_generator.metadata import Metadata
from local_feed_generator.scanner import ScannedFile, Scanner
from local_feed_generator.services.extractors import _clean_title

_logger = logging.getLogger("local_feed_generator")

# Source file extensions supported for processing (case-insensitive).
SOURCE_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".pdf", ".html"})

# Characters that are problematic in filenames for XML/URL contexts.
# These trigger a warning but do not block processing.
_SPECIAL_CHAR_PATTERN = re.compile(r'[<>&"\'\x00-\x1f]')

# Filenames that should be ignored during scanning (generated feeds, index pages).
# Comparison is case-insensitive (see callers that use ``filename.lower()``).
_IGNORE_FILENAMES = frozenset(
    {"podcast.xml", "articles.xml", "podcast.atom", "articles.atom", "index.html", "feed.json"}
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_config(config_file: str | None, data_dir: str | None) -> None:
    """Load configuration and reload the config singleton.

    If *data_dir* is provided it is set as the ``DATA_DIR`` environment
    variable so that the config singleton picks it up.  If *config_file*
    is provided its values are loaded into the environment (without
    overriding values already set in the shell).

    Args:
        config_file: Path to a ``.env``-style configuration file, or ``None``.
        data_dir: Data directory path from the CLI, or ``None``.

    Raises:
        click.ClickException: If the configuration cannot be loaded.
    """
    if data_dir:
        os.environ["DATA_DIR"] = data_dir
    if config_file:
        try:
            from dotenv import load_dotenv

            load_dotenv(config_file, override=False)
        except ImportError:
            _logger.warning(
                "python-dotenv not available; cannot load config file %s",
                config_file,
            )
    try:
        reload_config()
    except Exception as exc:
        raise click.ClickException(f"Configuration error: {exc}") from exc


def _resolve_data_dir(data_dir: str | None, file_path: str | None) -> Path:
    """Resolve the data directory from CLI args or config.

    When *file_path* is an absolute path, the data directory is derived
    from the file's parent directory.  When *file_path* is relative and
    *data_dir* is provided, *data_dir* is used as the base for resolving
    the file.  Otherwise *data_dir* (or the config default) is used.

    Args:
        data_dir: The ``--data-dir`` CLI value, or ``None``.
        file_path: The ``--file`` CLI value, or ``None``.

    Returns:
        An absolute :class:`~pathlib.Path` to the data directory.
    """
    if file_path is not None:
        file_p = Path(file_path)
        if file_p.is_absolute():
            return file_p.parent
    if data_dir is not None:
        return Path(data_dir).resolve()
    from local_feed_generator.config import get_config

    return Path(get_config().data_dir).resolve()


def _find_json_files(data_dir: Path) -> list[Path]:
    """Find all JSON metadata files in *data_dir*.

    Returns a sorted list of paths to ``.json`` files, excluding generated
    feed files (``podcast.xml``, ``articles.xml``, etc.).
    """
    json_files: list[Path] = []
    for path in data_dir.rglob("*.json"):
        if path.name.lower() in _IGNORE_FILENAMES:
            continue
        json_files.append(path)
    return sorted(json_files)


def _check_special_characters(filepath: Path) -> list[str]:
    """Check if *filepath*'s filename contains special characters.

    Returns a list of human-readable warning strings for any issues found.
    """
    filename = filepath.name
    warnings: list[str] = []
    if filename != filename.strip():
        warnings.append(f"Filename '{filename}' has leading/trailing whitespace: {filepath}")
    matches = _SPECIAL_CHAR_PATTERN.findall(filename)
    if matches:
        unique_chars = sorted(set(matches))
        warnings.append(
            f"Filename '{filename}' contains special characters {unique_chars}: {filepath}"
        )
    return warnings


def _is_orphaned_json(json_path: Path) -> bool:
    """Return ``True`` if *json_path* is orphaned.

    A JSON file is orphaned when the corresponding source file (the JSON
    name without the ``.json`` suffix) does not exist or has an unsupported
    extension.
    """
    source_filename = json_path.name[: -len(".json")]
    source_path = json_path.parent / source_filename
    if not source_path.exists():
        return True
    return source_path.suffix.lower() not in SOURCE_EXTENSIONS


def _generate_feeds(data_dir_path: Path) -> dict[str, int]:
    """Generate both feeds and return item counts.

    Creates a :class:`~local_feed_generator.feeds.FeedGenerator`, calls
    ``generate_all_feeds()``, then reads back the JSON metadata files to
    produce a summary of how many items ended up in each feed.

    Returns:
        A dict with ``"podcast"`` and ``"articles"`` keys.
    """
    from local_feed_generator.feeds import FeedGenerator

    generator = FeedGenerator(data_dir=str(data_dir_path))
    generator.generate_all_feeds()

    # Count items for the summary.
    podcast_count = 0
    article_count = 0
    for json_path in _find_json_files(data_dir_path):
        try:
            metadata = Metadata.from_json_file(json_path)
            if metadata.feed_eligible:
                if metadata.filetype == "mp4":
                    podcast_count += 1
                elif metadata.filetype in ("pdf", "html"):
                    article_count += 1
        except Exception:
            pass

    return {"podcast": podcast_count, "articles": article_count}


def _print_feed_summary(counts: dict[str, int]) -> None:
    """Print a feed summary to stdout."""
    click.echo(
        f"Feeds generated: {counts['podcast']} podcast item(s), "
        f"{counts['articles']} article item(s)."
    )


# ---------------------------------------------------------------------------
# Granular pipeline helpers
# ---------------------------------------------------------------------------


def _find_source_files(data_dir: Path, filetypes: tuple[str, ...]) -> list[Path]:
    """Find all source files of the given types in *data_dir*.

    Recursively walks *data_dir* and returns a sorted list of paths to files
    whose extension (case-insensitive, without the leading dot) matches one of
    *filetypes*.  Generated feed files (``podcast.xml``, etc.) and ``.json``
    metadata files are excluded.
    """
    results: list[Path] = []
    for root, _dirs, files in os.walk(data_dir):
        root_path = Path(root)
        for filename in files:
            if filename.lower() in _IGNORE_FILENAMES:
                continue
            filepath = root_path / filename
            filetype = filepath.suffix.lower().lstrip(".")
            if filetype in filetypes:
                results.append(filepath)
    return sorted(results)


def _get_json_metadata(json_path: Path) -> dict[str, Any] | None:
    """Read and parse a JSON metadata file.

    Returns ``None`` if the file does not exist or cannot be parsed.
    """
    if not json_path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError, OSError, TypeError:
        return None


def _write_json_metadata(data: dict[str, Any], source_path: Path) -> None:
    """Write a metadata dictionary atomically next to *source_path*.

    The JSON file is named ``<source_filename>.json`` and placed in the same
    directory as the source file.  Writes go to a temporary file first, then
    are renamed to the final path to avoid corruption from partial writes.
    """
    json_path = source_path.with_name(source_path.name + ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(json_path)


def _find_files_needing_transcription(data_dir: Path, force: bool = False) -> list[Path]:
    """Find MP4 files that need transcription.

    A file needs transcription when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the ``transcribe`` step map's
      ``status`` is not ``"complete"``.

    When *force* is ``True``, all MP4 files are returned regardless of
    their current transcription status.
    """
    mp4_files = _find_source_files(data_dir, ("mp4",))
    needs: list[Path] = []
    for mp4_file in mp4_files:
        json_path = mp4_file.with_name(mp4_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if force or metadata is None or not _is_step_complete(metadata, "transcribe"):
            needs.append(mp4_file)
    return needs


def _find_files_needing_summary(data_dir: Path, force: bool = False) -> list[Path]:
    """Find MP4, PDF, and HTML files that have content but need a summary.

    A file needs summarization when:
    - The JSON metadata file exists, **and**
    - It has a non-empty ``content`` field (transcript for MP4, extracted
      text for PDF/HTML), **and**
    - The ``summarize`` step map's ``status`` is not ``"complete"``.

    When *force* is ``True``, all files with content are returned regardless
    of their current summarization status.
    """
    source_files = _find_source_files(data_dir, ("mp4", "pdf", "html"))
    needs: list[Path] = []
    for source_file in source_files:
        json_path = source_file.with_name(source_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if metadata is None:
            continue
        content = metadata.get("content")
        if not content:
            continue
        if force or not _is_step_complete(metadata, "summarize"):
            needs.append(source_file)
    return needs


def _find_files_needing_metadata_extraction(data_dir: Path, force: bool = False) -> list[Path]:
    """Find PDF/HTML files that need metadata extraction.

    A file needs metadata extraction when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the ``extract`` step map's
      ``status`` is not ``"complete"``, or it is missing a ``title`` field.

    When *force* is ``True``, all PDF/HTML files are returned regardless
    of their current extraction status.
    """
    doc_files = _find_source_files(data_dir, ("pdf", "html"))
    needs: list[Path] = []
    for doc_file in doc_files:
        json_path = doc_file.with_name(doc_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if (
            force
            or metadata is None
            or not _is_step_complete(metadata, "extract")
            or not metadata.get("title")
        ):
            needs.append(doc_file)
    return needs


def _find_files_needing_json_generation(data_dir: Path, force: bool = False) -> list[Path]:
    """Find all source files that need JSON metadata generation.

    A file needs JSON generation when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the file is not feed-eligible
      (i.e. not all applicable step maps have ``status == "complete"``).

    When *force* is ``True``, all source files are returned regardless
    of their current step statuses.
    """
    source_files = _find_source_files(data_dir, ("mp4", "pdf", "html"))
    needs: list[Path] = []
    for source_file in source_files:
        json_path = source_file.with_name(source_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if (
            force
            or metadata is None
            or not _is_dict_feed_eligible(metadata, source_file.suffix.lower().lstrip("."))
        ):
            needs.append(source_file)
    return needs


def _find_files_needing_title(data_dir: Path, force: bool = False) -> list[Path]:
    """Find source files that need title generation.

    A file needs a title when:
    - A JSON metadata file exists for it, **and**
    - The source file exists, **and**
    - The source file has a supported extension (``.mp4``, ``.pdf``, ``.html``), **and**
    - The title does not have a ``generated`` field, **or** *force* is ``True``.

    When *force* is ``True``, all source files with JSON metadata are returned
    regardless of their current title.

    Args:
        data_dir: Absolute path to the data directory.
        force: When ``True``, include files that already have a generated title.

    Returns:
        A sorted list of absolute paths to source files needing title
        generation.
    """
    needs: list[Path] = []
    for json_file in data_dir.rglob("*.json"):
        if json_file.name.lower() in _IGNORE_FILENAMES:
            continue
        try:
            metadata_dict = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            continue
        source_file = json_file.with_name(json_file.stem)
        if not source_file.exists():
            continue
        if source_file.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        title = metadata_dict.get("title", "")
        # Normalize string title to dict for backward compatibility.
        if isinstance(title, str):
            title = {"filename": title}
        # A file needs a title if 'generated' is not set or empty.
        has_generated = isinstance(title, dict) and bool(title.get("generated", "").strip())
        if force or not has_generated:
            needs.append(source_file)
    return needs


def _get_content_for_title(file_path: Path, metadata_dict: dict[str, Any]) -> str:
    """Get content for title generation based on file type.

    For MP4 files, uses the content (transcript) from the metadata dict.
    For PDF/HTML files, reads the first 8000 characters of the file content.

    Args:
        file_path: Path to the source file.
        metadata_dict: The parsed JSON metadata dictionary.

    Returns:
        Content text suitable for title generation, or an empty string if
        the file type is unsupported or content cannot be retrieved.
    """
    ext = file_path.suffix.lower()
    if ext == ".mp4":
        return str(metadata_dict.get("content", ""))
    if ext in (".pdf", ".html"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return content[:8000]
        except OSError:
            return ""
    return ""


def _is_dict_feed_eligible(metadata: dict[str, Any], filetype: str) -> bool:
    """Check if a raw metadata dict indicates the file is feed-eligible.

    A file is feed-eligible when all applicable step maps have
    ``status == "complete"`` and there is no error field.

    - **MP4**: ``transcribe`` step map exists with ``status == "complete"``
      and ``summarize`` step map exists with ``status == "complete"``.
    - **PDF/HTML**: ``extract`` step map exists with ``status == "complete"``.
      If ``content`` exists, the ``summarize`` step map must also exist with
      ``status == "complete"``.
    """
    if metadata.get("error"):
        return False
    if filetype == "mp4":
        transcribe = metadata.get("transcribe")
        summarize = metadata.get("summarize")
        return (
            isinstance(transcribe, dict)
            and transcribe.get("status") == "complete"
            and isinstance(summarize, dict)
            and summarize.get("status") == "complete"
        )
    if filetype in ("pdf", "html"):
        extract = metadata.get("extract")
        if not (isinstance(extract, dict) and extract.get("status") == "complete"):
            return False
        # If content exists, summarize step must also be complete.
        content = metadata.get("content")
        if content:
            summarize = metadata.get("summarize")
            if not (isinstance(summarize, dict) and summarize.get("status") == "complete"):
                return False
        return True
    return False


def _is_step_complete(metadata: dict[str, Any], step: str) -> bool:
    """Check if a specific step is complete in a raw metadata dict.

    Checks the step map (``transcribe``, ``summarize``, or ``extract``)
    for the given step name.

    Args:
        metadata: The parsed JSON metadata dictionary.
        step: The step name ("transcribe", "summarize", or "extract").

    Returns:
        ``True`` if the step map exists and its ``status`` is ``"complete"``,
        ``False`` otherwise.
    """
    step_map = metadata.get(step)
    return isinstance(step_map, dict) and step_map.get("status") == "complete"


def _resolve_file_path(data_dir: Path, file_arg: str | None) -> Path | None:
    """Resolve a ``--file`` argument to an absolute path.

    If *file_arg* is an absolute path, it is used directly.  Otherwise it
    is resolved relative to *data_dir*.
    """
    if file_arg is None:
        return None
    file_path = Path(file_arg)
    if file_path.is_absolute():
        return file_path
    return data_dir / file_path


# ---------------------------------------------------------------------------
# Shared option decorators
# ---------------------------------------------------------------------------


def _data_dir_option(func: Any) -> Any:
    """Add the ``--data-dir`` required option to a Click command."""
    return click.option(
        "--data-dir",
        required=True,
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        help="Path to the data directory.",
    )(func)


def _data_dir_option_optional(func: Any) -> Any:
    """Add the ``--data-dir`` optional option to a Click command.

    When omitted, the data directory is read from the application
    configuration (``DATA_DIR`` environment variable).
    """
    return click.option(
        "--data-dir",
        required=False,
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=None,
        help="Path to the data directory. Defaults to DATA_DIR from configuration.",
    )(func)


def _config_file_option(func: Any) -> Any:
    """Add the ``--config-file`` optional option to a Click command."""
    return click.option(
        "--config-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="Path to a .env-style configuration file.",
    )(func)


def _file_option(func: Any) -> Any:
    """Add the ``--file`` optional flag to a Click command.

    When specified, only the given file is processed.  The path may be
    relative to the data directory or absolute.
    """
    return click.option(
        "--file",
        "file_path",
        required=False,
        default=None,
        type=click.Path(),
        help="Process only the specified file. Takes precedence over --data-dir.",
    )(func)


def _force_option(func: Any) -> Any:
    """Add the ``--force`` flag to a Click command.

    When specified, the step is run even if it appears to already be
    complete (based on the ``status`` dict).
    """
    return click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Run the step even if it appears to already be complete.",
    )(func)


def _dry_run_option(func: Any) -> Any:
    """Add the ``--dry-run`` flag to a Click command."""
    return click.option(
        "--dry-run",
        is_flag=True,
        default=False,
        help="Show what would be processed without making any changes.",
    )(func)


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def app() -> None:
    """Local Feed Generator — Privacy-first RSS/Atom feed generator.

    Processes PDF, HTML, and MP4 files in a data directory and generates
    RSS feeds for personal consumption through mobile applications.
    """


# ---------------------------------------------------------------------------
# Process command  (T5.2)
# ---------------------------------------------------------------------------


@app.command()
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
def process(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
) -> None:
    """Process all files in the data directory (batch mode).

    Scans the data directory for content files (MP4, PDF, HTML) without
    JSON metadata, processes each file through the Producer pipeline,
    and generates both RSS feeds afterwards.

    When ``--file`` is specified, only that file is processed.
    When ``--force`` is specified, all files are processed even if
    already feed-eligible.
    """
    _load_config(config_file, data_dir)

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    # --- Scan -----------------------------------------------------------
    scanner = Scanner(data_dir=data_dir_path)

    if file_path is not None:
        # Process only the specified file.
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        relative_path = resolved_file.relative_to(data_dir_path)
        filetype = resolved_file.suffix.lower().lstrip(".")
        files = [
            ScannedFile(
                filepath=str(relative_path),
                absolute_path=str(resolved_file),
                filename=resolved_file.name,
                filetype=filetype,
            )
        ]
    else:
        if force:
            # Force processing: scan all source files regardless of status.
            all_sources = _find_source_files(data_dir_path, ("mp4", "pdf", "html"))
            files = []
            for source in all_sources:
                relative_path = source.relative_to(data_dir_path)
                filetype = source.suffix.lower().lstrip(".")
                files.append(
                    ScannedFile(
                        filepath=str(relative_path),
                        absolute_path=str(source),
                        filename=source.name,
                        filetype=filetype,
                    )
                )
        else:
            files = scanner.scan_directory()

    if not files:
        click.echo("No files to process.")
        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)
        return

    click.echo(f"Found {len(files)} file(s) to process.")

    # --- Process --------------------------------------------------------
    from local_feed_generator.producer import Producer

    producer = Producer(data_dir=str(data_dir_path))

    success_count = 0
    error_count = 0

    for i, file in enumerate(files, 1):
        click.echo(f"[{i}/{len(files)}] Processing {file.filepath}...")
        try:
            result = producer.process_file(file)
            if result is not None and result.feed_eligible:
                success_count += 1
            else:
                error_count += 1
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=file.filepath)

    click.echo(f"\nProcessing complete: {success_count} succeeded, {error_count} failed.")

    # --- Generate feeds -------------------------------------------------
    counts = _generate_feeds(data_dir_path)
    _print_feed_summary(counts)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Watch command  (T5.3)
# ---------------------------------------------------------------------------


class _WatchHandler(FileSystemEventHandler):
    """File-system event handler for watch mode.

    On file creation or modification:

    * **Source files** (``.mp4``, ``.pdf``, ``.html``) — process the file
      and regenerate feeds.
    * **JSON files** — check whether a corresponding source file exists;
      if not, log a warning about an orphaned JSON file.
    * **Other files** — ignored.

    Events are debounced per file path: rapid successive events for the
    same file within the :attr:`_debounce_interval` window are collapsed
    into a single processing.  Different files are tracked independently
    and are not blocked by each other's debounce timers.
    """

    #: Minimum time (seconds) between processing the same file path.
    _debounce_interval: float = 1.0

    # Maximum number of error log entries per minute before rate limiting
    # suppresses further messages (see T6.4).
    _max_errors_per_minute = 10

    # Window in seconds for error rate limiting.
    _error_window_seconds = 60

    # Periodic rescan interval in seconds (T6.3).
    # Fallback for file systems (NFS, network drives, etc.) that may not
    # reliably trigger watchdog events.  Can be overridden via the
    # ``RESCAN_INTERVAL`` environment variable in the ``watch`` command.
    # A value of ``0`` disables the periodic rescan.
    _rescan_interval: int = 300

    def __init__(self, data_dir: Path, producer: Any) -> None:
        self.data_dir = data_dir
        self.producer = producer
        # Last event timestamp per file path, for debouncing (T6.2).
        self._last_event: dict[str, float] = {}
        # Lock protecting _last_event, since watchdog dispatches events
        # from a separate thread.
        self._debounce_lock = threading.Lock()
        # Timestamps of recent errors, used for rate limiting (T6.4).
        self._error_timestamps: list[float] = []
        # Timestamp of the last periodic rescan, or None if no rescan
        # has occurred yet (T6.3).
        self.last_rescan_time: float | None = None

    # watchdog callbacks --------------------------------------------------

    def on_created(self, event: Any) -> None:
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_modified(self, event: Any) -> None:
        if not event.is_directory:
            self._handle_event(event.src_path)

    # internal helpers ----------------------------------------------------

    def _log_processing_error(
        self,
        exc: Exception,
        filepath: str,
        stage: str | None = None,
    ) -> None:
        """Log a processing error with rate limiting (T6.4).

        Tracks error timestamps to prevent log flooding. After
        ``_max_errors_per_minute`` errors within a 60-second window, further
        detailed error messages are suppressed and a summary message is logged
        instead.

        Args:
            exc: The exception that occurred.
            filepath: Path of the file being processed.
            stage: Optional processing stage label for additional context.
        """
        now = time.monotonic()
        # Trim timestamps older than the error window.
        self._error_timestamps = [
            ts for ts in self._error_timestamps if now - ts < self._error_window_seconds
        ]

        if len(self._error_timestamps) >= self._max_errors_per_minute:
            # Rate-limited: log a summary instead of the detailed error.
            _logger.warning(
                "Error rate limit reached (%d errors in %ds); "
                "suppressing detailed error for file=%s",
                self._max_errors_per_minute,
                self._error_window_seconds,
                filepath,
            )
            return

        # Record this error timestamp and log the detailed error.
        self._error_timestamps.append(now)
        log_error(exc, filepath=filepath, stage=stage)

    def _handle_event(self, filepath: str) -> None:
        """Dispatch a single file-system event.

        Handles exceptions internally so that a single file processing error
        does not crash the watcher (T6.4). The watcher continues monitoring
        after errors.

        Implements per-filepath debouncing (T6.2): if an event for the same
        path was received within :attr:`_debounce_interval` seconds, the
        event is skipped and the debounce window is extended.
        """
        now = time.monotonic()
        with self._debounce_lock:
            last = self._last_event.get(filepath)
            if last is not None and (now - last) < self._debounce_interval:
                # Within the debounce window — extend the window and skip.
                self._last_event[filepath] = now
                return
            self._last_event[filepath] = now

        source_path = Path(filepath)

        # --- JSON events: check for orphaned JSON -----------------------
        if source_path.suffix.lower() == ".json":
            self._handle_json_event(source_path)
            return

        # --- Source files: process and regenerate feeds -----------------
        if source_path.suffix.lower() not in SOURCE_EXTENSIONS:
            return

        try:
            relative_path = source_path.relative_to(self.data_dir)
            scanned = ScannedFile(
                filepath=str(relative_path),
                absolute_path=str(source_path),
                filename=source_path.name,
                filetype=source_path.suffix.lower().lstrip("."),
            )

            click.echo(f"Processing {scanned.filepath}...")
            self.producer.process_file(scanned)

            from local_feed_generator.feeds import FeedGenerator

            generator = FeedGenerator(data_dir=str(self.data_dir))
            generator.generate_all_feeds()
            click.echo(f"Feeds regenerated after processing {scanned.filepath}.")
        except Exception as exc:
            # Log the error without crashing the watcher (T6.4).
            # The watcher continues monitoring after errors.
            self._log_processing_error(
                exc,
                filepath=str(source_path),
                stage="processing",
            )

    def _handle_json_event(self, json_path: Path) -> None:
        """Check a JSON file for orphaned status."""
        source_filename = json_path.name[: -len(".json")]
        source_path = json_path.parent / source_filename

        if not source_path.exists():
            _logger.warning("Orphaned JSON file detected (no source file): %s", json_path)
        elif source_path.suffix.lower() not in SOURCE_EXTENSIONS:
            _logger.warning(
                "Orphaned JSON file detected (unsupported source extension): %s",
                json_path,
            )

    # ------------------------------------------------------------------
    # Periodic rescan (T6.3)
    # ------------------------------------------------------------------

    def _periodic_rescan(self) -> None:
        """Periodically scan the data directory for missed files.

        This is a fallback for file systems (NFS, network drives, etc.)
        that may not reliably trigger watchdog events.  Uses the
        :class:`~local_feed_generator.scanner.Scanner` to find files that
        need processing based on JSON metadata freshness (missing or
        outdated JSON, incomplete step statuses).  The scanner already
        handles duplicate detection via file modification times, so files
        that were already processed by the event handler will not be
        returned.

        After processing any newly-discovered files, feeds are regenerated.
        """
        self.last_rescan_time = time.time()
        _logger.info("Periodic rescan started for %s", self.data_dir)

        scanner = Scanner(data_dir=self.data_dir)
        files = scanner.scan_directory()

        if not files:
            _logger.info("Periodic rescan: no new files to process.")
            return

        _logger.info("Periodic rescan: found %d file(s) to process.", len(files))

        for file in files:
            click.echo(f"Rescan: Processing {file.filepath}...")
            try:
                result = self.producer.process_file(file)
                if result is not None and result.feed_eligible:
                    click.echo(f"Rescan: Processed {file.filepath} successfully.")
                else:
                    click.echo(
                        f"Rescan: Processing {file.filepath} resulted in ineligible metadata."
                    )
            except Exception as exc:
                self._log_processing_error(
                    exc,
                    filepath=file.filepath,
                    stage="rescan",
                )

        counts = _generate_feeds(self.data_dir)
        _print_feed_summary(counts)
        _logger.info("Periodic rescan complete. Feeds regenerated.")


@app.command()
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
def watch(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
) -> None:
    """Watch the data directory for changes (incremental mode).

    Starts a file-system watcher that processes new and modified files
    in real-time, regenerating feeds after each file is processed.
    Press ``Ctrl+C`` to stop.

    When ``--file`` is specified, only that file is processed on startup
    and watched thereafter.
    When ``--force`` is specified, all files are processed on startup
    even if already feed-eligible.
    """
    _load_config(config_file, data_dir)

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    if not data_dir_path.exists():
        click.echo(
            f"Error: Data directory does not exist: {data_dir_path}",
            err=True,
        )
        sys.exit(1)

    # --- Initial scan --------------------------------------------------
    scanner = Scanner(data_dir=data_dir_path)

    if file_path is not None:
        # Process only the specified file.
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        relative_path = resolved_file.relative_to(data_dir_path)
        filetype = resolved_file.suffix.lower().lstrip(".")
        files = [
            ScannedFile(
                filepath=str(relative_path),
                absolute_path=str(resolved_file),
                filename=resolved_file.name,
                filetype=filetype,
            )
        ]
    elif force:
        # Force processing: scan all source files regardless of status.
        all_sources = _find_source_files(data_dir_path, ("mp4", "pdf", "html"))
        files = []
        for source in all_sources:
            relative_path = source.relative_to(data_dir_path)
            filetype = source.suffix.lower().lstrip(".")
            files.append(
                ScannedFile(
                    filepath=str(relative_path),
                    absolute_path=str(source),
                    filename=source.name,
                    filetype=filetype,
                )
            )
    else:
        files = scanner.scan_directory()

    from local_feed_generator.producer import Producer

    if files:
        click.echo(f"Initial scan: found {len(files)} file(s) to process.")
        producer = Producer(data_dir=str(data_dir_path))

        for file in files:
            click.echo(f"Processing {file.filepath}...")
            try:
                producer.process_file(file)
            except Exception as exc:
                log_error(exc, filepath=file.filepath)

        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)
    else:
        click.echo("Initial scan: no files to process.")
        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)

    # --- Start watching ------------------------------------------------
    producer = Producer(data_dir=str(data_dir_path))
    event_handler = _WatchHandler(data_dir=data_dir_path, producer=producer)

    from watchdog.observers import Observer

    observer = Observer()
    observer.schedule(event_handler, str(data_dir_path), recursive=True)
    observer.start()

    # --- Start periodic rescan thread (T6.3) ----------------------------
    # Fallback for file systems (NFS, network drives, etc.) that may not
    # reliably trigger watchdog events.
    rescan_interval = int(os.environ.get("RESCAN_INTERVAL", event_handler._rescan_interval))
    stop_event = threading.Event()

    def _rescan_loop() -> None:
        """Background thread loop for periodic rescan."""
        while not stop_event.wait(timeout=rescan_interval):
            try:
                event_handler._periodic_rescan()
            except Exception as exc:
                _logger.error("Periodic rescan failed: %s", exc, exc_info=True)

    if rescan_interval > 0:
        rescan_thread = threading.Thread(target=_rescan_loop, daemon=True, name="periodic-rescan")
        rescan_thread.start()
        click.echo(f"Periodic rescan enabled (every {rescan_interval}s).")
    else:
        rescan_thread = None
        click.echo("Periodic rescan disabled.")

    click.echo(f"\nWatching {data_dir_path} for changes... (Ctrl+C to stop)")

    # --- Graceful shutdown --------------------------------------------
    shutdown_event = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_event
        shutdown_event = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not shutdown_event:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        # Gracefully shut down the rescan thread (T6.3).
        stop_event.set()
        if rescan_thread is not None:
            rescan_thread.join(timeout=5)
            if rescan_thread.is_alive():
                _logger.warning("Rescan thread did not shut down within timeout.")
        click.echo("\nWatcher stopped.")


# ---------------------------------------------------------------------------
# Feeds command  (T5.4)
# ---------------------------------------------------------------------------


@app.command("feeds")
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
def feeds(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
) -> None:
    """Generate RSS feeds from existing JSON metadata.

    Before generating feeds, finalizes any incomplete JSON metadata files
    (i.e., files with step statuses that are not yet ``"complete"``) by
    filling in missing required fields, setting applicable step statuses
    to ``"complete"``, and adding ``completed_at`` timestamps. This makes
    files feed-eligible so they appear in the generated feeds.

    When ``--file`` is specified, only that file's metadata is included
    in the generated feeds.
    """
    _load_config(config_file, data_dir)

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    # Finalize incomplete JSON metadata before generating feeds.
    _finalize_incomplete_json(data_dir_path, file_path=file_path, force=force)

    from local_feed_generator.feeds import FeedGenerator

    generator = FeedGenerator(data_dir=str(data_dir_path))

    if file_path is not None:
        # Generate feeds including only the specified file.
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)

    generator.generate_all_feeds()

    # Count items for the summary.
    podcast_count = 0
    article_count = 0
    for json_path in _find_json_files(data_dir_path):
        try:
            metadata = Metadata.from_json_file(json_path)
            if metadata.feed_eligible:
                if metadata.filetype == "mp4":
                    podcast_count += 1
                elif metadata.filetype in ("pdf", "html"):
                    article_count += 1
        except Exception:
            pass

    click.echo("Feeds generated successfully.")
    click.echo(f"  Podcast feed: {podcast_count} item(s)")
    click.echo(f"  Articles feed: {article_count} item(s)")


# ---------------------------------------------------------------------------
# Validate command  (T5.5)
# ---------------------------------------------------------------------------


@app.command()
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
def validate(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
) -> None:
    """Validate all JSON metadata files in the data directory.

    Before validation, finalizes any incomplete JSON metadata files
    (i.e., files with step statuses that are not yet ``"complete"``) by
    filling in missing required fields, setting applicable step statuses
    to ``"complete"``, and adding ``completed_at`` timestamps. This makes
    files feed-eligible.

    Then scans for all JSON metadata files, validates each against the
    schema, checks for orphaned JSON files, and checks for special
    characters in filenames.  Exits with code 0 when all files are valid,
    1 when errors are found.

    When ``--file`` is specified, only that file's JSON metadata is
    validated.
    """
    _load_config(config_file, data_dir)

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    # Finalize incomplete JSON metadata before validation.
    _finalize_incomplete_json(data_dir_path, file_path=file_path, force=force)

    errors: list[str] = []
    warnings: list[str] = []

    # --- Find all JSON files -------------------------------------------
    if file_path is not None:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        json_files = [resolved_file.with_name(resolved_file.name + ".json")]
    else:
        json_files = _find_json_files(data_dir_path)

    # --- Validate each JSON file against the schema --------------------
    for json_path in json_files:
        if not json_path.exists():
            continue
        try:
            Metadata.from_json_file(json_path)
        except Exception as exc:
            errors.append(f"{json_path}: Schema validation failed - {exc}")

    # --- Check for orphaned JSON files ---------------------------------
    for json_path in json_files:
        if not json_path.exists():
            continue
        if _is_orphaned_json(json_path):
            warnings.append(f"Orphaned JSON file (no matching source): {json_path}")

    # --- Check for special characters in filenames ---------------------
    if file_path is None:
        for root, _dirs, files in os.walk(data_dir_path):
            root_path = Path(root)
            for filename in files:
                if filename.lower() in _IGNORE_FILENAMES:
                    continue
                source_path = root_path / filename
                if source_path.suffix.lower() in SOURCE_EXTENSIONS:
                    warnings.extend(_check_special_characters(source_path))
    else:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is not None:
            warnings.extend(_check_special_characters(resolved_file))

    # --- Output report -------------------------------------------------
    click.echo("Validation Report")
    click.echo("=" * 50)
    click.echo(f"Checked {len(json_files)} JSON metadata file(s).")

    if errors:
        click.echo(f"\nErrors ({len(errors)}):")
        for error in errors:
            click.echo(f"  ERROR: {error}")

    if warnings:
        click.echo(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            click.echo(f"  WARNING: {warning}")

    if not errors and not warnings:
        click.echo("\nAll files are valid. No errors or warnings found.")

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: transcribe  (T5.7)
# ---------------------------------------------------------------------------


@app.command()
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
@_dry_run_option
def transcribe(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Transcribe MP4 files to markdown text.

    Scans the data directory for MP4 files that do not yet have a complete
    transcript, transcribes each one using the faster-whisper model, and
    writes the transcript to the adjacent JSON metadata file.

    Files that already have ``transcribe.status="complete"`` are skipped
    unless ``--force`` is specified.  Use ``--dry-run`` to preview which
    files would be processed without making any changes.

    When ``--file`` is specified, only that MP4 file is transcribed.
    """
    _load_config(config_file, data_dir)

    from local_feed_generator.config import get_config
    from local_feed_generator.services.transcriber import Transcriber

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    if file_path is not None:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        files = [resolved_file]
    else:
        files = _find_files_needing_transcription(data_dir_path, force=force)

    if not files:
        click.echo("No MP4 files need transcription.")
        return

    click.echo(f"Found {len(files)} MP4 file(s) to transcribe.")

    if dry_run:
        click.echo("[Dry run] Files that would be transcribed:")
        for f in files:
            relative = f.relative_to(data_dir_path)
            click.echo(f"  - {relative}")
        return

    transcriber = Transcriber()

    # Pre-emptive CUDA availability check: surface a clear, actionable error
    # before attempting to transcribe any files.
    from local_feed_generator.services.transcriber import CUDA_ERROR_MESSAGE, check_cuda_available

    config = get_config()
    if config.transcription_device == "cuda" and not check_cuda_available():
        click.echo(f"Error: {CUDA_ERROR_MESSAGE}", err=True)
        sys.exit(1)

    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = source_file.relative_to(data_dir_path)
        click.echo(f"[{i}/{len(files)}] Transcribing {relative_path}...")
        try:
            # Set transcribe step to in_progress and write before transcription begins.
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            now = datetime.now(UTC)
            # Initialize or update transcribe step map
            transcribe_step = metadata.get("transcribe") or {}
            transcribe_step["status"] = "in_progress"
            transcribe_step["start_time"] = int(now.timestamp())
            transcribe_step["input_size"] = source_file.stat().st_size
            metadata["transcribe"] = transcribe_step
            metadata.update(
                {
                    "filename": source_file.name,
                    "filepath": str(relative_path),
                    "filetype": "mp4",
                }
            )
            _write_json_metadata(metadata, source_file)

            transcript = transcriber.transcribe(str(source_file))

            # Update transcribe step to complete after successful transcription.
            transcribe_step["status"] = "complete"
            transcribe_step["end_time"] = int(now.timestamp())
            transcribe_step["output_size"] = len(transcript.encode("utf-8")) if transcript else 0
            transcribe_step["words"] = len(transcript.split()) if transcript else 0
            transcribe_step["model"] = (
                config.transcription_remote_model or "remote"
                if config.transcription_endpoint
                else config.transcription_model
            )
            metadata["transcribe"] = transcribe_step
            metadata["content"] = transcript
            _write_json_metadata(metadata, source_file)

            success_count += 1
            word_count = len(transcript.split())
            click.echo(f"  Transcribed {word_count} words.")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="transcription")

    click.echo(f"\nTranscription complete: {success_count} succeeded, {error_count} failed.")

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: summarize  (T5.8)
# ---------------------------------------------------------------------------


@app.command()
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
@_dry_run_option
def summarize(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Generate summaries from existing content.

    Scans the data directory for MP4, PDF, and HTML files that have content
    (transcript for MP4, extracted text for PDF/HTML) but no summary,
    generates a key-takeaways summary via the LLM, and writes the summary
    to the adjacent JSON metadata file.

    Files with ``summarize.status="complete"`` are skipped unless
    ``--force`` is specified.  Use ``--dry-run`` to preview which files
    would be processed without making any changes.

    When ``--file`` is specified, only that file is summarized.
    """
    _load_config(config_file, data_dir)

    from local_feed_generator.config import get_config
    from local_feed_generator.services.summarizer import Summarizer

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    if file_path is not None:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        files = [resolved_file]
    else:
        files = _find_files_needing_summary(data_dir_path, force=force)

    if not files:
        click.echo("No files need summarization.")
        return

    click.echo(f"Found {len(files)} file(s) to summarize.")

    if dry_run:
        click.echo("[Dry run] Files that would be summarized:")
        for f in files:
            relative = f.relative_to(data_dir_path)
            click.echo(f"  - {relative}")
        return

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = source_file.relative_to(data_dir_path)
        click.echo(f"[{i}/{len(files)}] Summarizing {relative_path}...")
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path)
            if metadata is None:
                click.echo(f"  WARNING: No JSON metadata found for {relative_path}; skipping.")
                error_count += 1
                continue

            content = metadata.get("content") or ""
            if not content.strip():
                click.echo(f"  WARNING: No content found for {relative_path}; skipping.")
                error_count += 1
                continue

            # Set summarize step to in_progress and write before summarization begins.
            now = datetime.now(UTC)
            config = get_config()
            summarize_step = metadata.get("summarize") or {}
            summarize_step["status"] = "in_progress"
            summarize_step["start_time"] = int(now.timestamp())
            summarize_step["input_size"] = len(content.encode("utf-8")) if content else 0
            metadata["summarize"] = summarize_step
            _write_json_metadata(metadata, source_file)

            summary, prompt_tokens, completion_tokens = summarizer.summarize(content)

            # Update summarize step to complete after successful summarization.
            summarize_step["status"] = "complete"
            summarize_step["end_time"] = int(now.timestamp())
            summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
            summarize_step["words"] = len(summary.split()) if summary else 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = prompt_tokens
            summarize_step["output_tokens"] = completion_tokens
            metadata["summarize"] = summarize_step
            metadata["summary"] = summary
            # For MP4 files, leave description blank and use summary instead.
            # For PDF/HTML files, keep the existing description from extraction.
            if metadata.get("filetype") == "mp4":
                metadata["description"] = ""
            _write_json_metadata(metadata, source_file)

            success_count += 1
            click.echo(f"  Tokens in: {prompt_tokens}, Tokens out: {completion_tokens}")
            word_count = len(summary.split())
            click.echo(f"  Summarized {word_count} words.")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="summarization")

    click.echo(f"\nSummarization complete: {success_count} succeeded, {error_count} failed.")

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: extract  (T5.9)
# ---------------------------------------------------------------------------


@app.command("extract")
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
@_dry_run_option
def extract(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Extract from PDF/HTML files.

    Scans the data directory for PDF and HTML files that do not yet have
    a complete metadata extraction, extracts metadata (title, description,
    date, size) using pypdf or BeautifulSoup, and writes the extracted data
    to the adjacent JSON metadata file.

    Files with ``extract.status="complete"`` are skipped unless ``--force``
    is specified.  Use ``--dry-run`` to preview which files would be
    processed without making any changes.

    When ``--file`` is specified, only that PDF/HTML file is processed.
    """
    _load_config(config_file, data_dir)

    from local_feed_generator.services.extractors import HTMLExtractor, PDFExtractor

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    if file_path is not None:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        files = [resolved_file]
    else:
        files = _find_files_needing_metadata_extraction(data_dir_path, force=force)

    if not files:
        click.echo("No PDF/HTML files need metadata extraction.")
        return

    click.echo(f"Found {len(files)} PDF/HTML file(s) to extract metadata from.")

    if dry_run:
        click.echo("[Dry run] Files that would be processed:")
        for f in files:
            relative = f.relative_to(data_dir_path)
            click.echo(f"  - {relative}")
        return

    pdf_extractor = PDFExtractor()
    html_extractor = HTMLExtractor()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = source_file.relative_to(data_dir_path)
        click.echo(f"[{i}/{len(files)}] Extracting metadata from {relative_path}...")
        try:
            filetype = source_file.suffix.lower().lstrip(".")
            if filetype == "pdf":
                extracted = pdf_extractor.extract(str(source_file))
            elif filetype == "html":
                extracted = html_extractor.extract(str(source_file))
            else:
                click.echo(f"  WARNING: Unsupported file type '{filetype}'; skipping.")
                error_count += 1
                continue

            # Set extract step to in_progress and write before extraction completes.
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            now = datetime.now(UTC)
            title = extracted.get("title") or source_file.stem
            if not title.strip():
                title = source_file.stem
            # Initialize or update extract step map
            extract_step = metadata.get("extract") or {}
            extract_step["status"] = "in_progress"
            extract_step["start_time"] = int(now.timestamp())
            extract_step["input_size"] = source_file.stat().st_size
            metadata["extract"] = extract_step
            metadata.update(
                {
                    "filename": source_file.name,
                    "filepath": str(relative_path),
                    "filetype": filetype,
                    "title": title,
                    "description": extracted.get("description") or "",
                    "date": extracted.get("date", 0),
                    "size": extracted.get("size", 0),
                }
            )
            _write_json_metadata(metadata, source_file)

            # Update extract step to complete after successful extraction.
            extract_step["status"] = "complete"
            extract_step["end_time"] = int(now.timestamp())
            content = extracted.get("content")
            extract_step["output_size"] = len(content.encode("utf-8")) if content else 0
            extract_step["words"] = len(content.split()) if content else 0
            metadata["extract"] = extract_step
            metadata["content"] = content
            _write_json_metadata(metadata, source_file)

            success_count += 1
            click.echo(f"  Title: {title}")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="metadata_extraction")

    click.echo(f"\nMetadata extraction complete: {success_count} succeeded, {error_count} failed.")

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: title  (T5.10)
# ---------------------------------------------------------------------------


@app.command()
@_data_dir_option_optional
@_config_file_option
@_file_option
@_force_option
@_dry_run_option
def title(
    data_dir: str | None,
    config_file: str | None,
    file_path: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Generate titles for files that lack proper titles.

    Scans for JSON metadata files where the title is missing, empty, or is
    just the filename stem (with underscores replaced by spaces), and
    generates a descriptive title using the LLM.

    Files that already have a proper title are skipped unless ``--force``
    is specified.  Use ``--dry-run`` to preview which files would be
    processed without making any changes.

    When ``--file`` is specified, only that file is processed.
    """
    _load_config(config_file, data_dir)

    from local_feed_generator.services.summarizer import Summarizer

    data_dir_path = _resolve_data_dir(data_dir, file_path)

    if file_path is not None:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            click.echo(f"Error: File not found: {file_path}", err=True)
            sys.exit(1)
        files = [resolved_file]
    else:
        files = _find_files_needing_title(data_dir_path, force=force)

    if not files:
        click.echo("No files need title generation.")
        return

    click.echo(f"Found {len(files)} file(s) to generate titles for.")

    if dry_run:
        click.echo("[Dry run] Files that would be processed:")
        for f in files:
            relative = f.relative_to(data_dir_path)
            click.echo(f"  - {relative}")
        return

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = source_file.relative_to(data_dir_path)
        click.echo(f"[{i}/{len(files)}] Generating title for {relative_path}...")
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            if not json_path.exists():
                click.echo(f"  WARNING: No JSON metadata found for {source_file.name}; skipping.")
                error_count += 1
                continue

            metadata_dict = json.loads(json_path.read_text(encoding="utf-8"))

            # Check if title needs generation (redundant with _find_files_needing_title
            # but needed for the --file case).
            title = metadata_dict.get("title", "")
            # Normalize string title to dict for backward compatibility.
            if isinstance(title, str):
                title = {"filename": title}
            has_generated = isinstance(title, dict) and bool(title.get("generated", "").strip())
            if has_generated and not force:
                click.echo("  Already has a proper title; skipping.")
                continue

            # Generate title from content.
            content = _get_content_for_title(source_file, metadata_dict)
            if not content.strip():
                click.echo("  WARNING: No content available for title generation; skipping.")
                error_count += 1
                continue

            # Get filename and extracted_title from existing metadata for context.
            filename_title = _clean_title(source_file.stem)
            extracted_title = metadata_dict.get("extracted_title")

            new_title, _, _ = summarizer.generate_title(
                content,
                filename=filename_title,
                extracted_title=extracted_title,
            )
            new_title = _clean_title(new_title) or new_title

            # Update metadata title dict with generated title.
            if isinstance(title, dict):
                title["generated"] = new_title
            else:
                title = {"filename": filename_title, "generated": new_title}
            metadata_dict["title"] = title

            _write_json_metadata(metadata_dict, source_file)

            success_count += 1
            click.echo(f"  Title: {new_title}")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="title_generation")

    click.echo(f"\nTitle generation complete: {success_count} succeeded, {error_count} failed.")

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# JSON finalization utility
# ---------------------------------------------------------------------------


def _finalize_incomplete_json(
    data_dir_path: Path,
    file_path: str | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Finalize incomplete JSON metadata files in the data directory.

    Finds source files (MP4, PDF, HTML) that have incomplete JSON metadata
    (i.e., not all applicable step maps have ``status == "complete"``) and
    assembles the final metadata by:

    - Filling in all required fields, using existing values from intermediate
      JSON, file system statistics, or LLM-generated titles as fallbacks.
    - Setting all applicable step maps to ``"complete"`` status.
    - Preserving existing ``content`` (transcript for MP4) fields.

    This makes files feed-eligible so they appear in generated feeds.

    When *file_path* is provided, only that file is finalized.  When *force*
    is ``True``, JSON is recreated even if the file is already feed-eligible.

    Args:
        data_dir_path: Absolute path to the data directory.
        file_path: Optional relative path to a single source file to finalize.
        force: When ``True``, finalize even feed-eligible files.

    Returns:
        A tuple of ``(success_count, error_count)``.
    """
    from local_feed_generator.config import get_config
    from local_feed_generator.services.summarizer import Summarizer

    if file_path is not None:
        resolved_file = _resolve_file_path(data_dir_path, file_path)
        if resolved_file is None or not resolved_file.exists():
            return 0, 0
        files = [resolved_file]
    else:
        files = _find_files_needing_json_generation(data_dir_path, force=force)

    if not files:
        return 0, 0

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for source_file in files:
        try:
            relative_path = source_file.relative_to(data_dir_path)
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}

            # Skip files with errors — they should not be finalized.
            if metadata.get("error"):
                continue

            filetype = source_file.suffix.lower().lstrip(".")
            stat_result = source_file.stat()
            file_size = stat_result.st_size
            file_mtime = int(stat_result.st_mtime)
            now = datetime.now(UTC)
            config = get_config()

            # Ensure required fields are present.
            title = metadata.get("title")
            # Normalize string title to dict for backward compatibility.
            if isinstance(title, str):
                title = {"filename": title}
            if not isinstance(title, dict):
                title = {}
            filename_title = title.get("filename") or _clean_title(source_file.stem)
            extracted_title = title.get("extracted") or metadata.get("extracted_title")

            # Generate title via LLM if not already present.
            if not title.get("generated") or not title["generated"].strip():
                generated = filename_title
                try:
                    if filetype == "mp4":
                        content = metadata.get("content") or ""
                        if content.strip():
                            generated, _, _ = summarizer.generate_title(
                                content,
                                filename=filename_title,
                            )
                            generated = _clean_title(generated) or generated
                    else:
                        content = _get_content_for_title(source_file, metadata)
                        if content.strip():
                            generated, _, _ = summarizer.generate_title(
                                content,
                                filename=filename_title,
                                extracted_title=extracted_title,
                            )
                            generated = _clean_title(generated) or generated
                except Exception as gen_exc:
                    _logger.warning(
                        "Failed to generate title for %s: %s; falling back to filename",
                        source_file,
                        gen_exc,
                    )
                    generated = filename_title
                title["generated"] = generated

            # Ensure filename key is set.
            title["filename"] = filename_title

            description = metadata.get("description")
            summary = metadata.get("summary")

            # Build the final metadata dict.
            final_metadata: dict[str, Any] = {
                "title": title,
                "description": description,
                "filename": source_file.name,
                "filepath": str(relative_path),
                "filetype": filetype,
                "date": metadata.get("date") or file_mtime,
                "size": metadata.get("size") or file_size,
            }

            # Set step maps to complete and include type-specific fields.
            now_ts = int(now.timestamp())
            if filetype == "mp4":
                # For MP4 files, description is left blank and summary holds
                # the LLM-generated key takeaways.
                final_metadata["description"] = ""
                final_metadata["summary"] = summary or ""
                final_metadata["transcribe"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": file_size,
                    "output_size": len((metadata.get("content") or "").encode("utf-8")),
                    "words": len((metadata.get("content") or "").split()),
                    "model": config.transcription_model,
                }
                final_metadata["summarize"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": len((metadata.get("content") or "").encode("utf-8")),
                    "output_size": len((summary or "").encode("utf-8")),
                    "words": len((summary or "").split()),
                    "model": config.llm_model,
                }
                final_metadata["content"] = metadata.get("content") or ""
            elif filetype in ("pdf", "html"):
                final_metadata["extract"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": file_size,
                    "output_size": len((metadata.get("content") or "").encode("utf-8")),
                    "words": len((metadata.get("content") or "").split()),
                }
                final_metadata["content"] = metadata.get("content")

            _write_json_metadata(final_metadata, source_file)

            success_count += 1
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(source_file), stage="json_finalization")

    return success_count, error_count


# ---------------------------------------------------------------------------
# Entry point  (T5.6)
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``local-feed-generator`` CLI."""
    app()


if __name__ == "__main__":
    main()
