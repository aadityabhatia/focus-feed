"""CLI for Focus Feed.

Provides a Click-based command-line interface with granular subcommands for
each step of the processing pipeline:

- ``process``          — Batch process all files in the given paths.
- ``watch``            — Watch directories for changes (incremental mode).
- ``feeds``            — Generate RSS feeds from existing JSON metadata.
- ``validate``         — Validate all JSON metadata files.
- ``transcribe``       — Transcribe audio/video files to markdown text.
- ``summarize``        — Generate summaries from existing content.
- ``extract``          — Extract metadata from PDF/HTML files.
- ``title``            — Generate titles for files that lack proper titles.
- ``create-feed``      — Create a template ``feed.json`` file in a target directory.
- ``cleanup``          — Remove unknown keys from JSON metadata files.

Files and directories are passed as positional arguments, e.g.::

    uv run focus-feed process ./data
    uv run focus-feed transcribe videos/lecture.mp4 --force
    uv run focus-feed watch ./data
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import click

from focus_feed.cli._helpers import (
    _ALLOWED_TOP_LEVEL_KEYS,
    _LOG_LEVEL_CHOICES,
    _LOG_LEVEL_MAP,
    _check_special_characters,
    _cleanup_json_file,
    _configure_logging_level,
    _expand_paths_to_files,
    _finalize_incomplete_json,
    _find_feed_eligible_sources,
    _find_files_needing_json_generation,
    _find_files_needing_metadata_extraction,
    _find_files_needing_summary,
    _find_files_needing_title,
    _find_files_needing_transcription,
    _find_json_files,
    _find_source_files,
    _find_unknown_keys,
    _generate_feeds,
    _get_content_for_title,
    _get_json_metadata,
    _is_path_within,
    _is_step_complete,
    _load_config,
    _print_feed_summary,
    _resolve_data_dir_from_paths,
    _resolve_paths,
    _write_json_metadata,
)
from focus_feed.cli._options import _config_file_option, _dry_run_option, _force_option
from focus_feed.cli._watch import _WatchHandler
from focus_feed.errors import log_error
from focus_feed.feeds import _is_orphaned_json
from focus_feed.filetypes import (
    DOCLING_FILETYPES,
    DOCUMENT_FILETYPES,
    MEDIA_FILETYPES,
    SUPPORTED_EXTENSIONS,
)
from focus_feed.metadata import Metadata
from focus_feed.producer import _check_reflection_md
from focus_feed.scanner import ScannedFile, Scanner
from focus_feed.utils import clean_title, now_timestamp, should_ignore_filename

_clean_title = clean_title
_should_ignore_filename = should_ignore_filename

__all__ = [
    "_ALLOWED_TOP_LEVEL_KEYS",
    "_LOG_LEVEL_CHOICES",
    "_LOG_LEVEL_MAP",
    "_WatchHandler",
    "_check_special_characters",
    "_cleanup_json_file",
    "_config_file_option",
    "_configure_logging_level",
    "_dry_run_option",
    "_expand_paths_to_files",
    "_finalize_incomplete_json",
    "_find_feed_eligible_sources",
    "_find_files_needing_json_generation",
    "_find_files_needing_metadata_extraction",
    "_find_files_needing_summary",
    "_find_files_needing_title",
    "_find_files_needing_transcription",
    "_find_json_files",
    "_find_source_files",
    "_find_unknown_keys",
    "_force_option",
    "_generate_feeds",
    "_get_content_for_title",
    "_get_json_metadata",
    "_is_path_within",
    "_is_step_complete",
    "_load_config",
    "_print_feed_summary",
    "_resolve_data_dir_from_paths",
    "_resolve_paths",
    "_write_json_metadata",
    "app",
    "cleanup",
    "create_feed",
    "extract",
    "feeds",
    "main",
    "process",
    "summarize",
    "title",
    "transcribe",
    "validate",
    "watch",
]

_logger = logging.getLogger("focus_feed")


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="focus-feed")
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(_LOG_LEVEL_CHOICES, case_sensitive=False),
    default=None,
    envvar="LOG_LEVEL",
    help="Set the logging level (default: INFO). "
    "Can also be set via the LOG_LEVEL environment variable.",
)
def app(log_level: str | None) -> None:
    """Focus Feed — Privacy-first RSS/Atom feed generator.

    Processes audio/video files (MP3, M4A, WAV, MP4, MKV, etc.) and
    documents (PDF, HTML) from the given paths, then generates RSS feeds
    for personal consumption through mobile applications.
    """
    # Configure logging level early, before any subcommand logging occurs.
    # Precedence: CLI option > LOG_LEVEL env var > default INFO.
    effective_level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    _configure_logging_level(effective_level)


# ---------------------------------------------------------------------------
# Process command  (T5.2)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def process(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Process all files in the given paths (batch mode).

    Scans the given paths for supported audio/video files and documents
    (PDF, HTML) that need processing, processes each file through the Producer pipeline,
    and generates both RSS feeds afterwards.

    When ``--force`` is specified, all files are processed even if
    already feed-eligible.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting process command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    # --- Scan -----------------------------------------------------------
    if force:
        # Force processing: expand all source files regardless of status.
        files = _expand_paths_to_files(resolved_paths)
    else:
        scanner = Scanner(data_dir=data_dir_path)
        scanned = scanner.scan_directory()
        # Adjust filepath to be relative to CWD instead of data_dir.
        cwd = Path.cwd()
        files = [
            ScannedFile(
                filepath=os.path.relpath(Path(f.absolute_path), cwd),
                absolute_path=f.absolute_path,
                filename=f.filename,
                filetype=f.filetype,
                current_stage=f.current_stage,
                json_exists=f.json_exists,
            )
            for f in scanned
        ]

    if not files:
        click.echo("No files to process.")
        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)
        return

    click.echo(f"Found {len(files)} file(s) to process.")
    _logger.info("Found %d file(s) to process", len(files))

    # --- Process --------------------------------------------------------
    from focus_feed.producer import Producer

    producer = Producer(force=force)

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
            log_error(exc, filepath=file.filepath, stage="processing")

    click.echo(f"\nProcessing complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Process complete: %d succeeded, %d failed", success_count, error_count)

    # --- Generate feeds -------------------------------------------------
    counts = _generate_feeds(data_dir_path)
    _print_feed_summary(counts)

    if error_count > 0:
        _logger.warning("Exiting with code 1 due to %d error(s)", error_count)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Watch command  (T5.3)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def watch(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Watch directories for changes (incremental mode).

    Starts a file-system watcher that processes new and modified files
    in real-time, regenerating feeds after each file is processed.
    Press ``Ctrl+C`` to stop.

    When ``--force`` is specified, all files are processed on startup
    even if already feed-eligible.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting watch command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    if not data_dir_path.exists():
        _logger.critical("Data directory does not exist: %s", data_dir_path)
        click.echo(
            f"Error: Data directory does not exist: {data_dir_path}",
            err=True,
        )
        sys.exit(1)

    # --- Initial scan --------------------------------------------------
    if force:
        # Force processing: expand all source files regardless of status.
        files = _expand_paths_to_files(resolved_paths)
    else:
        scanner = Scanner(data_dir=data_dir_path)
        scanned = scanner.scan_directory()
        # Adjust filepath to be relative to CWD instead of data_dir.
        cwd = Path.cwd()
        files = [
            ScannedFile(
                filepath=os.path.relpath(Path(f.absolute_path), cwd),
                absolute_path=f.absolute_path,
                filename=f.filename,
                filetype=f.filetype,
                current_stage=f.current_stage,
                json_exists=f.json_exists,
            )
            for f in scanned
        ]

    from focus_feed.producer import Producer

    if files:
        click.echo(f"Initial scan: found {len(files)} file(s) to process.")
        producer = Producer(force=force)

        for file in files:
            click.echo(f"Processing {file.filepath}...")
            try:
                producer.process_file(file)
            except Exception as exc:
                log_error(exc, filepath=file.filepath, stage="processing")

        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)
    else:
        click.echo("Initial scan: no files to process.")
        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)

    # --- Start watching ------------------------------------------------
    producer = Producer()
    event_handler = _WatchHandler(data_dir=data_dir_path, producer=producer)

    from watchdog.observers import Observer

    observer = Observer()
    observer.schedule(event_handler, str(data_dir_path), recursive=True)
    observer.start()
    _logger.info("Watching %s for changes", str(data_dir_path))

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
                log_error(exc, stage="periodic_rescan")

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
        _logger.info("Watcher stopped")


# ---------------------------------------------------------------------------
# Feeds command  (T5.4)
# ---------------------------------------------------------------------------


@app.command("feeds")
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def feeds(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Generate RSS feeds from existing, summarized JSON metadata.

    This command treats JSON sidecars as read-only. Source files without a
    parseable sidecar or non-empty summary are noted and skipped. The
    ``--force`` option is retained for compatibility and has no effect.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting feeds command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    eligible_sources = _find_feed_eligible_sources(data_dir_path, paths=resolved_paths)

    # Warn if no feed.json is found anywhere in the data directory.
    feed_json_files = list(data_dir_path.glob("**/feed.json"))
    if not feed_json_files:
        _logger.warning("No feed.json found at %s; feeds not generated", str(data_dir_path))

    from focus_feed.feeds import FeedGenerator

    generator = FeedGenerator(
        data_dir=str(data_dir_path),
        eligible_paths=eligible_sources,
        persist_hashes=False,
    )

    generator.generate_all_feeds()

    # Count items for the summary.
    podcast_count = 0
    article_count = 0
    for source_file in eligible_sources:
        json_path = source_file.with_name(source_file.name + ".json")
        try:
            metadata = Metadata.from_json_file(json_path)
        except Exception:
            continue
        if metadata.feed_eligible:
            if metadata.filetype in MEDIA_FILETYPES:
                podcast_count += 1
            elif metadata.filetype in DOCUMENT_FILETYPES:
                article_count += 1

    click.echo("Feeds generated successfully.")
    click.echo(f"  Podcast feed: {podcast_count} item(s)")
    click.echo(f"  Articles feed: {article_count} item(s)")
    _logger.info("Feeds generated: %d podcast, %d articles", podcast_count, article_count)


# ---------------------------------------------------------------------------
# Validate command  (T5.5)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def validate(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Validate all JSON metadata files in the given paths.

    Before validation, finalizes any incomplete JSON metadata files
    (i.e., files with step statuses that are not yet ``"complete"``) by
    filling in missing required fields, setting applicable step statuses
    to ``"complete"``, and adding timestamps. This makes files
    feed-eligible.

    When ``--force`` is specified, all files are processed even if already
    feed-eligible.

    Then scans for all JSON metadata files, validates each against the
    schema, checks for orphaned JSON files, and checks for special
    characters in filenames.  Exits with code 0 when all files are valid,
    1 when errors are found.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting validate command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    # Finalize incomplete JSON metadata before validation.
    _finalize_incomplete_json(data_dir_path, paths=resolved_paths, force=force)

    errors: list[str] = []
    warnings: list[str] = []

    # --- Find all JSON files -------------------------------------------
    json_files = _find_json_files(data_dir_path)

    # --- Validate each JSON file against the schema --------------------
    for json_path in json_files:
        if not json_path.exists():
            continue
        try:
            Metadata.from_json_file(json_path)
        except Exception as exc:
            log_error(exc, filepath=str(json_path), stage="validation")
            errors.append(f"{json_path}: Schema validation failed - {exc}")

    # --- Check for orphaned JSON files ---------------------------------
    for json_path in json_files:
        if not json_path.exists():
            continue
        if _is_orphaned_json(json_path):
            warnings.append(f"Orphaned JSON file (no matching source): {json_path}")

    # --- Check for special characters in filenames ---------------------
    for root, _dirs, files in os.walk(data_dir_path):
        root_path = Path(root)
        for filename in files:
            if _should_ignore_filename(filename):
                continue
            source_path = root_path / filename
            if source_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                warnings.extend(_check_special_characters(source_path))

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

    _logger.info("Validation complete: %d errors, %d warnings", len(errors), len(warnings))

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: transcribe  (T5.7)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def transcribe(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Transcribe audio/video files (MP3, M4A, WAV, MP4, MKV, etc.) to markdown text.

    Scans the given paths for audio/video files that do not yet have a
    complete transcript, transcribes each one using the remote transcription
    API, and writes the transcript to the adjacent JSON metadata file.

    Files that already have ``transcribe.status="complete"`` are skipped
    unless ``--force`` is specified.  Use ``--dry-run`` to preview which
    files would be processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting transcribe command")

    from focus_feed.config import get_config
    from focus_feed.services.transcriber import Transcriber

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_transcription(data_dir_path, force=force, paths=resolved_paths)

    if not files:
        click.echo("No audio/video files need transcription.")
        return

    click.echo(f"Found {len(files)} audio/video file(s) to transcribe.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be transcribed", len(files))
        click.echo("[Dry run] Files that would be transcribed:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    transcriber = Transcriber()

    config = get_config()

    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Transcribing {relative_path}...")
        metadata: Any = None
        try:
            # Set transcribe step to in_progress and write before transcription begins.
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            # Initialize or update transcribe step map
            transcribe_step = metadata.get("transcribe") or {}
            transcribe_step["status"] = "in_progress"
            transcribe_step["start_time"] = now_timestamp()
            transcribe_step["input_size"] = source_file.stat().st_size
            metadata["transcribe"] = transcribe_step
            metadata.update(
                {
                    "filename": source_file.name,
                    "filetype": source_file.suffix.lower().lstrip("."),
                    "mtime": source_file.stat().st_mtime,
                }
            )
            _write_json_metadata(metadata, source_file)

            transcript = transcriber.transcribe(str(source_file))

            # Update transcribe step to complete after successful transcription.
            transcribe_step["status"] = "complete"
            transcribe_step["end_time"] = now_timestamp()
            transcribe_step["output_size"] = len(transcript.encode("utf-8")) if transcript else 0
            transcribe_step["words"] = len(transcript.split()) if transcript else 0
            transcribe_step["model"] = config.transcription_remote_model or "remote"
            metadata["transcribe"] = transcribe_step
            metadata["content"] = transcript
            _write_json_metadata(metadata, source_file)

            success_count += 1
            word_count = len(transcript.split())
            click.echo(f"  Transcribed {word_count} words.")
        except Exception as exc:
            error_count += 1
            if isinstance(metadata, dict):
                step = metadata.setdefault("transcribe", {})
                if not isinstance(step, dict):
                    step = {}
                    metadata["transcribe"] = step
                step["status"] = None
                step["error"] = str(exc)
                step["output_size"] = 0
                _write_json_metadata(metadata, source_file)
            log_error(exc, filepath=str(relative_path), stage="transcription")

    click.echo(f"\nTranscription complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Transcribe complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: summarize  (T5.8)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def summarize(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Generate summaries from existing content.

    Scans the given paths for audio/video, PDF, and HTML files that have
    content (transcript for audio/video, extracted text for PDF/HTML) but no summary,
    generates a key-takeaways summary via the LLM, and writes the summary
    to the adjacent JSON metadata file.

    Files with ``summarize.status="complete"`` are skipped unless
    ``--force`` is specified.  Use ``--dry-run`` to preview which files
    would be processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting summarize command")

    from focus_feed.config import get_config
    from focus_feed.services.summarizer import Summarizer

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_summary(data_dir_path, force=force, paths=resolved_paths)

    if not files:
        click.echo("No files need summarization.")
        return

    click.echo(f"Found {len(files)} file(s) to summarize.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be summarized", len(files))
        click.echo("[Dry run] Files that would be summarized:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Summarizing {relative_path}...")
        reflection_questions = None
        metadata: Any = None
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
            config = get_config()
            summarize_step = metadata.get("summarize") or {}
            summarize_step["status"] = "in_progress"
            summarize_step["start_time"] = now_timestamp()
            summarize_step["input_size"] = len(content.encode("utf-8")) if content else 0
            metadata["summarize"] = summarize_step
            _write_json_metadata(metadata, source_file)

            # Check for reflection.md in the source file's directory.
            reflection_questions = _check_reflection_md(str(source_file))
            summary, prompt_tokens, completion_tokens = summarizer.summarize(
                content,
                reflection_questions=reflection_questions,
                filepath=str(relative_path),
                filetype=metadata.get("filetype"),
            )

            # Update summarize step to complete after successful summarization.
            summarize_step["status"] = "complete"
            summarize_step["end_time"] = now_timestamp()
            summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
            summarize_step["words"] = len(summary.split()) if summary else 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = prompt_tokens
            summarize_step["output_tokens"] = completion_tokens
            metadata["summarize"] = summarize_step
            metadata["summary"] = summary
            # For media files, leave description blank and use summary instead.
            # For PDF/HTML files, keep the existing description from extraction.
            if metadata.get("filetype") in MEDIA_FILETYPES:
                metadata["description"] = ""
            _write_json_metadata(metadata, source_file)

            success_count += 1
            click.echo(f"  Tokens in: {prompt_tokens}, Tokens out: {completion_tokens}")
            word_count = len(summary.split())
            click.echo(f"  Summarized {word_count} words.")
        except Exception as exc:
            error_count += 1
            if isinstance(metadata, dict):
                step = metadata.setdefault("summarize", {})
                if not isinstance(step, dict):
                    step = {}
                    metadata["summarize"] = step
                step["status"] = None
                step["error"] = str(exc)
                step["output_size"] = 0
                _write_json_metadata(metadata, source_file)
            log_error(exc, filepath=str(relative_path), stage="summarization")

    click.echo(f"\nSummarization complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Summarize complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: extract  (T5.9)
# ---------------------------------------------------------------------------


@app.command("extract")
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def extract(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Extract from PDF, HTML, and Docling-supported document formats.

    Scans the given paths for PDF, HTML, and Docling-supported document
    formats that do not yet have a complete metadata extraction, extracts
    metadata (title, description, date, size) using pypdf, BeautifulSoup, or
    Docling, and writes the extracted data
    to the adjacent JSON metadata file.

    Files with ``extract.status="complete"`` are skipped unless ``--force``
    is specified.  Use ``--dry-run`` to preview which files would be
    processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting extract command")

    from focus_feed.services.extractors import DoclingExtractor, HTMLExtractor, PDFExtractor

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_metadata_extraction(
        data_dir_path, force=force, paths=resolved_paths
    )

    if not files:
        click.echo("No PDF/HTML/Docling files need metadata extraction.")
        return

    click.echo(f"Found {len(files)} PDF/HTML/Docling file(s) to extract metadata from.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be processed", len(files))
        click.echo("[Dry run] Files that would be processed:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    pdf_extractor = PDFExtractor()
    html_extractor = HTMLExtractor()
    docling_extractor = DoclingExtractor()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Extracting metadata from {relative_path}...")
        metadata: Any = None
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            filetype = source_file.suffix.lower().lstrip(".")
            if filetype == "pdf":
                extracted = pdf_extractor.extract(str(source_file))
            elif filetype == "html":
                extracted = html_extractor.extract(str(source_file))
            elif filetype in DOCLING_FILETYPES:
                extracted = docling_extractor.extract(str(source_file))
            else:
                click.echo(f"  WARNING: Unsupported file type '{filetype}'; skipping.")
                error_count += 1
                continue

            # Set extract step to in_progress and write before extraction completes.
            title = extracted.get("title") or source_file.stem
            if not title.strip():
                title = source_file.stem
            if isinstance(metadata.get("title"), dict):
                metadata["title"]["filename"] = title
                metadata["title"]["extracted"] = extracted.get("extracted")
                metadata["title"].setdefault("generated", None)
            else:
                metadata["title"] = {
                    "filename": title,
                    "extracted": extracted.get("extracted"),
                    "generated": None,
                }
            # Initialize or update extract step map
            extract_step = metadata.get("extract") or {}
            extract_step["status"] = "in_progress"
            extract_step["start_time"] = now_timestamp()
            extract_step["input_size"] = source_file.stat().st_size
            metadata["extract"] = extract_step
            metadata.update(
                {
                    "filename": source_file.name,
                    "filetype": filetype,
                    "mtime": source_file.stat().st_mtime,
                    "description": extracted.get("description") or "",
                    "date": extracted.get("date", 0),
                    "publication_date": extracted.get("publication_date"),
                    "size": extracted.get("size", 0),
                }
            )
            _write_json_metadata(metadata, source_file)

            # Update extract step to complete after successful extraction.
            extract_step["status"] = "complete"
            extract_step["end_time"] = now_timestamp()
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
            if isinstance(metadata, dict):
                step = metadata.setdefault("extract", {})
                if not isinstance(step, dict):
                    step = {}
                    metadata["extract"] = step
                step["status"] = None
                step["error"] = str(exc)
                step["output_size"] = 0
                _write_json_metadata(metadata, source_file)
            log_error(exc, filepath=str(relative_path), stage="metadata_extraction")

    click.echo(f"\nMetadata extraction complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Extract complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: title  (T5.10)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def title(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Generate titles for files that lack proper titles.

    Scans for JSON metadata files where the title is missing, empty, or is
    just the filename stem (with underscores replaced by spaces), and
    generates a descriptive title using the LLM.

    Files that already have a proper title are skipped unless ``--force``
    is specified.  Use ``--dry-run`` to preview which files would be
    processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting title command")

    from focus_feed.config import get_config
    from focus_feed.services.summarizer import Summarizer

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_title(data_dir_path, force=force, paths=resolved_paths)

    if not files:
        click.echo("No files need title generation.")
        return

    click.echo(f"Found {len(files)} file(s) to generate titles for.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be processed", len(files))
        click.echo("[Dry run] Files that would be processed:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Generating title for {relative_path}...")
        metadata_dict: Any = {}
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            if not json_path.exists():
                click.echo(f"  WARNING: No JSON metadata found for {source_file.name}; skipping.")
                error_count += 1
                continue

            metadata_dict = json.loads(json_path.read_text(encoding="utf-8"))

            # Check if title needs generation (redundant with _find_files_needing_title
            # but needed for the force case).
            title = metadata_dict.get("title", "")
            # Normalize string title to dict for backward compatibility.
            if isinstance(title, str):
                title = {"filename": title, "extracted": None, "generated": None}
            has_generated = isinstance(title, dict) and bool((title.get("generated") or "").strip())
            if has_generated and not force:
                click.echo("  Already has a proper title; skipping.")
                continue

            # Generate title from content.
            content = _get_content_for_title(source_file, metadata_dict)
            if not content.strip():
                click.echo("  WARNING: No content available for title generation; skipping.")
                error_count += 1
                continue

            # Get filename and extracted title from the nested title metadata.
            filename_title = _clean_title(source_file.stem)
            extracted = title.get("extracted") if isinstance(title, dict) else None

            # Initialize the generate_title step map before the LLM call.
            config = get_config()
            generate_title_step = metadata_dict.get("generate_title") or {}
            generate_title_step["status"] = "in_progress"
            generate_title_step["start_time"] = now_timestamp()
            generate_title_step["input_size"] = len(content.encode("utf-8")) if content else 0
            generate_title_step["output_size"] = 0
            generate_title_step["words"] = 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = 0
            generate_title_step["output_tokens"] = 0
            metadata_dict["generate_title"] = generate_title_step
            _write_json_metadata(metadata_dict, source_file)

            new_title, prompt_tokens, completion_tokens = summarizer.generate_title(
                content,
                filename=filename_title,
                extracted=extracted,
            )
            new_title = _clean_title(new_title) or new_title

            # Update the generate_title step map with completion info.
            generate_title_step["status"] = "complete"
            generate_title_step["end_time"] = now_timestamp()
            generate_title_step["output_size"] = len(new_title.encode("utf-8")) if new_title else 0
            generate_title_step["words"] = len(new_title.split()) if new_title else 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = prompt_tokens
            generate_title_step["output_tokens"] = completion_tokens
            metadata_dict["generate_title"] = generate_title_step

            # Update metadata title dict with generated title.
            if isinstance(title, dict):
                title["filename"] = filename_title
                title["generated"] = new_title
            else:
                title = {
                    "filename": filename_title,
                    "extracted": extracted,
                    "generated": new_title,
                }
            metadata_dict["title"] = title

            _write_json_metadata(metadata_dict, source_file)

            success_count += 1
            click.echo(f"  Title: {new_title}")
            click.echo(f"  Tokens in: {prompt_tokens}, Tokens out: {completion_tokens}")
        except Exception as exc:
            error_count += 1
            if isinstance(metadata_dict, dict):
                step = metadata_dict.setdefault("generate_title", {})
                if not isinstance(step, dict):
                    step = {}
                    metadata_dict["generate_title"] = step
                step["status"] = None
                step["error"] = str(exc)
                step["output_size"] = 0
                _write_json_metadata(metadata_dict, source_file)
            log_error(exc, filepath=str(relative_path), stage="title_generation")

    click.echo(f"\nTitle generation complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Title complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# JSON finalization utility
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Create-feed command
# ---------------------------------------------------------------------------


# Template for the feed.json file created by the ``create-feed`` subcommand.
# The ``link`` field is intentionally omitted: feed links are derived from
# ``BASE_URL`` + relative path at generation time (see feeds.py).
_FEED_JSON_TEMPLATE: dict[str, Any] = {
    "title": "Feed Title",
    "description": "Feed description",
    "language": "en-us",
    "author": "Feed Author",
    "image": {
        "url": "https://www.example.com/logo.jpg",
        "title": "Logo",
        "link": "https://www.example.com/",
    },
}


@app.command("create-feed")
@_config_file_option
@click.argument(
    "directory",
    type=click.Path(file_okay=False, dir_okay=True),
)
def create_feed(config_file: str | None, directory: str) -> None:
    """Create a template ``feed.json`` file in a target directory.

    Creates a ``feed.json`` file with default template fields (title,
    description, language, author, image) in the specified directory.
    The directory is created if it does not exist. If a ``feed.json``
    already exists, it is overwritten.
    """
    _logger.info("Starting create-feed command")
    target_dir = Path(directory)
    feed_json_path = target_dir / "feed.json"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        feed_json_path.write_text(json.dumps(_FEED_JSON_TEMPLATE, indent=2), encoding="utf-8")
        _logger.info("Created feed.json at %s", str(feed_json_path))
    except (OSError, PermissionError) as exc:
        log_error(
            exc,
            filepath=str(feed_json_path),
            stage="feed_json_creation",
        )
        raise click.ClickException(f"Failed to create feed.json: {exc}") from exc

    click.echo(f"Created feed.json at {feed_json_path}")


# ---------------------------------------------------------------------------
# Cleanup command
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@click.argument("paths", nargs=-1)
def cleanup(config_file: str | None, paths: tuple[str, ...]) -> None:
    """Remove unknown keys and useless sidecars from JSON metadata files.

    Scans the given paths for ``.json`` metadata files (excluding
    ``feed.json`` and other ignored filenames) and removes any keys
    that are not part of the Metadata schema:

    - Unknown top-level keys (not in ``Metadata.model_fields``).
    - Unknown keys inside the ``title`` dict (``filename``, ``extracted``,
      ``generated`` are allowed; a plain-string title is left as-is).
    - Unknown keys inside step-map dicts (``transcribe``, ``summarize``,
      ``extract``, ``generate_title``) — only ``StepMap.model_fields``
      keys are kept.

    Sidecars with neither a non-empty summary nor non-empty content are
    deleted. Other files are rewritten in-place only when unknown keys were
    found. A summary of checked, modified, deleted files and removed keys is
    printed.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting cleanup command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    json_files = _find_json_files(data_dir_path)

    if not json_files:
        click.echo("No JSON metadata files found.")
        return

    files_checked = 0
    files_modified = 0
    files_deleted = 0
    total_keys_removed = 0

    for json_path in json_files:
        files_checked += 1
        data = _get_json_metadata(json_path)
        if data is not None:
            summary = data.get("summary")
            content = data.get("content")
            has_summary = isinstance(summary, str) and bool(summary.strip())
            has_content = isinstance(content, str) and bool(content.strip())
            if not has_summary and not has_content:
                try:
                    json_path.unlink()
                    files_deleted += 1
                    relative = os.path.relpath(json_path, Path.cwd())
                    click.echo(f"  {relative}: deleted (no summary or content)")
                    _logger.info("Deleted sidecar without summary/content: %s", json_path)
                    continue
                except OSError as exc:
                    log_error(exc, filepath=str(json_path), stage="metadata_delete")
        try:
            keys_removed, unknown_key_paths = _cleanup_json_file(json_path)
        except OSError:
            # Error already logged by _cleanup_json_file.
            continue

        if keys_removed > 0:
            files_modified += 1
            total_keys_removed += keys_removed
            relative = os.path.relpath(json_path, Path.cwd())
            click.echo(f"  {relative}: removed {keys_removed} unknown key(s):")
            for key_path in unknown_key_paths:
                click.echo(f"    - {key_path}")
            _logger.info(
                "Cleaned %s: removed %d unknown key(s) %s",
                json_path,
                keys_removed,
                unknown_key_paths,
            )

    click.echo(
        f"\nCleanup complete: checked {files_checked} file(s), "
        f"modified {files_modified} file(s), deleted {files_deleted} file(s), "
        f"removed {total_keys_removed} key(s)."
    )
    _logger.info(
        "Cleanup complete: %d files checked, %d modified, %d deleted, %d keys removed",
        files_checked,
        files_modified,
        files_deleted,
        total_keys_removed,
    )


# ---------------------------------------------------------------------------
# Entry point  (T5.6)
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``focus-feed`` CLI."""
    app()


if __name__ == "__main__":
    main()
