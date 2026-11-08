"""Watch-mode filesystem event handler."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import Any

import click
from watchdog.events import FileSystemEventHandler

from focus_feed.errors import log_error
from focus_feed.feeds import FeedGenerator
from focus_feed.filetypes import SUPPORTED_EXTENSIONS
from focus_feed.scanner import ScannedFile, Scanner

_logger = logging.getLogger("focus_feed")


def _cli() -> Any:
    """Return the public CLI module for runtime-patchable dependencies."""
    from focus_feed import cli

    return cli


def _time() -> Any:
    """Return the CLI time module so existing patches remain effective."""
    return _cli().time


class _WatchHandler(FileSystemEventHandler):
    """File-system event handler for watch mode.

    On file creation or modification:

    * **Source files** (audio/video files and documents such as PDF/HTML) —
      process the file and regenerate feeds.
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
        _cli()._logger.info("Watch handler initialized for %s", str(self.data_dir))
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
            path = event.src_path
            _cli()._logger.info("Detected changes in %s", str(path))
            self._handle_event(path)

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
        now = _time().monotonic()
        # Trim timestamps older than the error window.
        self._error_timestamps = [
            ts for ts in self._error_timestamps if now - ts < self._error_window_seconds
        ]

        if len(self._error_timestamps) >= self._max_errors_per_minute:
            # Rate-limited: log a summary instead of the detailed error.
            _cli()._logger.warning(
                "Error rate limit reached (%d errors in %ds); "
                "suppressing detailed error for file=%s",
                self._max_errors_per_minute,
                self._error_window_seconds,
                filepath,
            )
            return

        # Record this error timestamp and log the detailed error.
        self._error_timestamps.append(now)
        getattr(_cli(), "log_error", log_error)(exc, filepath=filepath, stage=stage)

    def _handle_event(self, filepath: str) -> None:
        """Dispatch a single file-system event.

        Handles exceptions internally so that a single file processing error
        does not crash the watcher (T6.4). The watcher continues monitoring
        after errors.

        Implements per-filepath debouncing (T6.2): if an event for the same
        path was received within :attr:`_debounce_interval` seconds, the
        event is skipped and the debounce window is extended.
        """
        now = _time().monotonic()
        with self._debounce_lock:
            last = self._last_event.get(filepath)
            if last is not None and (now - last) < self._debounce_interval:
                # Within the debounce window — extend the window and skip.
                self._last_event[filepath] = now
                return
            self._last_event[filepath] = now

        source_path = Path(filepath)

        if _cli()._should_ignore_filename(source_path.name):
            return

        # --- JSON events: check for orphaned JSON -----------------------
        if source_path.suffix.lower() == ".json":
            self._handle_json_event(source_path)
            return

        # --- Source files: process and regenerate feeds -----------------
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # Check if the file has an error state in its JSON metadata and the
        # source file is not newer than the JSON.  In watch mode, a file
        # modification event may fire even when the source content hasn't
        # actually changed (e.g. metadata touch).  If the JSON already has
        # an error and the source isn't newer, there's no point retrying
        # without --force.
        json_path = source_path.with_name(source_path.name + ".json")
        if json_path.exists():
            existing_metadata = _cli()._get_json_metadata(json_path)
            if existing_metadata and existing_metadata.get("error"):
                source_mtime = source_path.stat().st_mtime
                json_mtime = json_path.stat().st_mtime
                if source_mtime <= json_mtime:
                    _cli()._logger.warning(
                        "File %s has error state but source not modified; use --force to retry",
                        filepath,
                    )
                    return

        try:
            relative_path = os.path.relpath(source_path, Path.cwd())
            scanned = ScannedFile(
                filepath=relative_path,
                absolute_path=str(source_path),
                filename=source_path.name,
                filetype=source_path.suffix.lower().lstrip("."),
            )

            click.echo(f"Processing {scanned.filepath}...")
            self.producer.process_file(scanned)

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
        if _cli()._should_ignore_filename(json_path.name):
            return

        source_filename = json_path.name[: -len(".json")]
        source_path = json_path.parent / source_filename

        if not source_path.exists():
            _cli()._logger.warning("Orphaned JSON file detected (no source file): %s", json_path)
        elif source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            _cli()._logger.warning(
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
        :class:`~focus_feed.scanner.Scanner` to find files that
        need processing based on JSON metadata freshness (missing or
        outdated JSON, incomplete step statuses).  The scanner already
        handles duplicate detection via file modification times, so files
        that were already processed by the event handler will not be
        returned.

        After processing any newly-discovered files, feeds are regenerated.
        """
        self.last_rescan_time = _time().time()
        _cli()._logger.info("Periodic rescan started for %s", self.data_dir)

        scanner = Scanner(data_dir=self.data_dir)
        files = scanner.scan_directory()

        if not files:
            _cli()._logger.info("Periodic rescan: no new files to process.")
            return

        _cli()._logger.info("Periodic rescan: found %d file(s) to process.", len(files))

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

        counts = _cli()._generate_feeds(self.data_dir)
        _cli()._print_feed_summary(counts)
        _cli()._logger.info("Periodic rescan complete. Feeds regenerated.")
