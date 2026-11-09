"""Bounded concurrency orchestration for Focus Feed."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
import threading
from typing import Any

from focus_feed.config import get_config
from focus_feed.scanner import ScannedFile

_logger = logging.getLogger("focus_feed")


class ConcurrencyOrchestrator:
    """Run independent file-processing tasks using a bounded thread pool."""

    def __init__(self, max_workers: int | None = None) -> None:
        limit = max_workers if max_workers is not None else get_config().max_workers
        self._max_workers = max(1, limit)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="focus-feed",
        )
        self._closed = False

    @property
    def max_workers(self) -> int:
        """Return the configured number of worker threads."""
        return self._max_workers

    def submit(self, scanned_file: ScannedFile, producer: Any) -> Future[Any]:
        """Submit one scanned file to the producer."""
        return self._executor.submit(producer.process_file, scanned_file)

    def run_all(self, files: Sequence[ScannedFile], producer: Any) -> list[Any]:
        """Process files concurrently and return results in input order."""
        futures: dict[Future[Any], int] = {
            self.submit(scanned_file, producer): index for index, scanned_file in enumerate(files)
        }
        results: list[Any] = [None] * len(files)

        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                _logger.error(
                    "Unhandled error processing %s",
                    files[index].filepath,
                    exc_info=True,
                )
                results[index] = exc

        return results

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool once."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=wait)

    def __enter__(self) -> ConcurrencyOrchestrator:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown()


class InFlightRegistry:
    """Track and wait for processing tasks by key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

    def is_processing(self, key: str) -> bool:
        """Return whether a task is registered for *key*."""
        with self._lock:
            return key in self._events

    def register(self, key: str) -> threading.Event:
        """Register a task for *key* and return its completion event."""
        with self._lock:
            event = threading.Event()
            self._events[key] = event
            return event

    def unregister(self, key: str) -> None:
        """Unregister *key* and signal any waiters."""
        with self._lock:
            event = self._events.pop(key, None)
        if event is not None:
            event.set()

    def wait(self, key: str, timeout: float | None = None) -> bool:
        """Wait for the registered task for *key* to complete."""
        with self._lock:
            event = self._events.get(key)
        if event is None:
            return True
        return event.wait(timeout)
