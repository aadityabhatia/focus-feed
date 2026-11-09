"""Tests for bounded parallel file processing."""

from concurrent.futures import Future
from pathlib import Path
import threading
import time
from unittest import mock

from click.testing import CliRunner

from focus_feed.cli import _WatchHandler, app
from focus_feed.concurrency import ConcurrencyOrchestrator, InFlightRegistry
from focus_feed.scanner import ScannedFile
from focus_feed.services.summarizer import LLMResponse, Summarizer, reset_summarize_semaphore


def _scanned_file(filename: str, directory: Path | None = None) -> ScannedFile:
    """Build a scanned file with predictable paths for concurrency tests."""
    path = (directory or Path("/data")) / filename
    return ScannedFile(
        filepath=filename,
        absolute_path=str(path),
        filename=filename,
        filetype=path.suffix.lstrip("."),
    )


def _eligible_metadata(feed_eligible: bool = True):
    """Create the Metadata shape the process command uses for its result count."""
    from focus_feed.metadata import Metadata, StepMap

    status = "complete" if feed_eligible else None
    return Metadata(
        title={"filename": "test", "extracted": None, "generated": None},
        filename="test.mp4",
        filetype="mp4",
        size=0,
        transcribe=StepMap(status=status),
        summarize=StepMap(status=status),
    )


def test_summarize_concurrency_cap(valid_env, monkeypatch):
    """Concurrent summarize calls should honor the configured semaphore cap."""
    from focus_feed.config import get_config

    monkeypatch.setenv("MAX_SUMMARIZE_CONCURRENCY", "2")
    reset_summarize_semaphore()
    config = get_config()
    assert config.max_summarize_concurrency == 2

    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def chat_completion(**_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.1)
            return LLMResponse("summary text", 1, 1)
        finally:
            with lock:
                active -= 1

    summarizer = Summarizer(client=mock.MagicMock())
    results: list[tuple[str, int, int]] = []
    with mock.patch.object(Summarizer, "_chat_completion", side_effect=chat_completion):
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    summarizer.summarize("some transcript text", filetype="mp4")
                )
            )
            for _ in range(6)
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            assert all(not thread.is_alive() for thread in threads)
            assert len(results) == 6
            assert 1 <= maximum_active <= 2
        finally:
            reset_summarize_semaphore()


def test_orchestrator_runs_files_concurrently():
    """The bounded worker pool should run all four files at once when possible."""
    barrier = threading.Barrier(4, timeout=5)
    sentinel = object()

    class BarrierProducer:
        def process_file(self, scanned_file):
            barrier.wait()
            return sentinel

    files = [_scanned_file(f"file-{index}.mp4") for index in range(4)]
    with ConcurrencyOrchestrator(max_workers=4) as orchestrator:
        results = orchestrator.run_all(files, BarrierProducer())

    assert results == [sentinel] * 4


def test_orchestrator_isolates_per_file_errors():
    """An exception from one file should be returned without losing other results."""

    class SelectiveProducer:
        def process_file(self, scanned_file):
            if scanned_file.filename == "bad.mp4":
                raise RuntimeError("file failed")
            return f"processed {scanned_file.filename}"

    files = [_scanned_file(name) for name in ("first.mp4", "bad.mp4", "last.mp4")]
    with ConcurrencyOrchestrator(max_workers=3) as orchestrator:
        results = orchestrator.run_all(files, SelectiveProducer())

    assert results[0] == "processed first.mp4"
    assert isinstance(results[1], RuntimeError)
    assert str(results[1]) == "file failed"
    assert results[2] == "processed last.mp4"


def test_in_flight_registry():
    """Registry operations should signal completion and unknown-key waits return immediately."""
    registry = InFlightRegistry()

    start = time.monotonic()
    assert registry.wait("unknown", timeout=1)
    assert time.monotonic() - start < 0.5

    event = registry.register("video.mp4")
    assert registry.is_processing("video.mp4")
    assert not registry.wait("video.mp4", timeout=0.01)

    registry.unregister("video.mp4")
    assert event.is_set()
    assert not registry.is_processing("video.mp4")
    assert registry.wait("video.mp4", timeout=0)


def test_watch_handler_in_flight_guard(valid_env, tmp_path):
    """Events during active processing should be skipped, then process again afterward."""
    source_file = tmp_path / "video.mp4"
    source_file.write_bytes(b"video")
    started = threading.Event()
    futures: list[Future] = []

    class AsyncOrchestrator:
        def submit(self, scanned_file, producer):
            future: Future[object] = Future()
            futures.append(future)

            def process():
                try:
                    future.set_result(producer.process_file(scanned_file))
                except Exception as exc:
                    future.set_exception(exc)

            threading.Thread(target=process, daemon=True).start()
            return future

        def shutdown(self, **_kwargs):
            return None

    producer = mock.MagicMock()

    def process_file(_scanned_file):
        started.set()
        time.sleep(0.2)
        return "done"

    producer.process_file.side_effect = process_file
    handler = _WatchHandler(
        data_dir=tmp_path,
        producer=producer,
        orchestrator=AsyncOrchestrator(),
    )

    with (
        mock.patch("focus_feed.cli.time.monotonic", side_effect=[0.0, 2.0, 4.0]),
        mock.patch("focus_feed.cli._watch.FeedGenerator") as feed_generator,
    ):
        handler._handle_event(str(source_file))
        assert started.wait(timeout=2)
        handler._handle_event(str(source_file))
        assert len(futures) == 1
        futures[0].result(timeout=2)
        assert handler._in_flight.wait(str(source_file), timeout=2)

        producer.process_file.assert_called_once()
        handler._handle_event(str(source_file))
        assert len(futures) == 2
        futures[1].result(timeout=2)

    assert producer.process_file.call_count == 2
    assert feed_generator.call_count == 2


def test_process_command_accepts_concurrency_flags(valid_env, tmp_path):
    """The process command should accept both concurrency options and process files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "first.mp4").write_bytes(b"video one")
    (data_dir / "second.pdf").write_bytes(b"document two")
    producer = mock.MagicMock()
    producer.process_file.return_value = _eligible_metadata()

    with (
        mock.patch("focus_feed.producer.Producer", return_value=producer),
        mock.patch("focus_feed.cli._generate_feeds", return_value={"podcast": 0, "articles": 0}),
    ):
        result = CliRunner().invoke(
            app,
            [
                "process",
                str(data_dir),
                "--max-workers",
                "2",
                "--max-summarize-jobs",
                "2",
            ],
        )

    assert result.exit_code == 0, result.output
    assert producer.process_file.call_count == 2
