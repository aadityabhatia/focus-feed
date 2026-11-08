"""Tests for the CLI module (Phase 5).

Tests cover:
- CLI framework: group, shared options, config loading, help text
- Process command: orchestration, progress, summary, exit codes
- Watch command: observer startup, event handling, graceful shutdown, error handling
- Generate-feeds command: skip processing, read JSON, generate feeds, summary
- Validate command: schema validation, orphaned JSON, special characters, exit codes
- Entry point: main() function, pyproject.toml configuration
- Positional args: process only specific files/directories
- --force flag: force processing even if complete
- Logging level: --log-level option, LOG_LEVEL env var, _configure_logging_level
"""

import json
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
import pytest

from focus_feed.cli import app
from focus_feed.scanner import ScannedFile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_ENV_KEYS = [
    "BASE_URL",
    "LLM_ENDPOINT",
    "LLM_API_KEY",
    "LLM_MODEL",
    "PROMPT_ARTICLE",
    "PROMPT_ARTICLE_REFLECTION",
    "PROMPT_PODCAST",
    "PROMPT_PODCAST_REFLECTION",
    "PROMPT_TITLE",
    "TRANSCRIPTION_ENDPOINT",
    "TRANSCRIPTION_API_KEY",
]


@pytest.fixture
def clear_required_env(monkeypatch):
    """Clear all required env vars, yielding control, then they're auto-restored by monkeypatch."""
    for key in REQUIRED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def _restore_env(reset_config):
    """Apply the shared config reset to every CLI test."""


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with test files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def watch_mocks():
    """Provide a context manager that sets up common watch command mocks.

    Reduces nested ``with mock.patch`` blocks from 4-5 levels to 1.
    """
    from contextlib import contextmanager

    @contextmanager
    def _setup(
        gen_return=None,
        producer_return=None,
        producer_process_return=None,
    ):
        if gen_return is None:
            gen_return = {"podcast": 0, "articles": 0}
        with (
            mock.patch("focus_feed.cli._generate_feeds") as mock_gen,
            mock.patch("focus_feed.producer.Producer") as MockProducer,
            mock.patch("watchdog.observers.Observer") as MockObserver,
            mock.patch("focus_feed.cli.time.sleep") as mock_sleep,
        ):
            mock_gen.return_value = gen_return
            if producer_process_return is not None:
                mock_producer = mock.MagicMock()
                mock_producer.process_file.return_value = producer_process_return
                MockProducer.return_value = mock_producer
            elif producer_return is not None:
                MockProducer.return_value = producer_return
            else:
                MockProducer.return_value = mock.MagicMock()
            mock_observer = mock.MagicMock()
            MockObserver.return_value = mock_observer
            mock_sleep.side_effect = KeyboardInterrupt
            yield {
                "mock_gen": mock_gen,
                "MockProducer": MockProducer,
                "MockObserver": MockObserver,
                "mock_observer": mock_observer,
                "mock_sleep": mock_sleep,
            }

    return _setup


def _make_scanned_file(
    filepath: str,
    absolute_path: str | None = None,
    filename: str | None = None,
    filetype: str | None = None,
) -> ScannedFile:
    """Helper to create a ScannedFile for testing."""
    if absolute_path is None:
        absolute_path = f"/data/{filepath}"
    if filename is None:
        filename = Path(filepath).name
    if filetype is None:
        filetype = Path(filepath).suffix.lstrip(".")
    return ScannedFile(
        filepath=filepath,
        absolute_path=absolute_path,
        filename=filename,
        filetype=filetype,
    )


def _make_valid_metadata(
    filepath: str = "videos/lecture.mp4",
    filename: str = "lecture.mp4",
    filetype: str = "mp4",
    error: str | None = None,
) -> dict:
    """Create a valid metadata dictionary for testing.

    Creates metadata with all applicable step maps set to "complete".
    If ``error`` is provided, the error field is set and step maps are
    not included (making the file not feed-eligible).
    """
    meta: dict = {
        "title": "Test Title",
        "description": "Test description",
        "filename": filename,
        "filetype": filetype,
        "date": 1728931200,
        "size": 12345,
    }
    if error:
        meta["error"] = error
    else:
        if filetype == "mp4":
            meta["transcribe"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
            meta["summarize"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
            meta["content"] = "Test transcript"
            meta["summary"] = "Test summary"
        else:
            meta["extract"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
            meta["summarize"] = {
                "status": "complete",
                "input_size": 0,
                "output_size": 0,
            }
    return meta


def _make_mock_metadata(feed_eligible: bool = True):
    """Create a mock Metadata object for testing."""
    mock_obj = mock.MagicMock()
    mock_obj.feed_eligible = feed_eligible
    return mock_obj


# ---------------------------------------------------------------------------
# T5.1: CLI Framework Tests
# ---------------------------------------------------------------------------


class TestFeedsCommand:
    """Tests for the feeds command (T5.4)."""

    def test_finalization_preserves_step_failure(self, valid_env, tmp_data_dir):
        """Finalization must not erase a step-map failure without top-level error."""
        from focus_feed.cli import _finalize_incomplete_json

        source = tmp_data_dir / "failed.mp4"
        source.write_bytes(b"video")
        metadata = {
            "title": {"filename": "failed", "extracted": None, "generated": None},
            "filename": "failed.mp4",
            "filetype": "mp4",
            "transcribe": {"status": "failed", "error": "transcription failed"},
            "summarize": {"status": None},
        }
        json_path = tmp_data_dir / "failed.mp4.json"
        original = json.dumps(metadata)
        json_path.write_text(original)

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            success_count, error_count = _finalize_incomplete_json(tmp_data_dir)

        assert (success_count, error_count) == (0, 0)
        assert json_path.read_text() == original
        MockSummarizer.return_value.generate_title.assert_not_called()

    def test_finalization_preserves_schema_fields_for_audio_media(self, valid_env, tmp_data_dir):
        """Finalization handles non-MP4 media and retains existing metadata fields."""
        from focus_feed.cli import _finalize_incomplete_json

        source = tmp_data_dir / "lecture.m4a"
        source.write_bytes(b"audio")
        metadata = {
            "title": {"filename": "lecture", "extracted": None, "generated": "Lecture"},
            "filename": "lecture.m4a",
            "filetype": "m4a",
            "date": 1728931200.123,
            "hash": "abc123",
            "publication_date": 1728000000.456,
            "duration": 45.5,
            "creation_time": "2024-01-15T10:30:00Z",
            "content": "Existing transcript",
            "summary": "Existing summary",
            "transcribe": {"status": "complete"},
            "summarize": {"status": "in_progress"},
        }
        json_path = tmp_data_dir / "lecture.m4a.json"
        json_path.write_text(json.dumps(metadata))

        success_count, error_count = _finalize_incomplete_json(tmp_data_dir)

        assert (success_count, error_count) == (1, 0)
        finalized = json.loads(json_path.read_text())
        for field in (
            "hash",
            "publication_date",
            "duration",
            "creation_time",
            "content",
            "summary",
        ):
            assert finalized[field] == metadata[field]
        assert finalized["transcribe"]["status"] == "complete"
        assert finalized["summarize"]["status"] == "complete"

    def test_feeds_counts_audio_media(self, runner, valid_env, tmp_data_dir):
        """The feeds command counts eligible non-MP4 media as podcasts."""
        (tmp_data_dir / "lecture.m4a").write_bytes(b"audio")
        metadata = _make_valid_metadata(
            filepath="lecture.m4a", filename="lecture.m4a", filetype="m4a"
        )
        metadata["transcribe"] = {"status": "complete"}
        metadata["summarize"] = {"status": "complete"}
        metadata["content"] = "Test transcript"
        metadata["summary"] = "Test summary"
        (tmp_data_dir / "lecture.m4a.json").write_text(json.dumps(metadata))

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            MockGen.return_value.generate_all_feeds.return_value = None
            result = runner.invoke(app, ["feeds", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert "Podcast feed: 1 item(s)" in result.output

    def test_generate_feeds_counts_audio_media(self, valid_env, tmp_data_dir):
        """The shared feed count helper should count all eligible media types."""
        from focus_feed.cli import _generate_feeds

        (tmp_data_dir / "lecture.m4a").write_bytes(b"audio")
        metadata = _make_valid_metadata(
            filepath="lecture.m4a", filename="lecture.m4a", filetype="m4a"
        )
        metadata["transcribe"] = {"status": "complete"}
        metadata["summarize"] = {"status": "complete"}
        metadata["content"] = "Test transcript"
        metadata["summary"] = "Test summary"
        (tmp_data_dir / "lecture.m4a.json").write_text(json.dumps(metadata))

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            MockGen.return_value.generate_all_feeds.return_value = None
            counts = _generate_feeds(tmp_data_dir)

        assert counts == {"podcast": 1, "articles": 0}

    def test_finalize_incomplete_json_does_not_touch_sibling_directories(self, valid_env, tmp_path):
        """A C10 path should not finalize similarly prefixed sibling directories."""
        from focus_feed.cli import _finalize_incomplete_json

        data_dir = tmp_path / "data"
        target_dir = data_dir / "C10"
        sibling_one = data_dir / "C101"
        sibling_two = data_dir / "C102"
        for directory in (target_dir, sibling_one, sibling_two):
            directory.mkdir(parents=True)
            source = directory / "notes.pdf"
            source.write_bytes(b"pdf data")
            metadata = {
                "title": "Notes",
                "filename": "notes.pdf",
                "filetype": "pdf",
                "description": "",
                "content": "Extracted notes",
                "extract": {"status": "in_progress"},
            }
            (directory / "notes.pdf.json").write_text(json.dumps(metadata))

        sibling_paths = [directory / "notes.pdf.json" for directory in (sibling_one, sibling_two)]
        originals = [path.read_text() for path in sibling_paths]
        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.summarize.return_value = ("Summary", 10, 5)
            mock_summarizer.generate_title.return_value = ("Generated Notes", 10, 5)
            MockSummarizer.return_value = mock_summarizer

            success_count, error_count = _finalize_incomplete_json(data_dir, paths=[target_dir])

        assert (success_count, error_count) == (1, 0)
        target_metadata = json.loads((target_dir / "notes.pdf.json").read_text())
        assert target_metadata["extract"]["status"] == "complete"
        assert [path.read_text() for path in sibling_paths] == originals
        mock_summarizer.summarize.assert_called_once()

    def test_feeds_skips_processing(self, runner, valid_env, tmp_data_dir):
        """Feeds should not process or modify files/metadata."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        json_path = tmp_data_dir / "doc.pdf.json"
        original = json.dumps(
            _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        )
        json_path.write_text(original)

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        mock_generator.generate_all_feeds.assert_called_once()
        assert json_path.read_text() == original
        assert sorted(path.name for path in tmp_data_dir.glob("*.json")) == ["doc.pdf.json"]

    def test_feeds_reads_existing_json(self, runner, valid_env, tmp_data_dir):
        """Feeds should read existing JSON metadata files."""
        # Create a valid MP4 metadata file
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video")
        (tmp_data_dir / "lecture.mp4.json").write_text(json.dumps(_make_valid_metadata()))
        # Create a valid PDF metadata file
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        pdf_metadata = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        pdf_metadata["summary"] = "Test article summary"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(pdf_metadata))

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "Podcast feed: 1 item(s)" in result.output
        assert "Articles feed: 1 item(s)" in result.output

    def test_feeds_empty_directory(self, runner, valid_env, tmp_data_dir):
        """Feeds should handle empty directory."""
        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert "0 item(s)" in result.output

    def test_feeds_with_config_file(self, runner, valid_env, tmp_data_dir, tmp_path):
        """Feeds should accept --config-file option."""
        config_file = tmp_path / "custom.env"
        config_file.write_text(
            "BASE_URL=https://custom.example.com\n"
            "LLM_ENDPOINT=https://llm.example.com/v1\n"
            "LLM_API_KEY=test-key\n"
            "LLM_MODEL=test-model\n"
            "PROMPT_ARTICLE=Test article prompt\n"
            "PROMPT_ARTICLE_REFLECTION=Test article reflection prompt\n"
            "PROMPT_PODCAST=Test podcast prompt\n"
            "PROMPT_PODCAST_REFLECTION=Test podcast reflection prompt\n"
            "PROMPT_TITLE=Test title prompt\n"
        )

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                [
                    "feeds",
                    str(tmp_data_dir),
                    "--config-file",
                    str(config_file),
                ],
            )

        assert result.exit_code == 0

    def test_feeds_does_not_count_error_metadata(self, runner, valid_env, tmp_data_dir):
        """Feeds should not count error metadata in the summary."""
        # Success metadata
        (tmp_data_dir / "good.mp4").write_bytes(b"video")
        (tmp_data_dir / "good.mp4.json").write_text(
            json.dumps(
                _make_valid_metadata(filepath="good.mp4", filename="good.mp4", filetype="mp4")
            )
        )
        # Error metadata
        (tmp_data_dir / "bad.mp4").write_bytes(b"video")
        error_meta = _make_valid_metadata(
            filepath="bad.mp4", filename="bad.mp4", filetype="mp4", error="Processing failed"
        )
        (tmp_data_dir / "bad.mp4.json").write_text(json.dumps(error_meta))

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        # Only 1 podcast item (the success one)
        assert "1 item(s)" in result.output

    def test_feeds_does_not_finalize_incomplete_metadata(self, runner, valid_env, tmp_data_dir):
        """Feeds should leave incomplete metadata untouched and not count it."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        json_path = tmp_data_dir / "lecture.mp4.json"
        original = json.dumps(
            {
                "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                "summarize": {"status": None, "input_size": 0, "output_size": 0},
                "title": "Test",
                "description": "",
                "filename": "lecture.mp4",
                "filepath": "lecture.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 1024,
                "content": "Some transcript content",
                "summary": "Some summary",
            }
        )
        json_path.write_text(original)

        with (
            mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer,
            mock.patch("focus_feed.feeds.FeedGenerator") as MockGen,
        ):
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator
            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        assert json_path.read_text() == original
        assert "Podcast feed: 0 item(s)" in result.output
        MockSummarizer.assert_not_called()

    def test_feeds_skips_finalization_for_complete_files(self, runner, valid_env, tmp_data_dir):
        """Feeds should leave already complete metadata byte-identical."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf data")
        # Already complete metadata (extract + summarize both complete)
        json_path = tmp_data_dir / "doc.pdf.json"
        original = json.dumps(
            {
                "extract": {"status": "complete", "input_size": 0, "output_size": 0},
                "summarize": {"status": "complete", "input_size": 0, "output_size": 0},
                "title": "Existing Title",
                "description": "Existing Description",
                "summary": "Existing summary",
                "filename": "doc.pdf",
                "filepath": "doc.pdf",
                "filetype": "pdf",
                "date": 1728931200,
                "size": 1024,
            }
        )
        json_path.write_text(original)

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            mock_generator = mock.MagicMock()
            mock_generator.generate_all_feeds.return_value = None
            MockGen.return_value = mock_generator

            result = runner.invoke(
                app,
                ["feeds", str(tmp_data_dir)],
            )

        assert result.exit_code == 0
        # File should still be valid and feed-eligible
        data = json.loads(json_path.read_text())
        assert data["extract"]["status"] == "complete"
        assert data["summarize"]["status"] == "complete"
        assert json_path.read_text() == original

    def test_feeds_does_not_generate_title_for_mp4_without_title(
        self, runner, valid_env, tmp_data_dir
    ):
        """Feeds should not generate missing titles or modify the sidecar."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        json_path = tmp_data_dir / "lecture.mp4.json"
        original = json.dumps(
            {
                "transcribe": {"status": "complete", "input_size": 0, "output_size": 0},
                "summarize": {"status": None, "input_size": 0, "output_size": 0},
                "title": "",
                "description": "",
                "filename": "lecture.mp4",
                "filepath": "lecture.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 1024,
                "content": "Some transcript content",
                "summary": "Some summary",
            }
        )
        json_path.write_text(original)

        with mock.patch("focus_feed.services.summarizer.Summarizer") as MockSummarizer:
            mock_summarizer = mock.MagicMock()
            mock_summarizer.generate_title.return_value = ("Generated Title", 30, 5)
            MockSummarizer.return_value = mock_summarizer

            with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
                mock_generator = mock.MagicMock()
                mock_generator.generate_all_feeds.return_value = None
                MockGen.return_value = mock_generator

                result = runner.invoke(
                    app,
                    ["feeds", str(tmp_data_dir)],
                )

        assert result.exit_code == 0
        assert json_path.read_text() == original
        mock_summarizer.generate_title.assert_not_called()

    def test_feeds_skips_item_without_json_sidecar(self, runner, valid_env, tmp_data_dir):
        """A source without a sidecar is reported and left untouched."""
        source = tmp_data_dir / "missing.mp4"
        source.write_bytes(b"video data")

        result = runner.invoke(app, ["feeds", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert "no JSON metadata sidecar" in result.output
        assert "missing.mp4" in result.output
        assert not source.with_name(source.name + ".json").exists()

    def test_feeds_skips_item_without_summary(self, runner, valid_env, tmp_data_dir):
        """A sidecar without a summary is reported and remains unchanged."""
        (tmp_data_dir / "nosummary.pdf").write_bytes(b"pdf data")
        json_path = tmp_data_dir / "nosummary.pdf.json"
        original = json.dumps(_make_valid_metadata("nosummary.pdf", "nosummary.pdf", "pdf"))
        json_path.write_text(original)

        result = runner.invoke(app, ["feeds", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert "no summary" in result.output
        assert "nosummary.pdf" in result.output
        assert json_path.read_text() == original

    def test_feeds_skips_item_with_unparseable_json(self, runner, valid_env, tmp_data_dir):
        """An invalid JSON sidecar is reported and remains unchanged."""
        (tmp_data_dir / "invalid.mp4").write_bytes(b"video data")
        json_path = tmp_data_dir / "invalid.mp4.json"
        original = "{invalid json"
        json_path.write_text(original)

        result = runner.invoke(app, ["feeds", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert "JSON metadata could not be parsed" in result.output
        assert "invalid.mp4" in result.output
        assert json_path.read_text() == original

    def test_feeds_does_not_modify_json_sidecars(self, runner, valid_env, tmp_data_dir):
        """Real feed generation should not persist a computed source hash."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        json_path = tmp_data_dir / "lecture.mp4.json"
        original = json.dumps(_make_valid_metadata("lecture.mp4", "lecture.mp4", "mp4"))
        json_path.write_text(original)
        (tmp_data_dir / "feed.json").write_text(
            json.dumps({"title": "Test Feed", "description": "Description"})
        )

        result = runner.invoke(app, ["feeds", str(tmp_data_dir)])

        assert result.exit_code == 0, result.output
        assert json_path.read_text() == original

    def test_feeds_includes_item_with_summary(self, runner, valid_env, tmp_data_dir):
        """A summarized, feed-eligible MP4 is included in the item count."""
        (tmp_data_dir / "lecture.mp4").write_bytes(b"video data")
        metadata = _make_valid_metadata("lecture.mp4", "lecture.mp4", "mp4")
        metadata["summary"] = "A non-empty summary"
        (tmp_data_dir / "lecture.mp4.json").write_text(json.dumps(metadata))

        with mock.patch("focus_feed.feeds.FeedGenerator") as MockGen:
            MockGen.return_value.generate_all_feeds.return_value = None
            result = runner.invoke(app, ["feeds", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert "Podcast feed: 1 item(s)" in result.output
        assert "lecture.mp4" not in result.output

    def test_feeds_scoped_to_requested_paths(self, runner, valid_env, tmp_data_dir):
        """Only source files under the requested path are counted."""
        for folder in ("first", "second"):
            directory = tmp_data_dir / folder
            directory.mkdir()
            (directory / "lecture.mp4").write_bytes(b"video data")
            metadata = _make_valid_metadata(
                filepath="lecture.mp4", filename="lecture.mp4", filetype="mp4"
            )
            metadata["summary"] = f"{folder} summary"
            (directory / "lecture.mp4.json").write_text(json.dumps(metadata))

        result = runner.invoke(app, ["feeds", str(tmp_data_dir / "first")])

        assert result.exit_code == 0
        assert "Podcast feed: 1 item(s)" in result.output
        assert "no JSON metadata" not in result.output
        assert "no summary" not in result.output
