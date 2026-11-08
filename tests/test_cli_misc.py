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
            meta["content"] = "Extracted content"
            meta["summary"] = "Summary text"
    return meta


def _make_mock_metadata(feed_eligible: bool = True):
    """Create a mock Metadata object for testing."""
    mock_obj = mock.MagicMock()
    mock_obj.feed_eligible = feed_eligible
    return mock_obj


# ---------------------------------------------------------------------------
# T5.1: CLI Framework Tests
# ---------------------------------------------------------------------------


class TestCreateFeedCommand:
    """Tests for the create-feed command."""

    def test_create_feed_help(self, runner):
        """The create-feed command should have help text."""
        result = runner.invoke(app, ["create-feed", "--help"])
        assert result.exit_code == 0
        assert "DIRECTORY" in result.output

    def test_create_feed_creates_feed_json(self, runner, tmp_path):
        """create-feed should create a feed.json file with template content."""
        target_dir = tmp_path / "my-course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        feed_json_path = target_dir / "feed.json"
        assert feed_json_path.exists()

        data = json.loads(feed_json_path.read_text(encoding="utf-8"))
        assert data["title"] == "Feed Title"
        assert data["description"] == "Feed description"

    def test_create_feed_no_link_field(self, runner, tmp_path):
        """The generated feed.json should NOT contain a link field."""
        target_dir = tmp_path / "my-course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert "link" not in data

    def test_create_feed_creates_directory_if_missing(self, runner, tmp_path):
        """create-feed should create the target directory if it doesn't exist."""
        target_dir = tmp_path / "nested" / "deep" / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        assert (target_dir / "feed.json").exists()

    def test_create_feed_overwrites_existing(self, runner, tmp_path):
        """create-feed should overwrite an existing feed.json."""
        target_dir = tmp_path / "course"
        target_dir.mkdir()
        (target_dir / "feed.json").write_text(
            json.dumps({"title": "Old Title", "description": "Old Desc"}),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert data["title"] == "Feed Title"
        assert data["description"] == "Feed description"

    def test_create_feed_output_message(self, runner, tmp_path):
        """create-feed should print a confirmation message with the path."""
        target_dir = tmp_path / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        assert "Created feed.json" in result.output
        assert str(target_dir / "feed.json") in result.output

    def test_create_feed_includes_language_and_author(self, runner, tmp_path):
        """create-feed should include language and author fields in the template."""
        target_dir = tmp_path / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert data["language"] == "en-us"
        assert data["author"] == "Feed Author"

    def test_create_feed_includes_image(self, runner, tmp_path):
        """create-feed should include an image dict in the template."""
        target_dir = tmp_path / "course"
        result = runner.invoke(
            app,
            ["create-feed", str(target_dir)],
        )

        assert result.exit_code == 0
        data = json.loads((target_dir / "feed.json").read_text(encoding="utf-8"))
        assert "image" in data
        assert data["image"]["url"] == "https://www.example.com/logo.jpg"
        assert data["image"]["title"] == "Logo"
        assert data["image"]["link"] == "https://www.example.com/"


class TestCleanupCommand:
    """Tests for the cleanup command."""

    def test_cleanup_help(self, runner):
        """The cleanup command should have help text."""
        result = runner.invoke(app, ["cleanup", "--help"])
        assert result.exit_code == 0
        assert "--config-file" in result.output
        assert "PATHS" in result.output

    def test_cleanup_requires_at_least_one_path(self, runner, valid_env):
        """Cleanup should fail with a usage error when no paths are given."""
        result = runner.invoke(app, ["cleanup"])
        assert result.exit_code != 0
        assert "At least one file or directory path is required." in result.output

    def test_cleanup_no_json_files(self, runner, valid_env, tmp_data_dir):
        """Cleanup should report no JSON files found in an empty directory."""
        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "No JSON metadata files found." in result.output

    def test_cleanup_deletes_sidecar_without_summary_or_content(
        self, runner, valid_env, tmp_data_dir
    ):
        """Cleanup should delete metadata sidecars with no extracted or summarized output."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta.pop("summary")
        meta.pop("content")
        sidecar = tmp_data_dir / "doc.pdf.json"
        sidecar.write_text(json.dumps(meta))

        result = runner.invoke(app, ["cleanup", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert not sidecar.exists()
        assert "doc.pdf.json: deleted (no summary or content)" in result.output
        assert "deleted 1 file(s)" in result.output

    def test_cleanup_keeps_sidecar_with_summary(self, runner, valid_env, tmp_data_dir):
        """Cleanup should preserve a sidecar with a non-empty summary."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta.pop("content")
        sidecar = tmp_data_dir / "doc.pdf.json"
        sidecar.write_text(json.dumps(meta))

        result = runner.invoke(app, ["cleanup", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert sidecar.exists()

    def test_cleanup_keeps_sidecar_with_content(self, runner, valid_env, tmp_data_dir):
        """Cleanup should preserve a sidecar with non-empty extracted content."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta.pop("summary")
        sidecar = tmp_data_dir / "doc.pdf.json"
        sidecar.write_text(json.dumps(meta))

        result = runner.invoke(app, ["cleanup", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert sidecar.exists()

    def test_cleanup_deletes_sidecar_with_empty_summary_and_content(
        self, runner, valid_env, tmp_data_dir
    ):
        """Cleanup should treat empty output strings as missing output."""
        (tmp_data_dir / "doc.pdf").write_bytes(b"pdf")
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["summary"] = ""
        meta["content"] = ""
        sidecar = tmp_data_dir / "doc.pdf.json"
        sidecar.write_text(json.dumps(meta))

        result = runner.invoke(app, ["cleanup", str(tmp_data_dir)])

        assert result.exit_code == 0
        assert not sidecar.exists()
        assert "deleted 1 file(s)" in result.output

    def test_cleanup_removes_unknown_top_level_key(self, runner, valid_env, tmp_data_dir):
        """Cleanup should remove unknown top-level keys from JSON metadata."""
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["unknown_field"] = "should be removed"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "unknown_field" in result.output
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert "unknown_field" not in data

    def test_cleanup_removes_unknown_title_keys(self, runner, valid_env, tmp_data_dir):
        """Cleanup should remove unknown keys from the title dict."""
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["title"] = {
            "filename": "doc",
            "extracted": "Doc Title",
            "generated": "Generated Title",
            "extra_title_key": "should be removed",
        }
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "title.extra_title_key" in result.output
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert "extra_title_key" not in data["title"]
        # Known keys should be preserved
        assert data["title"]["filename"] == "doc"
        assert data["title"]["extracted"] == "Doc Title"
        assert data["title"]["generated"] == "Generated Title"

    def test_cleanup_removes_unknown_step_map_keys(self, runner, valid_env, tmp_data_dir):
        """Cleanup should remove unknown keys from step-map dicts."""
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["extract"]["extra_step_key"] = "should be removed"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "extract.extra_step_key" in result.output
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert "extra_step_key" not in data["extract"]

    def test_cleanup_preserves_valid_metadata(self, runner, valid_env, tmp_data_dir):
        """Cleanup should not modify files with no unknown keys."""
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        original_content = (tmp_data_dir / "doc.pdf.json").read_text()

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "modified 0 file(s)" in result.output
        # File content should be unchanged
        assert (tmp_data_dir / "doc.pdf.json").read_text() == original_content

    def test_cleanup_does_not_modify_feed_json(self, runner, valid_env, tmp_data_dir):
        """Cleanup should not modify feed.json files."""
        (tmp_data_dir / "feed.json").write_text(
            json.dumps({"title": "My Feed", "unknown_key": "should stay"})
        )

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        # feed.json should still have the unknown key
        data = json.loads((tmp_data_dir / "feed.json").read_text())
        assert "unknown_key" in data

    def test_cleanup_handles_invalid_json_gracefully(self, runner, valid_env, tmp_data_dir):
        """Cleanup should handle invalid JSON files without crashing."""
        (tmp_data_dir / "bad.pdf.json").write_text("{invalid json content")

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        # Should report 0 files modified
        assert "modified 0 file(s)" in result.output

    def test_cleanup_handles_string_title(self, runner, valid_env, tmp_data_dir):
        """Cleanup should not modify files with string titles (backward compat)."""
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["title"] = "String Title"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "modified 0 file(s)" in result.output
        data = json.loads((tmp_data_dir / "doc.pdf.json").read_text())
        assert data["title"] == "String Title"

    def test_cleanup_summary_output(self, runner, valid_env, tmp_data_dir):
        """Cleanup should print a completion summary with counts."""
        meta = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        meta["unknown1"] = "val1"
        meta["unknown2"] = "val2"
        (tmp_data_dir / "doc.pdf.json").write_text(json.dumps(meta))

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "checked 1 file(s)" in result.output
        assert "modified 1 file(s)" in result.output
        assert "removed 2 key(s)" in result.output

    def test_cleanup_with_config_file(self, runner, valid_env, tmp_data_dir, tmp_path):
        """Cleanup should accept --config-file option."""
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

        (tmp_data_dir / "doc.pdf.json").write_text(
            json.dumps(_make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf"))
        )

        result = runner.invoke(
            app,
            [
                "cleanup",
                str(tmp_data_dir),
                "--config-file",
                str(config_file),
            ],
        )
        assert result.exit_code == 0

    def test_cleanup_multiple_files(self, runner, valid_env, tmp_data_dir):
        """Cleanup should process multiple JSON files."""
        # File 1: has unknown key
        meta1 = _make_valid_metadata(filepath="doc1.pdf", filename="doc1.pdf", filetype="pdf")
        meta1["unknown_key"] = "remove me"
        (tmp_data_dir / "doc1.pdf.json").write_text(json.dumps(meta1))

        # File 2: clean (no unknown keys)
        meta2 = _make_valid_metadata(filepath="doc2.pdf", filename="doc2.pdf", filetype="pdf")
        (tmp_data_dir / "doc2.pdf.json").write_text(json.dumps(meta2))

        result = runner.invoke(
            app,
            ["cleanup", str(tmp_data_dir)],
        )
        assert result.exit_code == 0
        assert "checked 2 file(s)" in result.output
        assert "modified 1 file(s)" in result.output

    def test_cleanup_nonexistent_path(self, runner, valid_env, tmp_path):
        """Cleanup should fail for a nonexistent path."""
        result = runner.invoke(
            app,
            ["cleanup", str(tmp_path / "nonexistent")],
        )
        assert result.exit_code != 0
        assert "Path does not exist" in result.output


class TestFindUnknownKeys:
    """Tests for the _find_unknown_keys helper function."""

    def test_no_unknown_keys(self):
        """Should return empty list for valid metadata."""
        from focus_feed.cli import _find_unknown_keys

        data = _make_valid_metadata()
        assert _find_unknown_keys(data) == []

    def test_unknown_top_level_key(self):
        """Should detect unknown top-level keys."""
        from focus_feed.cli import _find_unknown_keys

        data = _make_valid_metadata()
        data["unknown_field"] = "value"
        result = _find_unknown_keys(data)
        assert "unknown_field" in result

    def test_unknown_title_key(self):
        """Should detect unknown keys in title dict."""
        from focus_feed.cli import _find_unknown_keys

        data = _make_valid_metadata()
        data["title"] = {
            "filename": "doc",
            "extracted": "Title",
            "generated": "Gen",
            "extra": "bad",
        }
        result = _find_unknown_keys(data)
        assert "title.extra" in result

    def test_unknown_step_map_key(self):
        """Should detect unknown keys in step-map dicts."""
        from focus_feed.cli import _find_unknown_keys

        data = _make_valid_metadata()
        data["transcribe"]["extra_step"] = "bad"
        result = _find_unknown_keys(data)
        assert "transcribe.extra_step" in result

    def test_string_title_not_flagged(self):
        """Should not flag string titles (backward compatibility)."""
        from focus_feed.cli import _find_unknown_keys

        data = _make_valid_metadata()
        data["title"] = "String Title"
        assert _find_unknown_keys(data) == []

    def test_multiple_unknown_keys(self):
        """Should detect all unknown keys across all levels."""
        from focus_feed.cli import _find_unknown_keys

        data = _make_valid_metadata(filepath="doc.pdf", filename="doc.pdf", filetype="pdf")
        data["unknown_top"] = "val"
        data["title"] = {"filename": "doc", "extracted": "T", "generated": "G", "bad_title": "x"}
        data["extract"]["bad_step"] = "x"
        data["summarize"]["bad_step2"] = "x"
        result = _find_unknown_keys(data)
        assert "unknown_top" in result
        assert "title.bad_title" in result
        assert "extract.bad_step" in result
        assert "summarize.bad_step2" in result
        assert len(result) == 4
