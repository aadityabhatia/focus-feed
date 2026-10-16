"""Tests for the feed generator module.

Tests cover:
    - FeedGenerator base class: metadata discovery, error filtering, sorting,
      URL calculation, orphaned JSON handling.
    - Podcast feed generation: enclosure tags, item fields, RSS 2.0 + podcast
      extension validity.
    - Articles feed generation: item fields, RSS 2.0 validity.
    - generate_all_feeds(): error isolation between feeds, idempotency.
    - XML escaping of special characters.
    - Feed validation (T7.2): validate_feed(), validate_output(),
      FeedValidationResult, and optional validation in generate_all_feeds().
"""

import json
import os
from pathlib import Path
from unittest import mock

from local_feed_generator.feeds import (
    FeedGenerator,
    FeedValidationResult,
    _markdown_to_html,
    validate_feed,
)
from local_feed_generator.metadata import Metadata, StepMap

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_metadata(
    filepath: str,
    filetype: str = "mp4",
    title: str | None = None,
    description: str = "Test description",
    summary: str | None = None,
    transcript: str | None = None,
    size: int = 1000,
    feed_eligible: bool = True,
    error: str | None = None,
    date: int = 1728931200,
) -> Metadata:
    """Create a Metadata object for testing.

    Args:
        filepath: Relative path from data directory root.
        filetype: One of 'mp4', 'pdf', 'html'.
        title: Title string (defaults to filepath stem).
        description: Description string.
        summary: Summary string (for MP4 only).
        transcript: Transcript string (for MP4 only).
        size: File size in bytes.
        feed_eligible: If True, set status to "complete" for all applicable steps.
            If False, leave status as None.
        error: Error message (if provided, feed_eligible is set to False).
        date: UNIX epoch date (seconds).

    Returns:
        A validated Metadata instance.
    """
    filename = Path(filepath).name
    kwargs = {
        "title": title or filename,
        "filename": filename,
        "filepath": filepath,
        "filetype": filetype,
        "date": date,
        "size": size,
    }
    if filetype == "mp4":
        # For MP4, transcript maps to content, summary holds the LLM key
        # takeaways, and description is left blank.
        kwargs["content"] = transcript
        kwargs["description"] = ""  # MP4: description is blank
        kwargs["summary"] = summary  # MP4: summary holds LLM key takeaways
        if feed_eligible and not error:
            kwargs["transcribe"] = StepMap(
                status="complete",
                input_size=size,
                output_size=len(transcript.encode("utf-8")) if transcript else 0,
            )
            kwargs["summarize"] = StepMap(
                status="complete",
                input_size=len(transcript.encode("utf-8")) if transcript else 0,
                output_size=(
                    len((summary or description).encode("utf-8")) if (summary or description) else 0
                ),
            )
    else:
        kwargs["description"] = description
        if feed_eligible and not error:
            kwargs["extract"] = StepMap(
                status="complete",
                input_size=size,
                output_size=len(description.encode("utf-8")) if description else 0,
            )
    if error:
        kwargs["error"] = error
    return Metadata(**kwargs)


def write_metadata_file(
    data_dir: Path,
    filepath: str,
    **metadata_kwargs,
) -> Path:
    """Write a JSON metadata file to the data directory.

    Also creates a dummy source file so the JSON is not considered orphaned.

    Args:
        data_dir: Root data directory path.
        filepath: Relative path for the source file (e.g. 'videos/lecture.mp4').
        **metadata_kwargs: Additional keyword arguments passed to make_metadata.

    Returns:
        Path to the written JSON file.
    """
    filepath_obj = Path(filepath)
    filetype = filepath_obj.suffix.lower().lstrip(".")
    # Allow caller to override filetype, otherwise derive from filepath extension
    if "filetype" not in metadata_kwargs:
        metadata_kwargs["filetype"] = filetype
    metadata = make_metadata(filepath=filepath, **metadata_kwargs)

    # Create source file so JSON is not orphaned.
    source_path = data_dir / filepath
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if filetype == "html":
        source_path.write_text("<html></html>", encoding="utf-8")
    else:
        source_path.write_bytes(b"file data")

    # Write JSON metadata.
    json_path = data_dir / (filepath + ".json")
    json_path.write_text(metadata.to_json(), encoding="utf-8")
    return json_path


def parse_rss_file(path: Path) -> str:
    """Read and return the content of an RSS feed file."""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# FeedGenerator construction
# ---------------------------------------------------------------------------


class TestFeedGeneratorConstruction:
    """Tests for FeedGenerator initialization."""

    def test_init_with_explicit_args(self, tmp_path):
        """FeedGenerator should accept explicit data_dir and base_url."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        assert fg.data_dir == tmp_path.resolve()
        assert fg.base_url == "https://example.com/feeds"

    def test_init_with_string_args(self, tmp_path):
        """FeedGenerator should accept string paths."""
        fg = FeedGenerator(data_dir=str(tmp_path), base_url="https://example.com")
        assert fg.data_dir == tmp_path.resolve()
        assert fg.base_url == "https://example.com"

    def test_init_uses_config_defaults(self, tmp_path):
        """FeedGenerator should use config values when args are None."""
        env = {
            "BASE_URL": "https://config.example.com/feeds",
            "DATA_DIR": str(tmp_path),
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_SYSTEM_PROMPT": "Test prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            from local_feed_generator.config import reload_config

            reload_config()
            fg = FeedGenerator()
            assert fg.base_url == "https://config.example.com/feeds"
            assert fg.data_dir == tmp_path.resolve()

    def test_base_url_strips_trailing_slash(self, tmp_path):
        """Trailing slash should be stripped from base_url."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds/")
        assert fg.base_url == "https://example.com/feeds"

    def test_data_dir_is_absolute(self, tmp_path):
        """data_dir should be stored as an absolute path."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        assert fg.data_dir.is_absolute()


# ---------------------------------------------------------------------------
# Metadata discovery and filtering
# ---------------------------------------------------------------------------


class TestMetadataDiscovery:
    """Tests for _discover_metadata_files()."""

    def test_empty_directory_returns_empty(self, tmp_path):
        """An empty data directory should yield no metadata."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()
        assert result == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        """A nonexistent directory should yield no metadata."""
        fg = FeedGenerator(data_dir=tmp_path / "nonexistent", base_url="https://example.com")
        result = fg._discover_metadata_files()
        assert result == []

    def test_filters_error_entries(self, tmp_path):
        """Entries with error set should be excluded."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "videos/bad.mp4", error="Failed")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        assert len(result) == 1
        assert result[0].filepath == "videos/lecture.mp4"

    def test_includes_complete_entries(self, tmp_path):
        """Entries with status='complete' should be included."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        assert len(result) == 2

    def test_discovers_all_filetypes(self, tmp_path):
        """MP4, PDF, and HTML files should all be discovered."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        write_metadata_file(tmp_path, "documents/article.html")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        filetypes = {m.filetype for m in result}
        assert filetypes == {"mp4", "pdf", "html"}

    def test_recursive_discovery(self, tmp_path):
        """JSON files in nested subdirectories should be discovered."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course2/Sub/lesson.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        assert len(result) == 2
        filepaths = {m.filepath for m in result}
        assert "Course1/lecture.mp4" in filepaths
        assert "Course2/Sub/lesson.mp4" in filepaths

    def test_invalid_json_skipped_with_warning(self, tmp_path, caplog):
        """Invalid JSON files should be skipped with a warning."""
        # Create a source file and a corrupt JSON
        source = tmp_path / "bad.mp4"
        source.write_bytes(b"data")
        json_path = tmp_path / "bad.mp4.json"
        json_path.write_text("{invalid json content", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        assert result == []
        assert any(
            "Failed to parse" in record.message
            for record in caplog.records
            if record.levelno >= 30  # WARNING
        )

    def test_orphaned_json_skipped(self, tmp_path):
        """JSON files without a corresponding source file should be skipped."""
        # Create only the JSON, no source file
        json_path = tmp_path / "orphan.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {
                        "status": "complete",
                        "input_size": 1000,
                        "output_size": 500,
                    },
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                    },
                    "title": "Orphan",
                    "description": "desc",
                    "filename": "orphan.mp4",
                    "filepath": "orphan.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        assert result == []


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    """Tests for item sorting (folder name ascending, then filename ascending)."""

    def test_sorted_by_folder_then_filename(self, tmp_path):
        """Items should be sorted by folder name, then filename."""
        # Create files in different folders
        write_metadata_file(tmp_path, "zebra/lecture.mp4")
        write_metadata_file(tmp_path, "alpha/doc.pdf")
        write_metadata_file(tmp_path, "alpha/article.html")
        write_metadata_file(tmp_path, "zebra/notes.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        # Expected order: alpha/article.html, alpha/doc.pdf, zebra/lecture.mp4, zebra/notes.mp4
        filepaths = [m.filepath for m in result]
        assert filepaths == sorted(filepaths, key=lambda p: (Path(p).parent.name, Path(p).name))

    def test_same_folder_sorted_by_filename(self, tmp_path):
        """Within the same folder, items should be sorted by filename."""
        write_metadata_file(tmp_path, "docs/zebra.pdf")
        write_metadata_file(tmp_path, "docs/apple.html")
        write_metadata_file(tmp_path, "docs/mango.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        filenames = [m.filename for m in result]
        assert filenames == ["apple.html", "mango.mp4", "zebra.pdf"]

    def test_root_level_files_sorted_by_filename(self, tmp_path):
        """Files at the root level should be sorted by filename."""
        write_metadata_file(tmp_path, "zebra.mp4")
        write_metadata_file(tmp_path, "apple.pdf")
        write_metadata_file(tmp_path, "mango.html")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        filenames = [m.filename for m in result]
        assert filenames == ["apple.pdf", "mango.html", "zebra.mp4"]

    def test_sort_key_method(self):
        """_sort_key should return (folder_name, filename) tuple."""
        metadata = make_metadata(filepath="videos/lecture.mp4")
        key = FeedGenerator._sort_key(metadata)
        assert key == ("videos", "lecture.mp4")

    def test_sort_key_root_level(self):
        """_sort_key for root-level files should have empty folder name."""
        metadata = make_metadata(filepath="lecture.mp4")
        key = FeedGenerator._sort_key(metadata)
        assert key == ("", "lecture.mp4")


# ---------------------------------------------------------------------------
# URL calculation
# ---------------------------------------------------------------------------


class TestUrlCalculation:
    """Tests for absolute URL calculation."""

    def test_absolute_url(self, tmp_path):
        """Absolute URL should be BASE_URL + filepath."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        url = fg._absolute_url("videos/lecture.mp4")
        assert url == "https://example.com/feeds/videos/lecture.mp4"

    def test_absolute_url_no_trailing_slash(self, tmp_path):
        """Base URL without trailing slash should work correctly."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        url = fg._absolute_url("doc.pdf")
        assert url == "https://example.com/doc.pdf"

    def test_absolute_url_with_subdirectory(self, tmp_path):
        """URLs should preserve subdirectory structure."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        url = fg._absolute_url("Course1/Sub/lesson.mp4")
        assert url == "https://example.com/feeds/Course1/Sub/lesson.mp4"


# ---------------------------------------------------------------------------
# Podcast feed generation
# ---------------------------------------------------------------------------


class TestPodcastFeedGeneration:
    """Tests for generate_podcast_feed()."""

    def test_podcast_feed_file_created(self, tmp_path):
        """podcast.xml should be written to DATA_DIR."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        assert (tmp_path / "podcast.xml").exists()

    def test_podcast_feed_is_valid_rss(self, tmp_path):
        """Generated podcast feed should be valid RSS 2.0 XML."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<rss" in content
        assert 'version="2.0"' in content
        assert "<channel>" in content
        assert "</channel>" in content
        assert "</rss>" in content

    def test_podcast_feed_includes_mp4_items(self, tmp_path):
        """Podcast feed should include all MP4 files."""
        write_metadata_file(tmp_path, "videos/lecture1.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "videos/lecture2.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        assert len(items) == 2
        assert all(m.filetype == "mp4" for m in items)

    def test_podcast_feed_excludes_pdf_and_html(self, tmp_path):
        """Podcast feed should not include PDF or HTML files."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        write_metadata_file(tmp_path, "documents/article.html", filetype="html")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # Should only have one <item>
        assert content.count("<item>") == 1

    def test_podcast_feed_item_has_title(self, tmp_path):
        """Each podcast item should have a title."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", title="My Lecture")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<title>My Lecture</title>" in content

    def test_podcast_feed_item_has_description_from_summary(self, tmp_path):
        """Description should come from summary field."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            summary="Key takeaways from the lecture.",
            description="Fallback description",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "Key takeaways from the lecture." in content

    def test_podcast_feed_item_description_fallback_to_description(self, tmp_path):
        """Description should fall back to description field if summary is None."""
        # Simulate old-format metadata where description held the summary
        # (before the description/summary split was introduced for MP4).
        source = tmp_path / "videos" / "lecture.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"video data")
        json_path = tmp_path / "videos" / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {
                        "status": "complete",
                        "input_size": 1000,
                        "output_size": 500,
                    },
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                    },
                    "title": "lecture.mp4",
                    "description": "Fallback description",
                    "filename": "lecture.mp4",
                    "filepath": "videos/lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "Fallback description" in content

    def test_podcast_feed_item_has_enclosure(self, tmp_path):
        """Each podcast item should have an enclosure with URL, type, and length."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", size=12345678)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<enclosure" in content
        assert 'url="https://example.com/feeds/videos/lecture.mp4"' in content
        assert 'length="12345678"' in content
        assert 'type="audio/mp4"' in content

    def test_podcast_feed_item_has_link(self, tmp_path):
        """Each podcast item should have a link to the source file."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<link>https://example.com/feeds/videos/lecture.mp4</link>" in content

    def test_podcast_feed_item_has_guid(self, tmp_path):
        """Each podcast item should have a GUID based on filepath."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<guid" in content
        assert "https://example.com/feeds/videos/lecture.mp4" in content

    def test_podcast_feed_item_has_pubdate(self, tmp_path):
        """Each podcast item should have a publication date."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", date=1728931200)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<pubDate>" in content

    def test_podcast_feed_items_sorted(self, tmp_path):
        """Podcast feed items should be sorted by folder then filename."""
        write_metadata_file(tmp_path, "zebra/last.mp4")
        write_metadata_file(tmp_path, "alpha/first.mp4")
        write_metadata_file(tmp_path, "alpha/middle.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        filepaths = [m.filepath for m in items]
        assert filepaths == ["alpha/first.mp4", "alpha/middle.mp4", "zebra/last.mp4"]

    def test_podcast_feed_excludes_error_entries(self, tmp_path):
        """Error entries should not appear in the podcast feed."""
        write_metadata_file(tmp_path, "videos/good.mp4")
        write_metadata_file(tmp_path, "videos/bad.mp4", error="Failed")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert content.count("<item>") == 1
        assert "good.mp4" in content
        assert "bad.mp4" not in content

    def test_podcast_feed_has_podcast_extension(self, tmp_path):
        """Podcast feed should include itunes namespace."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "itunes" in content

    def test_podcast_feed_not_created_when_no_mp4(self, tmp_path):
        """Podcast feed should not be created when no MP4 files are present."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        assert items == []
        assert not (tmp_path / "podcast.xml").exists()

    def test_podcast_feed_overwrites_existing(self, tmp_path):
        """Regenerating podcast feed should overwrite existing file."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        # Add another file
        write_metadata_file(tmp_path, "videos/lecture2.mp4")

        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert content.count("<item>") == 2


# ---------------------------------------------------------------------------
# Articles feed generation
# ---------------------------------------------------------------------------


class TestArticlesFeedGeneration:
    """Tests for generate_articles_feed()."""

    def test_articles_feed_file_created(self, tmp_path):
        """articles.xml should be written to DATA_DIR."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        assert (tmp_path / "articles.xml").exists()

    def test_articles_feed_is_valid_rss(self, tmp_path):
        """Generated articles feed should be valid RSS 2.0 XML."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<rss" in content
        assert 'version="2.0"' in content
        assert "<channel>" in content
        assert "</channel>" in content
        assert "</rss>" in content

    def test_articles_feed_includes_pdf_and_html(self, tmp_path):
        """Articles feed should include all PDF and HTML files."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        write_metadata_file(tmp_path, "documents/article.html", filetype="html")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_articles_feed()

        assert len(items) == 2
        filetypes = {m.filetype for m in items}
        assert filetypes == {"pdf", "html"}

    def test_articles_feed_excludes_mp4(self, tmp_path):
        """Articles feed should not include MP4 files."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert content.count("<item>") == 1

    def test_articles_feed_item_has_title(self, tmp_path):
        """Each articles item should have a title."""
        write_metadata_file(tmp_path, "documents/doc.pdf", title="My Document")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<title>My Document</title>" in content

    def test_articles_feed_description_from_description(self, tmp_path):
        """Description should come from the description field for articles."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="The description field value",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "The description field value" in content

    def test_articles_feed_description_empty_when_none(self, tmp_path):
        """Description should be empty when no description is available."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<item>" in content

    def test_articles_feed_item_has_link(self, tmp_path):
        """Each articles item should have a link to the source file."""
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<link>https://example.com/feeds/documents/doc.pdf</link>" in content

    def test_articles_feed_item_has_guid(self, tmp_path):
        """Each articles item should have a GUID based on filepath."""
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<guid" in content
        assert "https://example.com/feeds/documents/doc.pdf" in content

    def test_articles_feed_item_has_pubdate(self, tmp_path):
        """Each articles item should have a publication date."""
        write_metadata_file(tmp_path, "documents/doc.pdf", date=1728931200)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<pubDate>" in content

    def test_articles_feed_items_sorted(self, tmp_path):
        """Articles feed items should be sorted by folder then filename."""
        write_metadata_file(tmp_path, "zebra/last.pdf")
        write_metadata_file(tmp_path, "alpha/first.html")
        write_metadata_file(tmp_path, "alpha/middle.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_articles_feed()

        filepaths = [m.filepath for m in items]
        assert filepaths == ["alpha/first.html", "alpha/middle.pdf", "zebra/last.pdf"]

    def test_articles_feed_excludes_error_entries(self, tmp_path):
        """Error entries should not appear in the articles feed."""
        write_metadata_file(tmp_path, "documents/good.pdf")
        write_metadata_file(tmp_path, "documents/bad.pdf", error="Failed")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert content.count("<item>") == 1
        assert "good.pdf" in content
        assert "bad.pdf" not in content

    def test_articles_feed_not_created_when_no_pdf_html(self, tmp_path):
        """Articles feed should not be created when no PDF/HTML files are present."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_articles_feed()

        assert items == []
        assert not (tmp_path / "articles.xml").exists()

    def test_articles_feed_overwrites_existing(self, tmp_path):
        """Regenerating articles feed should overwrite existing file."""
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        # Add another file
        write_metadata_file(tmp_path, "documents/doc2.html")

        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert content.count("<item>") == 2


# ---------------------------------------------------------------------------
# generate_all_feeds()
# ---------------------------------------------------------------------------


class TestGenerateAllFeeds:
    """Tests for generate_all_feeds()."""

    def test_generates_both_feeds(self, tmp_path):
        """generate_all_feeds should create both podcast.xml and articles.xml when
        root feed.json exists."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()
        assert len(results["podcast"]) == 1
        assert len(results["articles"]) == 1

    def test_returns_dict_with_both_keys(self, tmp_path):
        """Result should have 'podcast' and 'articles' keys."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert "podcast" in results
        assert "articles" in results

    def test_error_in_podcast_does_not_block_articles(self, tmp_path):
        """If podcast generation fails, articles should still be generated."""
        write_metadata_file(tmp_path, "documents/doc.pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        # Mock generate_podcast_feed to raise an exception
        with mock.patch.object(
            fg,
            "generate_podcast_feed",
            side_effect=RuntimeError("Podcast generation failed"),
        ):
            results = fg.generate_all_feeds()

        # Articles feed should still be generated
        assert (tmp_path / "articles.xml").exists()
        assert results["podcast"] == []
        assert len(results["articles"]) == 1

    def test_error_in_articles_does_not_block_podcast(self, tmp_path):
        """If articles generation fails, podcast should still be generated."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        # Mock generate_articles_feed to raise an exception
        with mock.patch.object(
            fg,
            "generate_articles_feed",
            side_effect=RuntimeError("Articles generation failed"),
        ):
            results = fg.generate_all_feeds()

        # Podcast feed should still be generated
        assert (tmp_path / "podcast.xml").exists()
        assert len(results["podcast"]) == 1
        assert results["articles"] == []

    def test_idempotent_regeneration(self, tmp_path):
        """Feeds should be regenerated identically each time."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        fg.generate_all_feeds()
        content1 = parse_rss_file(tmp_path / "podcast.xml")

        fg.generate_all_feeds()
        content2 = parse_rss_file(tmp_path / "podcast.xml")

        # Content should be the same (ignoring lastBuildDate which changes)
        assert content1.replace("lastBuildDate", "") == content2.replace("lastBuildDate", "")


# ---------------------------------------------------------------------------
# XML escaping
# ---------------------------------------------------------------------------


class TestXmlEscaping:
    """Tests for proper XML escaping of special characters."""

    def test_special_characters_in_title_escaped(self, tmp_path):
        """Titles with XML special characters should be properly escaped."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            title="Lecture <with> & \"quotes\" and 'apostrophes'",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # feedgen handles escaping automatically
        assert "<item>" in content
        # The title should be present (escaped)
        assert "quot" in content or "&quot;" in content or "Lecture" in content

    def test_special_characters_in_description_escaped(self, tmp_path):
        """Descriptions with XML special characters should be properly escaped."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            summary="Summary with <tags> & ampersand",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<item>" in content
        # The description should be present (escaped)
        assert "Summary" in content

    def test_special_characters_in_articles_description_escaped(self, tmp_path):
        """Articles descriptions with XML special characters should be escaped."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            description="Description with <html> & special chars",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<item>" in content
        assert "Description" in content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_json_files(self, tmp_path):
        """No JSON files and no feed.json should produce no root-level feeds."""
        (tmp_path / "video.mp4").write_bytes(b"data")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert results["podcast"] == []
        assert results["articles"] == []
        assert not (tmp_path / "podcast.xml").exists()
        assert not (tmp_path / "articles.xml").exists()

    def test_data_dir_not_created_if_missing(self, tmp_path):
        """Feed generator should handle nonexistent data directory gracefully."""
        fg = FeedGenerator(
            data_dir=tmp_path / "nonexistent",
            base_url="https://example.com/feeds",
        )
        results = fg.generate_all_feeds()

        assert results["podcast"] == []
        assert results["articles"] == []

    def test_mixed_status_files(self, tmp_path):
        """Only feed-eligible entries should appear in feeds."""
        write_metadata_file(tmp_path, "videos/good.mp4")
        write_metadata_file(tmp_path, "videos/bad.mp4", error="Failed")
        write_metadata_file(tmp_path, "documents/good.pdf")
        write_metadata_file(tmp_path, "documents/bad.pdf", error="Failed")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert len(results["podcast"]) == 1
        assert len(results["articles"]) == 1

    def test_json_with_invalid_schema_skipped(self, tmp_path, caplog):
        """JSON files that don't match the Metadata schema should be skipped."""
        # Create a source file
        (tmp_path / "bad.mp4").write_bytes(b"data")

        # Write invalid JSON (missing required fields)
        json_path = tmp_path / "bad.mp4.json"
        json_path.write_text(
            json.dumps({"transcribe": {"status": "complete"}}),  # Missing required fields
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        result = fg._discover_metadata_files()

        assert result == []
        assert any(
            "Failed to parse" in record.message for record in caplog.records if record.levelno >= 30
        )

    def test_feed_links_to_self(self, tmp_path):
        """Feed should include a self-referencing link."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert 'rel="self"' in content
        assert "https://example.com/feeds/podcast.xml" in content

    def test_articles_feed_links_to_self(self, tmp_path):
        """Articles feed should include a self-referencing link."""
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert 'rel="self"' in content
        assert "https://example.com/feeds/articles.xml" in content

    def test_feed_has_language(self, tmp_path):
        """Feed should include language element."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<language>en</language>" in content

    def test_feed_has_description(self, tmp_path):
        """Feed should include a description element."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<description>" in content


# ---------------------------------------------------------------------------
# Feed validation (validate_feed function)
# ---------------------------------------------------------------------------

# A minimal valid RSS 2.0 podcast feed with one item.
_VALID_PODCAST_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <link>https://example.com</link>
    <description>Test Podcast Feed</description>
    <item>
      <title>Test Item</title>
      <link>https://example.com/test.mp4</link>
      <description>Test description</description>
      <guid>https://example.com/test.mp4</guid>
      <enclosure url="https://example.com/test.mp4" length="12345" type="audio/mp4"/>
      <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

# A minimal valid RSS 2.0 articles feed with one item.
_VALID_ARTICLES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Articles</title>
    <link>https://example.com</link>
    <description>Test Articles Feed</description>
    <item>
      <title>Test Article</title>
      <link>https://example.com/test.pdf</link>
      <description>Test description</description>
      <guid>https://example.com/test.pdf</guid>
      <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


class TestValidateFeed:
    """Tests for the standalone validate_feed() function."""

    def test_valid_podcast_feed_passes(self):
        """A well-formed podcast feed with all required elements should pass."""
        result = validate_feed(_VALID_PODCAST_XML, feed_name="podcast.xml", is_podcast=True)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_valid_articles_feed_passes(self):
        """A well-formed articles feed with all required elements should pass."""
        result = validate_feed(_VALID_ARTICLES_XML, feed_name="articles.xml", is_podcast=False)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_empty_feed_warns_no_items(self):
        """A feed with no items should produce a warning but stay valid."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://example.com</link>
    <description>Empty</description>
  </channel>
</rss>
"""
        result = validate_feed(xml, feed_name="empty.xml", is_podcast=False)
        assert result.is_valid
        assert len(result.warnings) == 1
        assert "No <item>" in result.warnings[0]

    def test_malformed_xml_fails(self):
        """Malformed XML should produce an error and be marked invalid."""
        bad_xml = "<rss><channel><item><title>Unclosed</item></channel>"
        result = validate_feed(bad_xml, feed_name="bad.xml")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "not well-formed" in result.errors[0]

    def test_non_rss_root_fails(self):
        """A root element other than <rss> should produce an error."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Not RSS</title>
</feed>
"""
        result = validate_feed(xml, feed_name="atom.xml")
        assert not result.is_valid
        assert any("expected <rss>" in e for e in result.errors)

    def test_wrong_rss_version_fails(self):
        """RSS version other than 2.0 should produce an error."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="0.91">
  <channel>
    <title>Old RSS</title>
    <link>https://example.com</link>
    <description>Old</description>
  </channel>
</rss>
"""
        result = validate_feed(xml, feed_name="old.xml")
        assert not result.is_valid
        assert any("expected '2.0'" in e for e in result.errors)

    def test_missing_channel_fails(self):
        """A feed without a <channel> element should produce an error."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
</rss>
"""
        result = validate_feed(xml, feed_name="nochannel.xml")
        assert not result.is_valid
        assert any("Missing <channel>" in e for e in result.errors)

    def test_item_missing_title_fails(self):
        """An item without a <title> should produce an error."""
        xml = _VALID_PODCAST_XML.replace("      <title>Test Item</title>\n", "")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("missing or empty <title>" in e for e in result.errors)

    def test_item_missing_link_fails(self):
        """An item without a <link> should produce an error."""
        xml = _VALID_PODCAST_XML.replace("      <link>https://example.com/test.mp4</link>\n", "")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("missing or empty <link>" in e for e in result.errors)

    def test_item_missing_guid_fails(self):
        """An item without a <guid> should produce an error."""
        xml = _VALID_PODCAST_XML.replace("      <guid>https://example.com/test.mp4</guid>\n", "")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("missing or empty <guid>" in e for e in result.errors)

    def test_item_missing_pubdate_fails(self):
        """An item without <pubDate> or <dc:date> should produce an error."""
        xml = _VALID_PODCAST_XML.replace(
            "      <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>\n", ""
        )
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("missing <pubDate>" in e for e in result.errors)

    def test_item_with_dc_date_passes(self):
        """An item with <dc:date> instead of <pubDate> should pass."""
        xml = _VALID_PODCAST_XML.replace(
            "      <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>\n",
            '      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "2026-10-14T12:00:00Z</dc:date>\n",
        )
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert result.is_valid

    def test_item_empty_title_fails(self):
        """An item with an empty <title> should produce an error."""
        xml = _VALID_PODCAST_XML.replace(
            "      <title>Test Item</title>\n",
            "      <title></title>\n",
        )
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("missing or empty <title>" in e for e in result.errors)

    def test_podcast_item_missing_enclosure_fails(self):
        """A podcast item without an <enclosure> should produce an error."""
        xml = _VALID_PODCAST_XML.replace(
            '      <enclosure url="https://example.com/test.mp4" '
            'length="12345" type="audio/mp4"/>\n',
            "",
        )
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("missing <enclosure>" in e for e in result.errors)

    def test_podcast_enclosure_missing_url_fails(self):
        """An enclosure without a url attribute should produce an error."""
        xml = _VALID_PODCAST_XML.replace(' url="https://example.com/test.mp4"', "")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("enclosure missing 'url'" in e for e in result.errors)

    def test_podcast_enclosure_wrong_type_fails(self):
        """An enclosure with incorrect type should produce an error."""
        xml = _VALID_PODCAST_XML.replace(' type="audio/mp4"', ' type="audio/mpeg"')
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("expected 'audio/mp4'" in e for e in result.errors)

    def test_podcast_enclosure_missing_length_fails(self):
        """An enclosure without a length attribute should produce an error."""
        xml = _VALID_PODCAST_XML.replace(' length="12345"', "")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("enclosure missing 'length'" in e for e in result.errors)

    def test_podcast_enclosure_invalid_length_fails(self):
        """An enclosure with a non-integer length should produce an error."""
        xml = _VALID_PODCAST_XML.replace(' length="12345"', ' length="not-a-number"')
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any("not a valid integer" in e for e in result.errors)

    def test_articles_feed_no_enclosure_check(self):
        """Articles feed should not require enclosure tags."""
        # Same XML as podcast but validated as articles (no enclosure check)
        result = validate_feed(_VALID_PODCAST_XML, feed_name="articles.xml", is_podcast=False)
        # The enclosure is present but should not be required
        assert result.is_valid

    def test_articles_feed_item_without_enclosure_passes(self):
        """Articles feed items without enclosures should pass validation."""
        xml = (
            _VALID_ARTICLES_XML.replace(
                '      <enclosure url="https://example.com/test.mp4" '
                'length="12345" type="audio/mp4"/>\n',
                "",
            )
            if "<enclosure" in _VALID_ARTICLES_XML
            else _VALID_ARTICLES_XML
        )
        result = validate_feed(xml, feed_name="articles.xml", is_podcast=False)
        assert result.is_valid

    def test_multiple_items_individual_errors(self):
        """Errors should be reported per-item with correct indices."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Multi</title>
    <link>https://example.com</link>
    <description>Multi</description>
    <item>
      <title>Item 1</title>
      <link>https://example.com/1</link>
      <guid>https://example.com/1</guid>
      <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>
    </item>
    <item>
      <link>https://example.com/2</link>
      <guid>https://example.com/2</guid>
    </item>
  </channel>
</rss>
"""
        result = validate_feed(xml, feed_name="multi.xml", is_podcast=False)
        assert not result.is_valid
        # Item 2 is missing title and pubDate
        assert any("Item 2" in e and "title" in e for e in result.errors)
        assert any("Item 2" in e and "pubDate" in e for e in result.errors)
        # Item 1 should have no errors
        assert not any("Item 1" in e for e in result.errors)

    def test_unescaped_special_characters_detected(self):
        """Unescaped special characters should cause a parse error."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test & Invalid</title>
    <link>https://example.com</link>
    <description>Desc</description>
    <item>
      <title>Item <with unescaped></item>
    </item>
  </channel>
</rss>
"""
        result = validate_feed(xml, feed_name="bad.xml")
        assert not result.is_valid
        assert any("not well-formed" in e for e in result.errors)

    def test_feed_validation_result_dataclass(self):
        """FeedValidationResult should have correct default values."""
        result = FeedValidationResult(feed_name="test.xml")
        assert result.feed_name == "test.xml"
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.is_valid is True

    def test_feed_validation_result_with_errors(self):
        """FeedValidationResult with errors should report is_valid=False."""
        result = FeedValidationResult(
            feed_name="test.xml",
            valid=False,
            errors=["Some error"],
        )
        assert result.is_valid is False

    def test_feed_validation_result_with_only_warnings(self):
        """FeedValidationResult with only warnings should still be valid."""
        result = FeedValidationResult(
            feed_name="test.xml",
            warnings=["Some warning"],
        )
        assert result.is_valid is True


class TestValidateOutput:
    """Tests for FeedGenerator.validate_output()."""

    def test_validate_output_both_feeds_valid(self, tmp_path):
        """validate_output() should return valid results for generated feeds."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        results = fg.validate_output()
        assert "podcast" in results
        assert "articles" in results
        assert results["podcast"].is_valid
        assert results["articles"].is_valid

    def test_validate_output_missing_podcast_file(self, tmp_path):
        """validate_output() should report error when podcast.xml is missing."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.validate_output()

        assert not results["podcast"].is_valid
        assert any("does not exist" in e for e in results["podcast"].errors)

    def test_validate_output_missing_articles_file(self, tmp_path):
        """validate_output() should report error when articles.xml is missing."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.validate_output()

        assert not results["articles"].is_valid
        assert any("does not exist" in e for e in results["articles"].errors)

    def test_validate_output_after_podcast_only(self, tmp_path):
        """validate_output() should show articles missing if only podcast generated."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        results = fg.validate_output()
        assert results["podcast"].is_valid
        assert not results["articles"].is_valid
        assert any("does not exist" in e for e in results["articles"].errors)

    def test_validate_output_podcast_enclosure_checks(self, tmp_path):
        """validate_output() should enforce enclosure checks on podcast feed."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        results = fg.validate_output()
        assert results["podcast"].is_valid
        # Verify enclosure was checked (no errors means it passed)
        assert all("enclosure" not in e for e in results["podcast"].errors)

    def test_validate_output_articles_no_enclosure_check(self, tmp_path):
        """validate_output() should not require enclosures for articles feed."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        results = fg.validate_output()
        assert results["articles"].is_valid
        # Verify no enclosure-related errors
        assert all("enclosure" not in e for e in results["articles"].errors)


class TestGenerateAllFeedsWithValidation:
    """Tests for the optional validation parameter in generate_all_feeds()."""

    def test_generate_all_feeds_with_validation_logs_no_errors(self, tmp_path, caplog):
        """generate_all_feeds(validate=True) should log validation success."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        with caplog.at_level("INFO"):
            fg.generate_all_feeds(validate=True)

        assert any("Feed validation passed" in record.message for record in caplog.records)

    def test_generate_all_feeds_without_validation_no_logging(self, tmp_path, caplog):
        """generate_all_feeds(validate=False) should not log validation results."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        with caplog.at_level("INFO"):
            fg.generate_all_feeds(validate=False)

        assert not any("Feed validation" in record.message for record in caplog.records)

    def test_generate_all_feeds_default_no_validation(self, tmp_path, caplog):
        """generate_all_feeds() should not validate by default."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        with caplog.at_level("INFO"):
            fg.generate_all_feeds()

        assert not any("Feed validation" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Markdown to HTML conversion (_markdown_to_html)
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    """Tests for the _markdown_to_html() function."""

    def test_none_returns_empty(self):
        """None input should return an empty string."""
        assert _markdown_to_html(None) == ""

    def test_empty_string_returns_empty(self):
        """Empty string should return an empty string."""
        assert _markdown_to_html("") == ""

    # --- Headers ---

    def test_h1_header(self):
        """# h1 should produce <h1>."""
        assert _markdown_to_html("# Heading 1") == "<h1>Heading 1</h1>"

    def test_h2_header(self):
        """## h2 should produce <h2>."""
        assert _markdown_to_html("## Heading 2") == "<h2>Heading 2</h2>"

    def test_h3_header(self):
        """### h3 should produce <h3>."""
        assert _markdown_to_html("### Heading 3") == "<h3>Heading 3</h3>"

    def test_h4_header(self):
        """#### h4 should produce <h4>."""
        assert _markdown_to_html("#### Heading 4") == "<h4>Heading 4</h4>"

    def test_h5_header(self):
        """##### h5 should produce <h5>."""
        assert _markdown_to_html("##### Heading 5") == "<h5>Heading 5</h5>"

    def test_h6_header(self):
        """###### h6 should produce <h6>."""
        assert _markdown_to_html("###### Heading 6") == "<h6>Heading 6</h6>"

    def test_header_with_inline_bold(self):
        """Headers should support inline markdown (bold)."""
        result = _markdown_to_html("# Header with **bold**")
        assert result == "<h1>Header with <strong>bold</strong></h1>"

    def test_header_with_inline_italic(self):
        """Headers should support inline markdown (italic)."""
        result = _markdown_to_html("## Header with *italic*")
        assert result == "<h2>Header with <em>italic</em></h2>"

    # --- Bold ---

    def test_bold_double_asterisk(self):
        """**text** should produce <strong>text</strong>."""
        assert _markdown_to_html("**bold**") == "<p><strong>bold</strong></p>"

    def test_bold_in_paragraph(self):
        """Bold within a paragraph should be converted."""
        result = _markdown_to_html("This is **bold** text")
        assert result == "<p>This is <strong>bold</strong> text</p>"

    # --- Italic ---

    def test_italic_single_asterisk(self):
        """*text* should produce <em>text</em>."""
        assert _markdown_to_html("*italic*") == "<p><em>italic</em></p>"

    def test_italic_underscore(self):
        """_text_ should produce <em>text</em>."""
        assert _markdown_to_html("_italic_") == "<p><em>italic</em></p>"

    def test_italic_underscore_with_spaces(self):
        """_italic text_ should produce <em>italic text</em>."""
        assert _markdown_to_html("_italic text_") == "<p><em>italic text</em></p>"

    def test_italic_in_paragraph(self):
        """Italic within a paragraph should be converted."""
        result = _markdown_to_html("This is *italic* text")
        assert result == "<p>This is <em>italic</em> text</p>"

    def test_underscore_not_italic_in_word(self):
        """Underscores within words should not be treated as italic."""
        result = _markdown_to_html("file_name variable")
        assert result == "<p>file_name variable</p>"

    # --- Links ---

    def test_link(self):
        """[text](url) should produce <a href='url'>text</a>."""
        result = _markdown_to_html("[example](https://example.com)")
        assert '<a href="https://example.com">example</a>' in result

    def test_link_in_paragraph(self):
        """Links within a paragraph should be converted."""
        result = _markdown_to_html("See [example](https://example.com) for details")
        assert '<a href="https://example.com">example</a>' in result

    def test_link_with_mailto(self):
        """mailto: links should be allowed."""
        result = _markdown_to_html("[email](mailto:test@example.com)")
        assert '<a href="mailto:test@example.com">email</a>' in result

    def test_link_with_relative_url(self):
        """Relative URLs should be allowed."""
        result = _markdown_to_html("[page](/path/to/page)")
        assert '<a href="/path/to/page">page</a>' in result

    def test_javascript_link_blocked(self):
        """javascript: URLs should be blocked (XSS protection)."""
        result = _markdown_to_html("[bad](javascript:alert(1))")
        assert "javascript:" not in result
        assert "<a " not in result

    def test_unsafe_link_text_preserved(self):
        """When a link URL is unsafe, the link text should still appear."""
        result = _markdown_to_html("[safe text](javascript:alert(1))")
        assert "safe text" in result

    # --- Lists ---

    def test_unordered_list_dash(self):
        """- items should produce a <ul> with <li> elements."""
        result = _markdown_to_html("- item 1\n- item 2")
        assert "<ul>" in result
        assert "<li>item 1</li>" in result
        assert "<li>item 2</li>" in result
        assert "</ul>" in result

    def test_unordered_list_asterisk(self):
        """* items should produce a <ul> with <li> elements."""
        result = _markdown_to_html("* item 1\n* item 2")
        assert "<ul>" in result
        assert "<li>item 1</li>" in result
        assert "<li>item 2</li>" in result

    def test_unordered_list_plus(self):
        """+ items should produce a <ul> with <li> elements."""
        result = _markdown_to_html("+ item 1\n+ item 2")
        assert "<ul>" in result
        assert "<li>item 1</li>" in result
        assert "<li>item 2</li>" in result

    def test_ordered_list(self):
        """1. items should produce an <ol> with <li> elements."""
        result = _markdown_to_html("1. first\n2. second")
        assert "<ol>" in result
        assert "<li>first</li>" in result
        assert "<li>second</li>" in result
        assert "</ol>" in result

    def test_list_item_with_inline_markdown(self):
        """List items should support inline markdown."""
        result = _markdown_to_html("- **bold** item")
        assert "<li><strong>bold</strong> item</li>" in result

    # --- Paragraphs and line breaks ---

    def test_paragraph(self):
        """A single line of text should be wrapped in <p>."""
        result = _markdown_to_html("Some text")
        assert result == "<p>Some text</p>"

    def test_multiple_paragraphs(self):
        """Double newline should separate paragraphs."""
        result = _markdown_to_html("Para 1\n\nPara 2")
        assert "<p>Para 1</p>" in result
        assert "<p>Para 2</p>" in result

    def test_line_break_in_paragraph(self):
        """Single newline within a paragraph should produce <br>."""
        result = _markdown_to_html("Line 1\nLine 2")
        assert "<br>" in result
        assert "Line 1" in result
        assert "Line 2" in result

    # --- Escaping ---

    def test_html_special_chars_escaped(self):
        """HTML special characters should be escaped."""
        result = _markdown_to_html("<script>alert(1)</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_ampersand_escaped(self):
        """Ampersands should be escaped."""
        result = _markdown_to_html("a & b")
        assert "a &amp; b" in result

    def test_less_than_escaped(self):
        """Less-than signs should be escaped."""
        result = _markdown_to_html("a < b")
        assert "a &lt; b" in result

    def test_greater_than_escaped(self):
        """Greater-than signs should be escaped."""
        result = _markdown_to_html("a > b")
        assert "a &gt; b" in result

    def test_quotes_escaped(self):
        """Double quotes should be escaped."""
        result = _markdown_to_html('say "hello"')
        assert "&quot;" in result

    # --- Combined / complex ---

    def test_complex_markdown(self):
        """A complex markdown document should be converted correctly."""
        md = "# Title\n\nSome **bold** and *italic* text.\n\n- Item 1\n- Item 2\n\n[link](https://example.com)"
        result = _markdown_to_html(md)
        assert "<h1>Title</h1>" in result
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result
        assert '<a href="https://example.com">link</a>' in result

    def test_markdown_and_feedgen_escaping(self, tmp_path):
        """HTML output from _markdown_to_html should be valid when embedded in feedgen output."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="**Bold** and *italic* text with <tags>",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # The HTML tags from markdown should be present (XML-escaped by feedgen)
        assert "&lt;strong&gt;Bold&lt;/strong&gt;" in content
        assert "&lt;em&gt;italic&lt;/em&gt;" in content
        # The raw <tags> from the input should be escaped by both the markdown
        # converter and feedgen (double-escaped in XML output)
        assert "&amp;lt;tags&amp;gt;" in content


# ---------------------------------------------------------------------------
# Per-folder feed.json discovery
# ---------------------------------------------------------------------------


def write_feed_json(data_dir: Path, folder: str, title: str, description: str) -> Path:
    """Write a feed.json file to a subfolder within data_dir.

    Args:
        data_dir: Root data directory path.
        folder: Relative folder path (e.g. "Course1" or "Course1/Sub").
            Use "." for the data root.
        title: Feed title.
        description: Feed description.

    Returns:
        Path to the written feed.json file.
    """
    folder_path = data_dir / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    feed_json_path = folder_path / "feed.json"
    feed_json_path.write_text(
        json.dumps({"title": title, "description": description}),
        encoding="utf-8",
    )
    return feed_json_path


class TestDiscoverFeedJsonDirs:
    """Tests for _discover_feed_json_dirs()."""

    def test_no_feed_json_returns_empty(self, tmp_path):
        """No feed.json files should return an empty list."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_feed_json_dirs()

        assert result == []

    def test_finds_single_feed_json(self, tmp_path):
        """A single feed.json should be found."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1 Feed", "Description 1")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_feed_json_dirs()

        assert len(result) == 1
        folder_path, config = result[0]
        assert folder_path == tmp_path / "Course1"
        assert config["title"] == "Course 1 Feed"
        assert config["description"] == "Description 1"

    def test_finds_multiple_feed_json(self, tmp_path):
        """Multiple feed.json files should all be found."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course2/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Desc 1")
        write_feed_json(tmp_path, "Course2", "Course 2", "Desc 2")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_feed_json_dirs()

        assert len(result) == 2
        titles = {config["title"] for _, config in result}
        assert titles == {"Course 1", "Course 2"}

    def test_finds_nested_feed_json(self, tmp_path):
        """Feed.json in deeply nested directories should be found."""
        write_metadata_file(tmp_path, "Course1/Sub/Deep/lecture.mp4")
        write_feed_json(tmp_path, "Course1/Sub/Deep", "Deep Course", "Deep Desc")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_feed_json_dirs()

        assert len(result) == 1
        folder_path, config = result[0]
        assert folder_path == tmp_path / "Course1" / "Sub" / "Deep"
        assert config["title"] == "Deep Course"

    def test_finds_root_feed_json(self, tmp_path):
        """A feed.json at the data root should be found."""
        write_metadata_file(tmp_path, "lecture.mp4")
        write_feed_json(tmp_path, ".", "Root Feed", "Root Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_feed_json_dirs()

        assert len(result) == 1
        folder_path, config = result[0]
        assert folder_path == tmp_path
        assert config["title"] == "Root Feed"

    def test_invalid_feed_json_skipped(self, tmp_path, caplog):
        """Invalid feed.json should be skipped with a warning."""
        write_feed_json(tmp_path, "Course1", "Valid", "Valid desc")
        # Write an invalid feed.json in another folder
        bad_folder = tmp_path / "Course2"
        bad_folder.mkdir(parents=True, exist_ok=True)
        (bad_folder / "feed.json").write_text("{invalid json", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_feed_json_dirs()

        assert len(result) == 1
        assert result[0][1]["title"] == "Valid"
        assert any(
            "Failed to parse feed.json" in record.message
            for record in caplog.records
            if record.levelno >= 30  # WARNING
        )


# ---------------------------------------------------------------------------
# Per-folder metadata filtering
# ---------------------------------------------------------------------------


class TestFilterMetadataForFolder:
    """Tests for _filter_metadata_for_folder()."""

    def test_filters_to_subfolder(self, tmp_path):
        """Should return only metadata within the specified folder."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_metadata_file(tmp_path, "Course2/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        all_metadata = fg._discover_metadata_files()

        course1_metadata = fg._filter_metadata_for_folder(all_metadata, tmp_path / "Course1")

        assert len(course1_metadata) == 2
        filepaths = {m.filepath for m in course1_metadata}
        assert filepaths == {"Course1/lecture.mp4", "Course1/doc.pdf"}

    def test_filters_to_deeply_nested_folder(self, tmp_path):
        """Should return only metadata within a deeply nested folder."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/Sub/lesson.mp4")
        write_metadata_file(tmp_path, "Course1/Sub/Deep/video.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        all_metadata = fg._discover_metadata_files()

        deep_metadata = fg._filter_metadata_for_folder(
            all_metadata, tmp_path / "Course1" / "Sub" / "Deep"
        )

        assert len(deep_metadata) == 1
        assert deep_metadata[0].filepath == "Course1/Sub/Deep/video.mp4"

    def test_root_folder_returns_all(self, tmp_path):
        """Filtering by the root data_dir should return all metadata."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course2/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        all_metadata = fg._discover_metadata_files()

        root_metadata = fg._filter_metadata_for_folder(all_metadata, tmp_path)

        assert len(root_metadata) == 2

    def test_folder_with_no_metadata_returns_empty(self, tmp_path):
        """A folder with no metadata files should return an empty list."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        (tmp_path / "EmptyFolder").mkdir(parents=True, exist_ok=True)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        all_metadata = fg._discover_metadata_files()

        empty_metadata = fg._filter_metadata_for_folder(all_metadata, tmp_path / "EmptyFolder")

        assert empty_metadata == []

    def test_does_not_match_sibling_prefix(self, tmp_path):
        """A folder named 'ile' should not match files in 'ile-something'."""
        write_metadata_file(tmp_path, "ile/lecture.mp4")
        write_metadata_file(tmp_path, "ile-something/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        all_metadata = fg._discover_metadata_files()

        ile_metadata = fg._filter_metadata_for_folder(all_metadata, tmp_path / "ile")

        assert len(ile_metadata) == 1
        assert ile_metadata[0].filepath == "ile/lecture.mp4"


# ---------------------------------------------------------------------------
# Per-folder feed generation (_generate_feeds_for_folder)
# ---------------------------------------------------------------------------


class TestGenerateFeedsForFolder:
    """Tests for _generate_feeds_for_folder()."""

    def test_generates_both_feeds_when_both_file_types_present(self, tmp_path):
        """Both podcast.xml and articles.xml should be generated when both
        MP4 and PDF/HTML files are present."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        results = fg._generate_feeds_for_folder(
            folder=folder,
            title="Course 1",
            description="Description",
            metadata_items=folder_metadata,
        )

        assert len(results["podcast"]) == 1
        assert len(results["articles"]) == 1
        assert (folder / "podcast.xml").exists()
        assert (folder / "articles.xml").exists()

    def test_generates_only_podcast_when_mp4_present(self, tmp_path):
        """Only podcast.xml should be generated when only MP4 files are present."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        results = fg._generate_feeds_for_folder(
            folder=folder,
            title="Course 1",
            description="Description",
            metadata_items=folder_metadata,
        )

        assert len(results["podcast"]) == 1
        assert results["articles"] == []
        assert (folder / "podcast.xml").exists()
        assert not (folder / "articles.xml").exists()

    def test_generates_only_articles_when_pdf_html_present(self, tmp_path):
        """Only articles.xml should be generated when only PDF/HTML files are present."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_metadata_file(tmp_path, "Course1/article.html")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        results = fg._generate_feeds_for_folder(
            folder=folder,
            title="Course 1",
            description="Description",
            metadata_items=folder_metadata,
        )

        assert results["podcast"] == []
        assert len(results["articles"]) == 2
        assert not (folder / "podcast.xml").exists()
        assert (folder / "articles.xml").exists()

    def test_no_feeds_generated_when_no_metadata(self, tmp_path):
        """No feeds should be generated when there are no metadata items."""
        folder = tmp_path / "Course1"
        folder.mkdir()

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        results = fg._generate_feeds_for_folder(
            folder=folder,
            title="Course 1",
            description="Description",
            metadata_items=[],
        )

        assert results["podcast"] == []
        assert results["articles"] == []
        assert not (folder / "podcast.xml").exists()
        assert not (folder / "articles.xml").exists()

    def test_uses_title_from_feed_json(self, tmp_path):
        """Feed title from feed.json should appear in the generated feed."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        fg._generate_feeds_for_folder(
            folder=folder,
            title="My Custom Course Feed",
            description="Custom description",
            metadata_items=folder_metadata,
        )

        content = parse_rss_file(folder / "podcast.xml")
        assert "<title>My Custom Course Feed</title>" in content
        assert "<description>Custom description</description>" in content

    def test_uses_default_title_when_none_provided(self, tmp_path):
        """Default title should be used when title is None."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        fg._generate_feeds_for_folder(
            folder=folder,
            title=None,
            description=None,
            metadata_items=folder_metadata,
        )

        content = parse_rss_file(folder / "podcast.xml")
        assert "<title>Local Podcast Feed</title>" in content

    def test_podcast_feed_has_correct_self_link(self, tmp_path):
        """Per-folder podcast feed should have correct self-referencing link."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        fg._generate_feeds_for_folder(
            folder=folder,
            title="Course 1",
            description="Description",
            metadata_items=folder_metadata,
        )

        content = parse_rss_file(folder / "podcast.xml")
        assert 'rel="self"' in content
        assert "https://example.com/feeds/Course1/podcast.xml" in content

    def test_articles_feed_has_correct_self_link(self, tmp_path):
        """Per-folder articles feed should have correct self-referencing link."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        fg._generate_feeds_for_folder(
            folder=folder,
            title="Course 1",
            description="Description",
            metadata_items=folder_metadata,
        )

        content = parse_rss_file(folder / "articles.xml")
        assert 'rel="self"' in content
        assert "https://example.com/feeds/Course1/articles.xml" in content

    def test_error_in_podcast_does_not_block_articles(self, tmp_path):
        """If podcast generation fails, articles should still be generated."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        with mock.patch.object(
            fg,
            "generate_podcast_feed",
            side_effect=RuntimeError("Podcast generation failed"),
        ):
            results = fg._generate_feeds_for_folder(
                folder=folder,
                title="Course 1",
                description="Description",
                metadata_items=folder_metadata,
            )

        assert results["podcast"] == []
        assert len(results["articles"]) == 1
        assert not (folder / "podcast.xml").exists()
        assert (folder / "articles.xml").exists()

    def test_error_in_articles_does_not_block_podcast(self, tmp_path):
        """If articles generation fails, podcast should still be generated."""
        folder = tmp_path / "Course1"
        folder.mkdir()
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, folder)

        with mock.patch.object(
            fg,
            "generate_articles_feed",
            side_effect=RuntimeError("Articles generation failed"),
        ):
            results = fg._generate_feeds_for_folder(
                folder=folder,
                title="Course 1",
                description="Description",
                metadata_items=folder_metadata,
            )

        assert len(results["podcast"]) == 1
        assert results["articles"] == []
        assert (folder / "podcast.xml").exists()
        assert not (folder / "articles.xml").exists()


# ---------------------------------------------------------------------------
# generate_all_feeds() with per-folder feeds
# ---------------------------------------------------------------------------


class TestGenerateAllFeedsWithPerFolder:
    """Tests for generate_all_feeds() with per-folder feed.json files."""

    def test_no_root_feeds_without_feed_json(self, tmp_path):
        """Root feeds should NOT be generated when no feed.json exists."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert results["podcast"] == []
        assert results["articles"] == []
        assert not (tmp_path / "podcast.xml").exists()
        assert not (tmp_path / "articles.xml").exists()

    def test_root_feed_json_with_no_mp4_skips_podcast(self, tmp_path):
        """Root feed.json with no MP4 files should skip podcast.xml."""
        write_metadata_file(tmp_path, "documents/doc.pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert results["podcast"] == []
        assert not (tmp_path / "podcast.xml").exists()
        assert len(results["articles"]) == 1
        assert (tmp_path / "articles.xml").exists()

    def test_root_feed_json_with_no_pdf_html_skips_articles(self, tmp_path):
        """Root feed.json with no PDF/HTML files should skip articles.xml."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert results["articles"] == []
        assert not (tmp_path / "articles.xml").exists()
        assert len(results["podcast"]) == 1
        assert (tmp_path / "podcast.xml").exists()

    def test_root_feed_json_with_no_eligible_files_skips_both(self, tmp_path):
        """Root feed.json with no eligible files should skip both feeds."""
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert results["podcast"] == []
        assert results["articles"] == []
        assert not (tmp_path / "podcast.xml").exists()
        assert not (tmp_path / "articles.xml").exists()

    def test_root_feed_json_with_both_file_types_generates_both(self, tmp_path):
        """Root feed.json with both MP4 and PDF/HTML files should generate both feeds."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert len(results["podcast"]) == 1
        assert len(results["articles"]) == 1
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()

    def test_generates_per_folder_feeds(self, tmp_path):
        """Per-folder feeds should be generated when feed.json exists."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_feed_json(tmp_path, "Course1", "Course 1 Feed", "Course 1 Description")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root feeds should also exist (since root feed.json exists)
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()

        # Per-folder feeds should also exist
        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "articles.xml").exists()

    def test_per_folder_feed_uses_title_from_feed_json(self, tmp_path):
        """Per-folder feed title should come from feed.json."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "My Course Feed", "My Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<title>My Course Feed</title>" in content
        assert "<description>My Description</description>" in content

    def test_per_folder_podcast_only_when_mp4_present(self, tmp_path):
        """Per-folder podcast.xml should only be generated if MP4 files exist."""
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert not (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "articles.xml").exists()

    def test_per_folder_articles_only_when_pdf_html_present(self, tmp_path):
        """Per-folder articles.xml should only be generated if PDF/HTML files exist."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert not (tmp_path / "Course1" / "articles.xml").exists()

    def test_per_folder_no_feeds_when_no_eligible_files(self, tmp_path):
        """No per-folder feeds should be generated if no eligible files exist."""
        write_feed_json(tmp_path, "EmptyCourse", "Empty Course", "No files here")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert not (tmp_path / "EmptyCourse" / "podcast.xml").exists()
        assert not (tmp_path / "EmptyCourse" / "articles.xml").exists()

    def test_per_folder_feeds_only_include_files_in_that_folder(self, tmp_path):
        """Per-folder feeds should only include files from that folder's subtree."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course2/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        write_feed_json(tmp_path, "Course2", "Course 2", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Course1 feed should only have Course1 files
        content1 = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "Course1/lecture.mp4" in content1
        assert "Course2/lecture.mp4" not in content1

        # Course2 feed should only have Course2 files
        content2 = parse_rss_file(tmp_path / "Course2" / "podcast.xml")
        assert "Course2/lecture.mp4" in content2
        assert "Course1/lecture.mp4" not in content2

    def test_deeply_nested_folder_feeds(self, tmp_path):
        """Feeds should be generated for deeply nested folders with feed.json."""
        write_metadata_file(tmp_path, "Course1/Sub/Deep/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/Sub/Deep/doc.pdf")
        write_feed_json(tmp_path, "Course1/Sub/Deep", "Deep Course", "Deep Description")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        deep_dir = tmp_path / "Course1" / "Sub" / "Deep"
        assert (deep_dir / "podcast.xml").exists()
        assert (deep_dir / "articles.xml").exists()

        # Root feeds should also exist (root feed.json exists)
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()

    def test_overlapping_feed_trees(self, tmp_path):
        """Parent and child folders with feed.json should both generate feeds.

        The parent folder's feed should include all files in its subtree
        (including child folder files), while the child folder's feed should
        only include its own files.
        """
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/Sub/lesson.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        write_feed_json(tmp_path, "Course1/Sub", "Course 1 Sub", "Sub Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Both feeds should exist
        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "Sub" / "podcast.xml").exists()

        # Parent feed should include both files
        parent_content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "Course1/lecture.mp4" in parent_content
        assert "Course1/Sub/lesson.mp4" in parent_content

        # Child feed should only include its own file
        child_content = parse_rss_file(tmp_path / "Course1" / "Sub" / "podcast.xml")
        assert "Course1/Sub/lesson.mp4" in child_content
        assert "Course1/lecture.mp4" not in child_content

    def test_root_feed_json_uses_custom_title(self, tmp_path):
        """Root-level feed.json should provide title/description for root feeds."""
        write_metadata_file(tmp_path, "lecture.mp4")
        write_metadata_file(tmp_path, "doc.pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Custom Feed", "description": "Root Custom Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        podcast_content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<title>Root Custom Feed</title>" in podcast_content
        assert "<description>Root Custom Desc</description>" in podcast_content

        articles_content = parse_rss_file(tmp_path / "articles.xml")
        assert "<title>Root Custom Feed</title>" in articles_content
        assert "<description>Root Custom Desc</description>" in articles_content

    def test_per_folder_feed_excludes_error_entries(self, tmp_path):
        """Per-folder feeds should exclude error entries."""
        write_metadata_file(tmp_path, "Course1/good.mp4")
        write_metadata_file(tmp_path, "Course1/bad.mp4", error="Failed")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "good.mp4" in content
        assert "bad.mp4" not in content

    def test_multiple_folders_with_feed_json(self, tmp_path):
        """Multiple folders with feed.json should each get their own feeds."""
        write_metadata_file(tmp_path, "Math/lecture.mp4")
        write_metadata_file(tmp_path, "Math/notes.pdf")
        write_metadata_file(tmp_path, "Science/experiment.mp4")
        write_metadata_file(tmp_path, "Science/article.html")
        write_feed_json(tmp_path, "Math", "Math Course", "Math Description")
        write_feed_json(tmp_path, "Science", "Science Course", "Science Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Math feeds
        math_podcast = parse_rss_file(tmp_path / "Math" / "podcast.xml")
        assert "Math/lecture.mp4" in math_podcast
        assert "Science" not in math_podcast

        math_articles = parse_rss_file(tmp_path / "Math" / "articles.xml")
        assert "Math/notes.pdf" in math_articles

        # Science feeds
        science_podcast = parse_rss_file(tmp_path / "Science" / "podcast.xml")
        assert "Science/experiment.mp4" in science_podcast
        assert "Math" not in science_podcast

        science_articles = parse_rss_file(tmp_path / "Science" / "articles.xml")
        assert "Science/article.html" in science_articles

    def test_mixed_feed_json_and_no_feed_json(self, tmp_path):
        """Folders without feed.json should not get per-folder feeds; root feeds
        only exist if root feed.json exists."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "NoFeedConfig/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Course1 should have per-folder feeds
        assert (tmp_path / "Course1" / "podcast.xml").exists()

        # NoFeedConfig should NOT have per-folder feeds
        assert not (tmp_path / "NoFeedConfig" / "podcast.xml").exists()

        # Root feeds should NOT exist (no root feed.json)
        assert not (tmp_path / "podcast.xml").exists()
        assert not (tmp_path / "articles.xml").exists()

    def test_returns_root_level_results(self, tmp_path):
        """generate_all_feeds should return root-level results when root feed.json exists."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        # Results should contain root-level items
        assert "podcast" in results
        assert "articles" in results
        # Root podcast should include root-level MP4
        root_podcast_paths = {m.filepath for m in results["podcast"]}
        assert "videos/lecture.mp4" in root_podcast_paths
        # Root articles should include root-level PDF
        root_article_paths = {m.filepath for m in results["articles"]}
        assert "documents/doc.pdf" in root_article_paths

    def test_per_folder_feed_with_html_files(self, tmp_path):
        """Per-folder articles.xml should include HTML files."""
        write_metadata_file(tmp_path, "Course1/article.html")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert not (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "articles.xml").exists()

        content = parse_rss_file(tmp_path / "Course1" / "articles.xml")
        assert "Course1/article.html" in content

    def test_per_folder_feed_with_mixed_file_types(self, tmp_path):
        """Per-folder feeds should handle mixed MP4, PDF, and HTML files."""
        write_metadata_file(tmp_path, "Course1/video.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_metadata_file(tmp_path, "Course1/page.html")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Both feeds should exist
        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "articles.xml").exists()

        # Podcast should only have MP4
        podcast_content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "video.mp4" in podcast_content
        assert "doc.pdf" not in podcast_content
        assert "page.html" not in podcast_content

        # Articles should have PDF and HTML
        articles_content = parse_rss_file(tmp_path / "Course1" / "articles.xml")
        assert "doc.pdf" in articles_content
        assert "page.html" in articles_content
        assert "video.mp4" not in articles_content

    def test_per_folder_feed_items_sorted(self, tmp_path):
        """Per-folder feed items should be sorted by folder then filename."""
        write_metadata_file(tmp_path, "Course1/zebra.mp4")
        write_metadata_file(tmp_path, "Course1/apple.mp4")
        write_metadata_file(tmp_path, "Course1/mango.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        # Discover and filter metadata the same way generate_all_feeds does
        all_metadata = fg._discover_metadata_files()
        folder_metadata = fg._filter_metadata_for_folder(all_metadata, tmp_path / "Course1")

        # The metadata items should be sorted by folder then filename
        filepaths = [m.filepath for m in folder_metadata]
        assert filepaths == [
            "Course1/apple.mp4",
            "Course1/mango.mp4",
            "Course1/zebra.mp4",
        ]

    def test_feed_json_with_missing_title_uses_default(self, tmp_path):
        """Per-folder feed with missing title should use default title."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        (tmp_path / "Course1").mkdir(parents=True, exist_ok=True)
        (tmp_path / "Course1" / "feed.json").write_text(
            json.dumps({"description": "Only description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        # Should use default podcast title since title is missing
        assert "<title>Local Podcast Feed</title>" in content
        assert "<description>Only description</description>" in content


# ---------------------------------------------------------------------------
# Nested feed scenarios (Course1/Unit1 structure)
# ---------------------------------------------------------------------------


class TestNestedFeedScenarios:
    """Comprehensive tests for nested feed.json files and feed scope.

    These tests verify the full Course1/Unit1 directory structure:

    data/
      feed.json                    ← Root feed with ALL files
      Course1/
        feed.json                 ← Course1 feed with Course1/* files
        lecture1.mp4
        Unit1/
          feed.json               ← Unit1 feed with Course1/Unit1/* files
          lecture2.mp4
    """

    def _setup_nested_structure(self, tmp_path: Path) -> None:
        """Create the Course1/Unit1 nested structure with feed.json files.

        Creates:
            - data/feed.json (root)
            - data/Course1/feed.json
            - data/Course1/lecture1.mp4 (+ metadata)
            - data/Course1/notes.pdf (+ metadata)
            - data/Course1/Unit1/feed.json
            - data/Course1/Unit1/lecture2.mp4 (+ metadata)
            - data/Course1/Unit1/article.html (+ metadata)
        """
        # Root feed.json
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )
        # Course1 feed.json
        write_feed_json(tmp_path, "Course1", "Course 1 Feed", "Course 1 Description")
        # Course1/Unit1 feed.json
        write_feed_json(tmp_path, "Course1/Unit1", "Unit 1 Feed", "Unit 1 Description")
        # Source files + metadata (MP4 and PDF/HTML for both podcast and articles feeds)
        write_metadata_file(tmp_path, "Course1/lecture1.mp4")
        write_metadata_file(tmp_path, "Course1/notes.pdf")
        write_metadata_file(tmp_path, "Course1/Unit1/lecture2.mp4")
        write_metadata_file(tmp_path, "Course1/Unit1/article.html")

    def test_nested_feeds_generate_in_respective_folders(self, tmp_path):
        """Nested feed.json files should generate feeds in their respective folders."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root feeds
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()
        # Course1 feeds
        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "articles.xml").exists()
        # Unit1 feeds
        assert (tmp_path / "Course1" / "Unit1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "Unit1" / "articles.xml").exists()

    def test_root_feeds_contain_all_files(self, tmp_path):
        """Root feeds should contain files from all subdirectories."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root podcast feed should contain both MP4 files
        root_podcast = parse_rss_file(tmp_path / "podcast.xml")
        assert "Course1/lecture1.mp4" in root_podcast
        assert "Course1/Unit1/lecture2.mp4" in root_podcast

        # Root articles feed should contain both PDF/HTML files
        root_articles = parse_rss_file(tmp_path / "articles.xml")
        assert "Course1/notes.pdf" in root_articles
        assert "Course1/Unit1/article.html" in root_articles

    def test_course1_feeds_contain_own_and_nested_files(self, tmp_path):
        """Course1 feeds should contain files from Course1/ and Course1/Unit1/."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Course1 podcast feed should contain both MP4 files (filepath relative to data_dir)
        course1_podcast = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "Course1/lecture1.mp4" in course1_podcast
        assert "Course1/Unit1/lecture2.mp4" in course1_podcast

        # Course1 articles feed should contain both PDF/HTML files
        course1_articles = parse_rss_file(tmp_path / "Course1" / "articles.xml")
        assert "Course1/notes.pdf" in course1_articles
        assert "Course1/Unit1/article.html" in course1_articles

    def test_unit1_feeds_contain_only_own_files(self, tmp_path):
        """Unit1 feeds should only contain files from Course1/Unit1/."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Unit1 podcast feed should only contain lecture2.mp4
        unit1_podcast = parse_rss_file(tmp_path / "Course1" / "Unit1" / "podcast.xml")
        assert "Course1/Unit1/lecture2.mp4" in unit1_podcast
        assert "Course1/lecture1.mp4" not in unit1_podcast

        # Unit1 articles feed should only contain article.html
        unit1_articles = parse_rss_file(tmp_path / "Course1" / "Unit1" / "articles.xml")
        assert "Course1/Unit1/article.html" in unit1_articles
        assert "Course1/notes.pdf" not in unit1_articles

    def test_inner_feed_is_subset_of_outer_feed(self, tmp_path):
        """Inner feeds should be subsets of outer feeds (nested relationship)."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root feed contains both files
        root_podcast = parse_rss_file(tmp_path / "podcast.xml")
        root_items = root_podcast.count("<item>")
        assert root_items == 2

        # Course1 feed contains both files (it's a subset of root)
        course1_podcast = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        course1_items = course1_podcast.count("<item>")
        assert course1_items == 2

        # Unit1 feed contains only one file (subset of Course1)
        unit1_podcast = parse_rss_file(tmp_path / "Course1" / "Unit1" / "podcast.xml")
        unit1_items = unit1_podcast.count("<item>")
        assert unit1_items == 1

        # Verify subset relationship: Unit1 files are in Course1, which are in root
        assert "Course1/Unit1/lecture2.mp4" in root_podcast
        assert "Course1/Unit1/lecture2.mp4" in course1_podcast
        assert "Course1/Unit1/lecture2.mp4" in unit1_podcast

        assert "Course1/lecture1.mp4" in root_podcast
        assert "Course1/lecture1.mp4" in course1_podcast
        assert "Course1/lecture1.mp4" not in unit1_podcast

    def test_nested_feeds_listed_in_index_html(self, tmp_path):
        """All feeds from root and nested folders should be listed in index.html."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # Root feeds
        assert ">podcast.xml<" in content
        assert ">articles.xml<" in content
        # Course1 feeds
        assert ">Course1/podcast.xml<" in content
        assert ">Course1/articles.xml<" in content
        # Unit1 feeds
        assert ">Course1/Unit1/podcast.xml<" in content
        assert ">Course1/Unit1/articles.xml<" in content

    def test_nested_feeds_use_correct_titles(self, tmp_path):
        """Each nested feed should use the title from its own feed.json."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root feed title
        root_content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<title>Root Feed</title>" in root_content

        # Course1 feed title
        course1_content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<title>Course 1 Feed</title>" in course1_content

        # Unit1 feed title
        unit1_content = parse_rss_file(tmp_path / "Course1" / "Unit1" / "podcast.xml")
        assert "<title>Unit 1 Feed</title>" in unit1_content

    def test_nested_feeds_self_links_correct(self, tmp_path):
        """Each nested feed should have a correct self-referencing link."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root feed self link
        root_content = parse_rss_file(tmp_path / "podcast.xml")
        assert 'href="https://example.com/feeds/podcast.xml"' in root_content

        # Course1 feed self link
        course1_content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert 'href="https://example.com/feeds/Course1/podcast.xml"' in course1_content

        # Unit1 feed self link
        unit1_content = parse_rss_file(tmp_path / "Course1" / "Unit1" / "podcast.xml")
        assert 'href="https://example.com/feeds/Course1/Unit1/podcast.xml"' in unit1_content

    def test_filter_metadata_includes_deeply_nested_files(self, tmp_path):
        """_filter_metadata_for_folder should include deeply nested files."""
        write_metadata_file(tmp_path, "Course1/lecture1.mp4")
        write_metadata_file(tmp_path, "Course1/Unit1/lecture2.mp4")
        write_metadata_file(tmp_path, "Course1/Unit1/Sub/deep.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        all_metadata = fg._discover_metadata_files()

        # Filter for Course1 should include all three files
        course1_metadata = fg._filter_metadata_for_folder(all_metadata, tmp_path / "Course1")
        course1_paths = {m.filepath for m in course1_metadata}
        assert course1_paths == {
            "Course1/lecture1.mp4",
            "Course1/Unit1/lecture2.mp4",
            "Course1/Unit1/Sub/deep.mp4",
        }

        # Filter for Unit1 should only include the two files under Unit1
        unit1_metadata = fg._filter_metadata_for_folder(
            all_metadata, tmp_path / "Course1" / "Unit1"
        )
        unit1_paths = {m.filepath for m in unit1_metadata}
        assert unit1_paths == {
            "Course1/Unit1/lecture2.mp4",
            "Course1/Unit1/Sub/deep.mp4",
        }

    def test_root_feeds_include_subdirectory_files(self, tmp_path):
        """Root feeds should include files from subdirectories without feed.json."""
        # Root has feed.json, Course1 has feed.json, but Course1/Sub does not
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )
        write_feed_json(tmp_path, "Course1", "Course 1", "Desc")
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/Sub/lesson.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Root feed should include files from Course1/Sub (no feed.json there)
        root_podcast = parse_rss_file(tmp_path / "podcast.xml")
        assert "Course1/lecture.mp4" in root_podcast
        assert "Course1/Sub/lesson.mp4" in root_podcast

        # Course1 feed should also include files from Course1/Sub
        course1_podcast = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "Course1/lecture.mp4" in course1_podcast
        assert "Course1/Sub/lesson.mp4" in course1_podcast

    def test_three_level_nested_feeds(self, tmp_path):
        """Feeds should work with three levels of nesting."""
        # Level 0: root
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )
        # Level 1: Course1
        write_feed_json(tmp_path, "Course1", "Course 1", "Desc 1")
        # Level 2: Course1/Unit1
        write_feed_json(tmp_path, "Course1/Unit1", "Unit 1", "Desc 2")
        # Level 3: Course1/Unit1/Module1
        write_feed_json(tmp_path, "Course1/Unit1/Module1", "Module 1", "Desc 3")

        write_metadata_file(tmp_path, "Course1/lecture1.mp4")
        write_metadata_file(tmp_path, "Course1/Unit1/lecture2.mp4")
        write_metadata_file(tmp_path, "Course1/Unit1/Module1/lecture3.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # All feed files should exist
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "Unit1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "Unit1" / "Module1" / "podcast.xml").exists()

        # Root feed: 3 items
        root_content = parse_rss_file(tmp_path / "podcast.xml")
        assert root_content.count("<item>") == 3

        # Course1 feed: 3 items (lecture1, lecture2, lecture3)
        course1_content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert course1_content.count("<item>") == 3

        # Unit1 feed: 2 items (lecture2, lecture3)
        unit1_content = parse_rss_file(tmp_path / "Course1" / "Unit1" / "podcast.xml")
        assert unit1_content.count("<item>") == 2

        # Module1 feed: 1 item (lecture3)
        module1_content = parse_rss_file(tmp_path / "Course1" / "Unit1" / "Module1" / "podcast.xml")
        assert module1_content.count("<item>") == 1

        # Verify specific files in each feed
        assert "Course1/Unit1/Module1/lecture3.mp4" in root_content
        assert "Course1/Unit1/Module1/lecture3.mp4" in course1_content
        assert "Course1/Unit1/Module1/lecture3.mp4" in unit1_content
        assert "Course1/Unit1/Module1/lecture3.mp4" in module1_content

        assert "Course1/lecture1.mp4" in root_content
        assert "Course1/lecture1.mp4" in course1_content
        assert "Course1/lecture1.mp4" not in unit1_content
        assert "Course1/lecture1.mp4" not in module1_content


# ---------------------------------------------------------------------------
# Index HTML generation (_generate_index_html)
# ---------------------------------------------------------------------------


class TestGenerateIndexHtml:
    """Tests for _generate_index_html() and index.html generation."""

    def test_index_html_created_after_generate_all_feeds(self, tmp_path):
        """generate_all_feeds should create index.html in data_dir."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert (tmp_path / "index.html").exists()

    def test_index_html_has_correct_title(self, tmp_path):
        """index.html should have title 'Feed Index'."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<title>Feed Index</title>" in content

    def test_index_html_has_h1_heading(self, tmp_path):
        """index.html should have an h1 heading 'Feed Index'."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<h1>Feed Index</h1>" in content

    def test_index_html_has_podcast_feeds_section(self, tmp_path):
        """index.html should have a 'Podcast Feeds' section."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<h2>Podcast Feeds</h2>" in content

    def test_index_html_has_articles_feeds_section(self, tmp_path):
        """index.html should have an 'Articles Feeds' section."""
        write_metadata_file(tmp_path, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<h2>Articles Feeds</h2>" in content

    def test_index_html_lists_root_feeds(self, tmp_path):
        """index.html should list root-level feed files with correct URLs."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://example.com/feeds/podcast.xml"' in content
        assert 'href="https://example.com/feeds/articles.xml"' in content
        assert ">podcast.xml<" in content
        assert ">articles.xml<" in content

    def test_index_html_lists_per_folder_feeds(self, tmp_path):
        """index.html should list per-folder feed files with correct URLs."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # Root feeds
        assert 'href="https://example.com/feeds/podcast.xml"' in content
        assert 'href="https://example.com/feeds/articles.xml"' in content
        # Per-folder feeds
        assert 'href="https://example.com/feeds/Course1/podcast.xml"' in content
        assert 'href="https://example.com/feeds/Course1/articles.xml"' in content
        assert ">Course1/podcast.xml<" in content
        assert ">Course1/articles.xml<" in content

    def test_index_html_sorted_by_path(self, tmp_path):
        """Feeds in index.html should be sorted by relative path."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        write_metadata_file(tmp_path, "Alpha/lecture.mp4")
        write_feed_json(tmp_path, "Alpha", "Alpha Feed", "Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # Find positions of the two per-folder podcast feeds
        alpha_pos = content.find("Alpha/podcast.xml")
        course1_pos = content.find("Course1/podcast.xml")
        assert alpha_pos < course1_pos

    def test_index_html_empty_directory(self, tmp_path):
        """index.html should be generated even with no feeds (empty sections)."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert (tmp_path / "index.html").exists()
        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<h1>Feed Index</h1>" in content
        assert "<h2>Podcast Feeds</h2>" in content
        assert "<h2>Articles Feeds</h2>" in content

    def test_index_html_overwrites_existing(self, tmp_path):
        """generate_all_feeds should overwrite existing index.html."""
        # Create an initial index.html
        (tmp_path / "index.html").write_text(
            "<html><body>Old content</body></html>",
            encoding="utf-8",
        )

        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "Old content" not in content
        assert "Feed Index" in content

    def test_index_html_nonexistent_data_dir(self, tmp_path):
        """_generate_index_html should handle nonexistent data directory gracefully."""
        fg = FeedGenerator(
            data_dir=tmp_path / "nonexistent",
            base_url="https://example.com/feeds",
        )
        # Should not raise an exception
        fg._generate_index_html()
        # index.html should not be created since directory doesn't exist
        assert not (tmp_path / "nonexistent" / "index.html").exists()

    def test_index_html_special_characters_escaped(self, tmp_path):
        """Special characters in filenames should be HTML-escaped in index."""
        # Create a folder with special characters in the name
        folder = tmp_path / "Course & Co"
        folder.mkdir(parents=True)
        # Create a dummy XML file directly
        (folder / "podcast.xml").write_text("<rss></rss>", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg._generate_index_html()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # The & in the path should be escaped
        assert "&amp;" in content
        assert "&Co" not in content  # Should not have unescaped &

    def test_index_html_only_podcast_feeds(self, tmp_path):
        """index.html should only list podcast feeds when only MP4 files exist."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://example.com/feeds/podcast.xml"' in content
        # articles.xml should NOT exist since no PDF/HTML files
        assert 'href="https://example.com/feeds/articles.xml"' not in content
        # Both sections should exist even if articles feed is empty
        assert "<h2>Articles Feeds</h2>" in content

    def test_index_html_only_articles_feeds(self, tmp_path):
        """index.html should list articles feeds when only PDF/HTML files exist."""
        write_metadata_file(tmp_path, "documents/doc.pdf")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # podcast.xml should NOT exist since no MP4 files
        assert 'href="https://example.com/feeds/podcast.xml"' not in content
        assert 'href="https://example.com/feeds/articles.xml"' in content
        # Both sections should exist even if podcast feed is empty
        assert "<h2>Podcast Feeds</h2>" in content

    def test_index_html_has_css_styles(self, tmp_path):
        """index.html should include the CSS styles from the format spec."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "font-family: Arial, sans-serif" in content
        assert "color: #333" in content
        assert "list-style: none" in content
        assert "color: #0066cc" in content

    def test_index_html_has_meta_charset(self, tmp_path):
        """index.html should include meta charset for proper encoding."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert '<meta charset="utf-8">' in content

    def test_index_html_with_deeply_nested_feeds(self, tmp_path):
        """index.html should list feeds in deeply nested folders."""
        write_metadata_file(tmp_path, "Course1/Sub/Deep/lecture.mp4")
        write_feed_json(tmp_path, "Course1/Sub/Deep", "Deep Course", "Deep Desc")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://example.com/feeds/Course1/Sub/Deep/podcast.xml"' in content
        assert ">Course1/Sub/Deep/podcast.xml<" in content

    def test_index_html_with_multiple_folders(self, tmp_path):
        """index.html should list feeds from multiple folders."""
        write_metadata_file(tmp_path, "Math/lecture.mp4")
        write_feed_json(tmp_path, "Math", "Math Course", "Math Desc")
        write_metadata_file(tmp_path, "Science/experiment.mp4")
        write_feed_json(tmp_path, "Science", "Science Course", "Science Desc")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://example.com/feeds/Math/podcast.xml"' in content
        assert 'href="https://example.com/feeds/Science/podcast.xml"' in content

    def test_index_html_includes_all_xml_files(self, tmp_path):
        """index.html should discover all XML files via rglob, not just feeds."""
        # Create a non-standard XML file
        (tmp_path / "custom.xml").write_text("<root></root>", encoding="utf-8")
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg._generate_index_html()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://example.com/feeds/custom.xml"' in content
        assert ">custom.xml<" in content
        # Should be in "Other Feeds" section
        assert "<h2>Other Feeds</h2>" in content

    def test_index_html_no_other_feeds_section_when_empty(self, tmp_path):
        """index.html should not have 'Other Feeds' section when no non-standard XML files exist."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<h2>Other Feeds</h2>" not in content

    def test_index_html_generated_even_if_feed_generation_fails(self, tmp_path):
        """index.html should still be generated if individual feed generation fails."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        # Mock generate_podcast_feed to raise but articles should still work
        with mock.patch.object(
            fg,
            "generate_podcast_feed",
            side_effect=RuntimeError("Podcast generation failed"),
        ):
            fg.generate_all_feeds()

        # index.html should still be created (it only lists XML files that exist)
        assert (tmp_path / "index.html").exists()

    def test_index_html_absolute_url_uses_base_url(self, tmp_path):
        """index.html links should use the configured BASE_URL."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://feed.bhatia.dev")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://feed.bhatia.dev/podcast.xml"' in content
        # articles.xml should NOT exist since no PDF/HTML files
        assert 'href="https://feed.bhatia.dev/articles.xml"' not in content

    def test_index_html_url_has_no_double_slash(self, tmp_path):
        """index.html URLs should not have double slashes from base_url."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds/")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="https://example.com/feeds/podcast.xml"' in content
        assert "//podcast.xml" not in content

    def test_index_html_correct_structure(self, tmp_path):
        """index.html should have correct HTML structure (DOCTYPE, html, head, body)."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "<html>" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "</head>" in content
        assert "<body>" in content
        assert "</body>" in content

    def test_index_html_lists_feeds_from_root_and_subfolders(self, tmp_path):
        """index.html should list feeds from both root and subfolders."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # Root feeds
        assert ">podcast.xml<" in content
        assert ">articles.xml<" in content
        # Subfolder feeds
        assert ">Course1/podcast.xml<" in content
        assert ">Course1/articles.xml<" in content

    def test_index_html_podcast_feeds_sorted(self, tmp_path):
        """Podcast feeds in index.html should be sorted by relative path."""
        write_metadata_file(tmp_path, "Zeta/lecture.mp4")
        write_feed_json(tmp_path, "Zeta", "Zeta", "Desc")
        write_metadata_file(tmp_path, "Alpha/lecture.mp4")
        write_feed_json(tmp_path, "Alpha", "Alpha", "Desc")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        alpha_pos = content.find("Alpha/podcast.xml")
        zeta_pos = content.find("Zeta/podcast.xml")
        assert alpha_pos < zeta_pos

    def test_index_html_articles_feeds_sorted(self, tmp_path):
        """Articles feeds in index.html should be sorted by relative path."""
        write_metadata_file(tmp_path, "Zeta/doc.pdf")
        write_feed_json(tmp_path, "Zeta", "Zeta", "Desc")
        write_metadata_file(tmp_path, "Alpha/doc.pdf")
        write_feed_json(tmp_path, "Alpha", "Alpha", "Desc")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        alpha_pos = content.find("Alpha/articles.xml")
        zeta_pos = content.find("Zeta/articles.xml")
        assert alpha_pos < zeta_pos
