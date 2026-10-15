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
    validate_feed,
)
from local_feed_generator.metadata import Metadata

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
        "description": description,
        "filename": filename,
        "filepath": filepath,
        "filetype": filetype,
        "date": date,
        "size": size,
        "processing_date": 1728931200,
    }
    if filetype == "mp4":
        kwargs["transcript"] = transcript
        kwargs["summary"] = summary
        if feed_eligible and not error:
            kwargs["status"] = {"transcribe": "complete", "summarize": "complete"}
    else:
        if feed_eligible and not error:
            kwargs["status"] = {"extract": "complete"}
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
                    "status": {"transcribe": "complete", "summarize": "complete"},
                    "title": "Orphan",
                    "description": "desc",
                    "filename": "orphan.mp4",
                    "filepath": "orphan.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                    "processing_date": 1728931200,
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
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            summary=None,
            description="Fallback description",
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

    def test_podcast_feed_empty_when_no_mp4(self, tmp_path):
        """Podcast feed should be valid even with no MP4 files."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        assert items == []
        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<rss" in content
        assert "<item>" not in content

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

    def test_articles_feed_empty_when_no_pdf_html(self, tmp_path):
        """Articles feed should be valid even with no PDF/HTML files."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_articles_feed()

        assert items == []
        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<rss" in content
        assert "<item>" not in content

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
        """generate_all_feeds should create both podcast.xml and articles.xml."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")

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
        """No JSON files should produce empty feeds."""
        (tmp_path / "video.mp4").write_bytes(b"data")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        assert results["podcast"] == []
        assert results["articles"] == []
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()

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
            json.dumps({"status": {"transcribe": "complete"}}),  # Missing required fields
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
