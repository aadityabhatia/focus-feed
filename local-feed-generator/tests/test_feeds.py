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
import logging
from pathlib import Path
from typing import Any
from unittest import mock

from local_feed_generator.feeds import (
    FeedGenerator,
    FeedValidationResult,
    _get_summarization_attribution,
    _markdown_to_html,
    validate_feed,
)
from local_feed_generator.metadata import Metadata, StepMap
import pytest

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
    date: int | None = 1728931200,
    publication_date: float | None = None,
    hash: str | None = None,
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
        publication_date: Publication date from document metadata (seconds).
            If None, the date field is used for pubDate in articles feed.
        hash: SHA256 hash string (optional; used for guid generation).

    Returns:
        A validated Metadata instance.
    """
    filename = Path(filepath).name
    kwargs: dict[str, Any] = {
        "title": title or filename,
        "filename": filename,
        "filepath": filepath,
        "filetype": filetype,
        "date": date,
        "size": size,
    }
    if hash is not None:
        kwargs["hash"] = hash
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
        if summary is not None:
            kwargs["summary"] = summary
        if feed_eligible and not error:
            kwargs["extract"] = StepMap(
                status="complete",
                input_size=size,
                output_size=len(description.encode("utf-8")) if description else 0,
            )
            kwargs["summarize"] = StepMap(
                status="complete",
                input_size=len(description.encode("utf-8")) if description else 0,
                output_size=len((summary or description).encode("utf-8"))
                if (summary or description)
                else 0,
            )
    if error:
        kwargs["error"] = error
    if publication_date is not None:
        kwargs["publication_date"] = publication_date
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

    @pytest.mark.parametrize(
        "data_dir_arg,base_url",
        [
            (None, "https://example.com/feeds"),
            ("https://example.com", "https://example.com"),
            (None, "https://example.com/feeds/"),  # trailing slash stripped
        ],
    )
    def test_init_with_args(self, tmp_path, data_dir_arg, base_url):
        """FeedGenerator should accept explicit data_dir and base_url; strip trailing slash."""
        dir_arg = tmp_path if data_dir_arg is None else str(tmp_path)
        fg = FeedGenerator(data_dir=dir_arg, base_url=base_url)
        assert fg.data_dir == tmp_path.resolve()
        assert fg.data_dir.is_absolute()
        expected_url = base_url.rstrip("/") if base_url.endswith("/") else base_url
        assert fg.base_url == expected_url

    def test_init_stores_data_dir_basename(self, tmp_path):
        """FeedGenerator should store the data_dir basename for URL construction."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com")
        assert fg.data_dir_basename == "data"

    def test_init_data_dir_basename_for_root(self, tmp_path):
        """FeedGenerator should store empty basename when data_dir is root."""
        fg = FeedGenerator(data_dir=Path("/"), base_url="https://example.com")
        assert fg.data_dir_basename == ""

    def test_init_data_dir_is_required(self):
        """FeedGenerator should raise TypeError when data_dir is not provided."""
        with pytest.raises(TypeError):
            FeedGenerator()  # type: ignore[call-arg]

    def test_init_data_dir_none_raises_type_error(self):
        """FeedGenerator should raise TypeError when data_dir is explicitly None."""
        with pytest.raises(TypeError):
            FeedGenerator(data_dir=None, base_url="https://example.com")  # type: ignore[arg-type]

    def test_init_nonexistent_data_dir_logs_critical(self, tmp_path, caplog):
        """FeedGenerator should log CRITICAL when data_dir does not exist."""
        nonexistent = tmp_path / "nonexistent"
        with caplog.at_level(logging.CRITICAL, logger="local_feed_generator"):
            FeedGenerator(data_dir=nonexistent, base_url="https://example.com")
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) == 1
        assert "Data directory does not exist" in critical_records[0].message


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

    def test_feed_json_not_treated_as_orphaned(self, tmp_path):
        """feed.json is a configuration file and should not be flagged as orphaned."""
        (tmp_path / "feed.json").write_text('{"title": "My Feed"}', encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        result = fg._discover_metadata_files()

        # feed.json is config, not metadata — it should not appear in results
        # and should not produce an orphaned JSON warning.
        assert result == []

    def test_feed_json_not_parsed_as_metadata(self, tmp_path, caplog):
        """feed.json should be excluded from metadata discovery entirely.

        It must not produce a "Failed to parse metadata" warning, since it is
        a feed configuration file, not a per-source metadata file.
        """
        (tmp_path / "feed.json").write_text('{"title": "My Feed"}', encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        with caplog.at_level(logging.WARNING):
            result = fg._discover_metadata_files()

        assert result == []
        assert not any(
            "Failed to parse" in record.message
            for record in caplog.records
            if record.levelno >= 30  # WARNING
        )

    def test_feed_json_skipped_alongside_valid_metadata(self, tmp_path, caplog):
        """feed.json should be skipped without warnings even when valid metadata exists."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            '{"title": "My Feed", "description": "desc"}', encoding="utf-8"
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com")
        with caplog.at_level(logging.WARNING):
            result = fg._discover_metadata_files()

        assert len(result) == 1
        assert result[0].filepath == "lecture.mp4"
        assert not any(
            "Failed to parse" in record.message
            for record in caplog.records
            if record.levelno >= 30  # WARNING
        )


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

    @pytest.mark.parametrize(
        "filepath,expected_key",
        [
            ("videos/lecture.mp4", ("videos", "lecture.mp4")),
            ("lecture.mp4", ("", "lecture.mp4")),
        ],
    )
    def test_sort_key_method(self, filepath, expected_key):
        """_sort_key should return (folder_name, filename) tuple."""
        metadata = make_metadata(filepath=filepath)
        key = FeedGenerator._sort_key(metadata)
        assert key == expected_key


# ---------------------------------------------------------------------------
# URL calculation
# ---------------------------------------------------------------------------


class TestUrlCalculation:
    """Tests for absolute URL calculation."""

    @pytest.mark.parametrize(
        "base_url,filepath,expected",
        [
            (
                "https://example.com/feeds",
                "videos/lecture.mp4",
                "https://example.com/feeds/videos/lecture.mp4",
            ),
            ("https://example.com", "doc.pdf", "https://example.com/doc.pdf"),
            (
                "https://example.com/feeds",
                "Course1/Sub/lesson.mp4",
                "https://example.com/feeds/Course1/Sub/lesson.mp4",
            ),
        ],
    )
    def test_absolute_url(self, tmp_path, base_url, filepath, expected):
        """Absolute URL should be BASE_URL + filepath (no data_dir basename)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fg = FeedGenerator(data_dir=data_dir, base_url=base_url)
        url = fg._absolute_url(filepath)
        assert url == expected

    def test_absolute_url_no_data_dir_basename(self, tmp_path):
        """URL should NOT include the data_dir basename in the path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com")
        url = fg._absolute_url("videos/lecture.mp4")
        assert url == "https://example.com/videos/lecture.mp4"

    def test_absolute_url_for_root(self, tmp_path):
        """When data_dir is root, URL should still be BASE_URL + filepath."""
        fg = FeedGenerator(data_dir=Path("/"), base_url="https://example.com")
        url = fg._absolute_url("doc.pdf")
        assert url == "https://example.com/doc.pdf"


# ---------------------------------------------------------------------------
# Podcast feed generation
# ---------------------------------------------------------------------------


class TestPodcastFeedGeneration:
    """Tests for generate_podcast_feed()."""

    @pytest.mark.parametrize(
        "xml_name,method,filetype,filepath",
        [
            ("podcast.xml", "generate_podcast_feed", "mp4", "videos/lecture.mp4"),
            ("articles.xml", "generate_articles_feed", "pdf", "documents/doc.pdf"),
        ],
    )
    def test_feed_file_created(self, tmp_path, xml_name, method, filetype, filepath):
        """Feed XML should be written to data_dir."""
        write_metadata_file(tmp_path, filepath, filetype=filetype)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        getattr(fg, method)()

        assert (tmp_path / xml_name).exists()

    @pytest.mark.parametrize(
        "method,filetype,filepath",
        [
            ("generate_podcast_feed", "mp4", "videos/lecture1.mp4"),
            ("generate_articles_feed", "pdf", "documents/doc.pdf"),
        ],
    )
    def test_feed_includes_correct_items(self, tmp_path, method, filetype, filepath):
        """Feed should include only items of the correct filetype."""
        write_metadata_file(tmp_path, filepath, filetype=filetype)
        # Add a file of the other type in a different folder
        other_type = "pdf" if filetype == "mp4" else "mp4"
        write_metadata_file(tmp_path, f"other/{other_type}_file.{other_type}", filetype=other_type)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = getattr(fg, method)()

        assert len(items) == 1
        assert all(m.filetype == filetype for m in items)

    @pytest.mark.parametrize(
        "method,xml_name,filetype,filepath,title",
        [
            ("generate_podcast_feed", "podcast.xml", "mp4", "videos/lecture.mp4", "My Lecture"),
            ("generate_articles_feed", "articles.xml", "pdf", "documents/doc.pdf", "My Document"),
        ],
    )
    def test_feed_item_has_title(self, tmp_path, method, xml_name, filetype, filepath, title):
        """Each feed item should have a title (with module prefix applied)."""
        write_metadata_file(tmp_path, filepath, filetype=filetype, title=title)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        getattr(fg, method)()

        content = parse_rss_file(tmp_path / xml_name)
        # Module prefix is applied to the final title after precedence resolution.
        # The parent directory name is prepended if the title doesn't already
        # start with the folder name (case-insensitive).
        parent_dir = Path(filepath).parent.name
        title_normalized = title.replace("_", " ").lower()
        if title_normalized.startswith(parent_dir.lower()):
            expected_title = title
        else:
            expected_title = f"{parent_dir}: {title}"
        assert f"<title>{expected_title}</title>" in content

    def test_podcast_feed_item_description_from_summary(self, tmp_path):
        """Podcast feed description should come from summary field (FR-014)."""
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

    def test_podcast_feed_description_fallback_to_description(self, tmp_path):
        """Podcast description should fall back to description field if summary is None."""
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
        """Each podcast item should have an enclosure with URL, type, and length (FR-018)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, "videos/lecture.mp4", size=12345678)

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(data_dir / "podcast.xml")
        assert "<enclosure" in content
        assert 'url="https://example.com/feeds/videos/lecture.mp4"' in content
        assert 'length="12345678"' in content
        assert 'type="audio/mp4"' in content

    @pytest.mark.parametrize(
        "field,substring",
        [
            ("link", "https://example.com/feeds/videos/lecture.mp4"),
            ("guid", "https://example.com/feeds/videos/lecture.mp4"),
            ("pubdate", "<pubDate>"),
        ],
    )
    def test_podcast_feed_item_has_required_fields(self, tmp_path, field, substring):
        """Each podcast item should have link, guid, and pubDate (feedgen built-in)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(data_dir / "podcast.xml")
        if field == "link":
            assert f"<link>{substring}</link>" in content
        elif field == "guid":
            assert "<guid" in content
            assert substring in content
        elif field == "pubdate":
            assert substring in content

    def test_podcast_feed_items_sorted(self, tmp_path):
        """Podcast feed items should be sorted by folder then filename (FR-021)."""
        write_metadata_file(tmp_path, "zebra/last.mp4")
        write_metadata_file(tmp_path, "alpha/first.mp4")
        write_metadata_file(tmp_path, "alpha/middle.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        filepaths = [m.filepath for m in items]
        assert filepaths == ["alpha/first.mp4", "alpha/middle.mp4", "zebra/last.mp4"]

    @pytest.mark.parametrize(
        "method,xml_name,good_filetype,bad_filetype,good_path,bad_path",
        [
            (
                "generate_podcast_feed",
                "podcast.xml",
                "mp4",
                "pdf",
                "videos/good.mp4",
                "videos/bad.mp4",
            ),
            (
                "generate_articles_feed",
                "articles.xml",
                "pdf",
                "mp4",
                "documents/good.pdf",
                "documents/bad.pdf",
            ),
        ],
    )
    def test_feed_excludes_error_entries(
        self, tmp_path, method, xml_name, good_filetype, bad_filetype, good_path, bad_path
    ):
        """Error entries should not appear in feeds."""
        write_metadata_file(tmp_path, good_path, filetype=good_filetype)
        write_metadata_file(tmp_path, bad_path, filetype=bad_filetype, error="Failed")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        getattr(fg, method)()

        content = parse_rss_file(tmp_path / xml_name)
        assert content.count("<item>") == 1
        assert Path(good_path).name in content
        assert Path(bad_path).name not in content

    def test_podcast_feed_has_podcast_extension(self, tmp_path):
        """Podcast feed should include itunes namespace (podcast RSS spec)."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "itunes" in content

    @pytest.mark.parametrize(
        "method,xml_name,exists_filetype,exists_path,not_exists_path",
        [
            (
                "generate_podcast_feed",
                "podcast.xml",
                "pdf",
                "documents/doc.pdf",
                "videos/lecture.mp4",
            ),
            (
                "generate_articles_feed",
                "articles.xml",
                "mp4",
                "videos/lecture.mp4",
                "documents/doc.pdf",
            ),
        ],
    )
    def test_feed_not_created_when_no_applicable_files(
        self, tmp_path, method, xml_name, exists_filetype, exists_path, not_exists_path
    ):
        """Feed should not be created when no applicable file types are present."""
        write_metadata_file(tmp_path, exists_path, filetype=exists_filetype)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = getattr(fg, method)()

        assert items == []
        assert not (tmp_path / xml_name).exists()

    @pytest.mark.parametrize(
        "method,xml_name,filetype,filepath",
        [
            ("generate_podcast_feed", "podcast.xml", "mp4", "videos/lecture.mp4"),
            ("generate_articles_feed", "articles.xml", "pdf", "documents/doc.pdf"),
        ],
    )
    def test_feed_overwrites_existing(self, tmp_path, method, xml_name, filetype, filepath):
        """Regenerating feed should overwrite existing file."""
        write_metadata_file(tmp_path, filepath, filetype=filetype)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        getattr(fg, method)()

        # Add another file
        if filetype == "mp4":
            write_metadata_file(tmp_path, "videos/lecture2.mp4", filetype="mp4")
        else:
            write_metadata_file(tmp_path, "documents/doc2.html", filetype="html")

        getattr(fg, method)()

        content = parse_rss_file(tmp_path / xml_name)
        assert content.count("<item>") == 2


# ---------------------------------------------------------------------------
# Articles feed generation
# ---------------------------------------------------------------------------


class TestArticlesFeedGeneration:
    """Tests for generate_articles_feed()."""

    def test_articles_feed_file_created(self, tmp_path):
        """articles.xml should be written to data_dir."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        assert (tmp_path / "articles.xml").exists()

    def test_articles_feed_includes_pdf_and_html(self, tmp_path):
        """Articles feed should include all PDF and HTML files."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        write_metadata_file(tmp_path, "documents/article.html", filetype="html")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_articles_feed()

        assert len(items) == 2
        filetypes = {m.filetype for m in items}
        assert filetypes == {"pdf", "html"}

    def test_articles_feed_item_has_title(self, tmp_path):
        """Each articles item should have a title (with module prefix applied)."""
        write_metadata_file(tmp_path, "documents/doc.pdf", title="My Document")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # Module prefix "documents" is applied since title "My Document" doesn't
        # start with "documents".
        assert "<title>documents: My Document</title>" in content

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

    def test_articles_feed_description_from_summary(self, tmp_path):
        """Article feed description should use summary field, not content (FR-014)."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            summary="Key takeaways from the document.",
            description="Fallback description",
            transcript="This is the extracted content that should NOT be used.",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # Summary should be used, not content or description
        assert "Key takeaways from the document." in content
        assert "This is the extracted content" not in content
        assert "Fallback description" not in content

    def test_articles_feed_description_fallback_to_description(self, tmp_path):
        """Article feed description should fall back to description when summary is None."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="The fallback description",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "The fallback description" in content

    @pytest.mark.parametrize(
        "field,substring",
        [
            ("link", "https://example.com/feeds/documents/doc.pdf"),
            ("guid", "https://example.com/feeds/documents/doc.pdf"),
            ("pubdate", "<pubDate>"),
        ],
    )
    def test_articles_feed_item_has_required_fields(self, tmp_path, field, substring):
        """Each articles item should have link, guid, and pubDate (feedgen built-in)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, "documents/doc.pdf")

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(data_dir / "articles.xml")
        if field == "link":
            assert f"<link>{substring}</link>" in content
        elif field == "guid":
            assert "<guid" in content
            assert substring in content
        elif field == "pubdate":
            assert substring in content

    def test_articles_feed_items_sorted(self, tmp_path):
        """Articles feed items should be sorted by folder then filename."""
        write_metadata_file(tmp_path, "zebra/last.pdf")
        write_metadata_file(tmp_path, "alpha/first.html")
        write_metadata_file(tmp_path, "alpha/middle.pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_articles_feed()

        filepaths = [m.filepath for m in items]
        assert filepaths == ["alpha/first.html", "alpha/middle.pdf", "zebra/last.pdf"]

    def test_articles_feed_pubdate_uses_publication_date(self, tmp_path):
        """Articles feed pubDate should use publication_date when available."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            date=1728931200,  # 2024-10-14
            publication_date=1719200000.0,  # 2024-06-24
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # pubDate should reflect publication_date (June 2024), not date (Oct 2024)
        from datetime import UTC, datetime

        expected = datetime.fromtimestamp(1719200000.0, tz=UTC).strftime("%a, %d %b %Y")
        assert expected in content

    def test_articles_feed_pubdate_falls_back_to_date(self, tmp_path):
        """Articles feed pubDate should fall back to date when publication_date is None."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            date=1728931200,  # 2024-10-14
            publication_date=None,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # pubDate should reflect date (Oct 2024) since publication_date is None
        from datetime import UTC, datetime

        expected = datetime.fromtimestamp(1728931200, tz=UTC).strftime("%a, %d %b %Y")
        assert expected in content

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

    @pytest.mark.parametrize(
        "failing_feed,working_feed",
        [
            ("generate_podcast_feed", "articles"),
            ("generate_articles_feed", "podcast"),
        ],
    )
    def test_error_in_one_feed_does_not_block_other(self, tmp_path, failing_feed, working_feed):
        """If one feed generation fails, the other should still be generated."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        with mock.patch.object(
            fg,
            failing_feed,
            side_effect=RuntimeError("Feed generation failed"),
        ):
            results = fg.generate_all_feeds()

        # The working feed should still be generated
        assert len(results[working_feed]) == 1
        assert results["podcast" if failing_feed == "generate_podcast_feed" else "articles"] == []

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
    """Tests for proper XML escaping of special characters.

    These tests verify that our feed generator passes content with special
    characters through to feedgen without crashing. feedgen handles the actual
    XML escaping; we only need to ensure our code doesn't interfere.
    """

    @pytest.mark.parametrize(
        "filetype,xml_name,field,content",
        [
            ("mp4", "podcast.xml", "title", 'Lecture <with> & "quotes"'),
            ("mp4", "podcast.xml", "summary", "Summary with <tags> & ampersand"),
            ("pdf", "articles.xml", "description", "Description with <html> & special chars"),
        ],
    )
    def test_special_characters_handled_by_feedgen(
        self, tmp_path, filetype, xml_name, field, content
    ):
        """Content with XML special characters should be handled without crashing."""
        kwargs = {field: content} if field != "title" else {"title": content}
        write_metadata_file(tmp_path, f"file.{filetype}", filetype=filetype, **kwargs)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        if filetype == "mp4":
            fg.generate_podcast_feed()
        else:
            fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / xml_name)
        assert "<item>" in content


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
        (tmp_path / "bad.mp4").write_bytes(b"data")
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

    @pytest.mark.parametrize(
        "filetype,xml_name",
        [
            ("mp4", "podcast.xml"),
            ("pdf", "articles.xml"),
        ],
    )
    def test_feed_links_to_self(self, tmp_path, filetype, xml_name):
        """Feed should include a self-referencing link."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, f"file.{filetype}", filetype=filetype)

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        if filetype == "mp4":
            fg.generate_podcast_feed()
        else:
            fg.generate_articles_feed()

        content = parse_rss_file(data_dir / xml_name)
        assert 'rel="self"' in content
        assert f"https://example.com/feeds/{xml_name}" in content


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

# An Atom feed (wrong root element for RSS validation).
_ATOM_FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Not RSS</title>
</feed>
"""

# An RSS feed with wrong version (0.91 instead of 2.0).
_WRONG_VERSION_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="0.91">
  <channel>
    <title>Old RSS</title>
    <link>https://example.com</link>
    <description>Old</description>
  </channel>
</rss>
"""

# An RSS feed with no <channel> element.
_NO_CHANNEL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
</rss>
"""

# Malformed XML (unclosed tags).
_MALFORMED_XML = "<rss><channel><item><title>Unclosed</item></channel>"

# XML with unescaped special characters.
_UNESCAPED_XML = """\
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


# ---------------------------------------------------------------------------
# Feed validation (validate_feed function)
# ---------------------------------------------------------------------------


class TestValidateFeed:
    """Tests for the standalone validate_feed() function."""

    @pytest.mark.parametrize(
        "xml,feed_name,is_podcast",
        [
            (_VALID_PODCAST_XML, "podcast.xml", True),
            (_VALID_ARTICLES_XML, "articles.xml", False),
        ],
    )
    def test_valid_feed_passes(self, xml, feed_name, is_podcast):
        """A well-formed feed with all required elements should pass."""
        result = validate_feed(xml, feed_name=feed_name, is_podcast=is_podcast)
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

    @pytest.mark.parametrize(
        "xml,expected_error_substring",
        [
            (_MALFORMED_XML, "not well-formed"),
            (_UNESCAPED_XML, "not well-formed"),
            (_ATOM_FEED_XML, "expected <rss>"),
            (_WRONG_VERSION_XML, "expected '2.0'"),
            (_NO_CHANNEL_XML, "Missing <channel>"),
        ],
    )
    def test_feed_structure_errors(self, xml, expected_error_substring):
        """Invalid feed structure should produce appropriate errors."""
        result = validate_feed(xml, feed_name="test.xml")
        assert not result.is_valid
        assert any(expected_error_substring in e for e in result.errors)

    @pytest.mark.parametrize(
        "element,error_substring",
        [
            ("<title>Test Item</title>", "missing or empty <title>"),
            ("<link>https://example.com/test.mp4</link>", "missing or empty <link>"),
            ("<guid>https://example.com/test.mp4</guid>", "missing or empty <guid>"),
            ("<pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>", "missing <pubDate>"),
        ],
    )
    def test_item_missing_required_element_fails(self, element, error_substring):
        """An item missing a required element should produce an error."""
        xml = _VALID_PODCAST_XML.replace(f"      {element}\n", "")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any(error_substring in e for e in result.errors)

    @pytest.mark.parametrize(
        "old_element,new_element,error_substring",
        [
            ("<title>Test Item</title>", "<title></title>", "missing or empty <title>"),
            (
                "<link>https://example.com/test.mp4</link>",
                "<link></link>",
                "missing or empty <link>",
            ),
            (
                "<guid>https://example.com/test.mp4</guid>",
                "<guid></guid>",
                "missing or empty <guid>",
            ),
        ],
    )
    def test_item_empty_required_element_fails(self, old_element, new_element, error_substring):
        """An item with an empty required element should produce an error."""
        xml = _VALID_PODCAST_XML.replace(f"      {old_element}\n", f"      {new_element}\n")
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any(error_substring in e for e in result.errors)

    def test_item_with_dc_date_passes(self):
        """An item with <dc:date> instead of <pubDate> should pass."""
        xml = _VALID_PODCAST_XML.replace(
            "      <pubDate>Wed, 14 Oct 2026 12:00:00 +0000</pubDate>\n",
            '      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "2026-10-14T12:00:00Z</dc:date>\n",
        )
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert result.is_valid

    @pytest.mark.parametrize(
        "xml_modification,error_substring",
        [
            (
                lambda xml: xml.replace(
                    '      <enclosure url="https://example.com/test.mp4" '
                    'length="12345" type="audio/mp4"/>\n',
                    "",
                ),
                "missing <enclosure>",
            ),
            (
                lambda xml: xml.replace(' url="https://example.com/test.mp4"', ""),
                "enclosure missing 'url'",
            ),
            (
                lambda xml: xml.replace(' length="12345"', ""),
                "enclosure missing 'length'",
            ),
            (
                lambda xml: xml.replace(' type="audio/mp4"', ' type="audio/mpeg"'),
                "expected 'audio/mp4'",
            ),
            (
                lambda xml: xml.replace(' length="12345"', ' length="not-a-number"'),
                "not a valid integer",
            ),
        ],
    )
    def test_podcast_enclosure_issues(self, xml_modification, error_substring):
        """Podcasts with enclosure issues should produce errors."""
        xml = xml_modification(_VALID_PODCAST_XML)
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert not result.is_valid
        assert any(error_substring in e for e in result.errors)

    @pytest.mark.parametrize(
        "xml,is_podcast,should_be_valid",
        [
            # Podcast XML validated as articles: enclosure present but not required
            (_VALID_PODCAST_XML, False, True),
            # Articles XML (no enclosure) validated as articles: should pass
            (_VALID_ARTICLES_XML, False, True),
            # Podcast XML validated as podcast: enclosure required, present -> valid
            (_VALID_PODCAST_XML, True, True),
        ],
    )
    def test_articles_feed_skips_enclosure_check(self, xml, is_podcast, should_be_valid):
        """Articles feed validation should skip enclosure checks."""
        result = validate_feed(xml, feed_name="test.xml", is_podcast=is_podcast)
        assert result.is_valid == should_be_valid

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

    @pytest.mark.parametrize(
        "valid,kwargs,expected_valid,expected_errors",
        [
            (True, {"feed_name": "test.xml"}, True, []),
            (
                False,
                {"feed_name": "test.xml", "valid": False, "errors": ["Some error"]},
                False,
                ["Some error"],
            ),
            (True, {"feed_name": "test.xml", "warnings": ["Some warning"]}, True, []),
        ],
    )
    def test_feed_validation_result_dataclass(self, valid, kwargs, expected_valid, expected_errors):
        """FeedValidationResult should have correct default values and behavior."""
        result = FeedValidationResult(**kwargs)
        assert result.is_valid == expected_valid
        assert result.errors == expected_errors


class TestValidateOutput:
    """Tests for FeedGenerator.validate_output()."""

    def test_validate_output_both_feeds_valid(self, tmp_path):
        """validate_output() should return valid results for generated feeds."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
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

    @pytest.mark.parametrize("feed_name", ["podcast", "articles"])
    def test_validate_output_missing_feed_file(self, tmp_path, feed_name):
        """validate_output() should report error when a feed XML file is missing."""
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.validate_output()

        assert not results[feed_name].is_valid
        assert any("does not exist" in e for e in results[feed_name].errors)

    @pytest.mark.parametrize(
        "filetype,generate_method",
        [
            ("mp4", "generate_podcast_feed"),
            ("pdf", "generate_articles_feed"),
        ],
    )
    def test_validate_output_enclosure_behavior(self, tmp_path, filetype, generate_method):
        """validate_output() should enforce enclosure checks on podcast but not articles."""
        write_metadata_file(tmp_path, f"test.{filetype}", filetype=filetype)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        getattr(fg, generate_method)()

        results = fg.validate_output()
        assert results["podcast" if filetype == "mp4" else "articles"].is_valid
        assert all("enclosure" not in e for e in results["podcast"].errors)


class TestGenerateAllFeedsWithValidation:
    """Tests for the optional validation parameter in generate_all_feeds()."""

    def _setup_feeds(self, tmp_path):
        """Create root feed.json and metadata files for validation tests."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

    def test_generate_all_feeds_with_validation_logs_no_errors(self, tmp_path, caplog):
        """generate_all_feeds(validate=True) should log validation success."""
        self._setup_feeds(tmp_path)
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        with caplog.at_level("INFO"):
            fg.generate_all_feeds(validate=True)

        assert any("Feed validation passed" in record.message for record in caplog.records)

    @pytest.mark.parametrize("validate_arg", [False, None])
    def test_generate_all_feeds_no_validation_no_logging(self, tmp_path, caplog, validate_arg):
        """generate_all_feeds without validate=True should not log validation results."""
        self._setup_feeds(tmp_path)
        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        with caplog.at_level("INFO"):
            if validate_arg is None:
                fg.generate_all_feeds()  # default: no validation
            else:
                fg.generate_all_feeds(validate=False)

        assert not any("Feed validation" in record.message for record in caplog.records)


class TestMarkdownToHtml:
    """Tests for the _markdown_to_html() function."""

    @pytest.mark.parametrize(
        "md,expected",
        [
            (None, ""),
            ("", ""),
            # Headers
            ("# Heading 1", "<h1>Heading 1</h1>"),
            ("## Heading 2", "<h2>Heading 2</h2>"),
            ("### Heading 3", "<h3>Heading 3</h3>"),
            ("# Header with **bold**", "<h1>Header with <strong>bold</strong></h1>"),
            # Bold — single-paragraph output has outer <p> stripped
            ("**bold**", "<strong>bold</strong>"),
            ("This is **bold** text", "This is <strong>bold</strong> text"),
            # Italic — single-paragraph output has outer <p> stripped
            ("*italic*", "<em>italic</em>"),
            ("_italic_", "<em>italic</em>"),
            # Paragraphs — single-paragraph output has outer <p> stripped
            ("Some text", "Some text"),
        ],
    )
    def test_markdown_conversion(self, md, expected):
        """Parameterized test for markdown-to-HTML conversion."""
        assert _markdown_to_html(md) == expected

    @pytest.mark.parametrize(
        "md,expected_substring,should_not_contain",
        [
            ("[example](https://example.com)", '<a href="https://example.com">example</a>', None),
            (
                "[email](mailto:test@example.com)",
                '<a href="mailto:test@example.com">email</a>',
                None,
            ),
            ("[page](/path/to/page)", '<a href="/path/to/page">page</a>', None),
            ("[bad](javascript:alert(1))", None, "javascript:"),
        ],
    )
    def test_link_conversion(self, md, expected_substring, should_not_contain):
        """Parameterized test for link conversion and XSS protection."""
        result = _markdown_to_html(md)
        if expected_substring:
            assert expected_substring in result
        if should_not_contain:
            assert should_not_contain not in result
            assert "<a " not in result
            # Link text should still appear even for unsafe URLs
            assert "safe text" in result or "bad" in result

    @pytest.mark.parametrize(
        "md,expected_substrings",
        [
            ("- item 1\n- item 2", ["<ul>", "<li>item 1</li>", "<li>item 2</li>", "</ul>"]),
            ("1. first\n2. second", ["<ol>", "<li>first</li>", "<li>second</li>", "</ol>"]),
        ],
    )
    def test_list_conversion(self, md, expected_substrings):
        """Parameterized test for list conversion."""
        result = _markdown_to_html(md)
        for substring in expected_substrings:
            assert substring in result

    @pytest.mark.parametrize(
        "md,expected_substrings",
        [
            ("- **bold** item", ["<li><strong>bold</strong> item</li>"]),
            ("Para 1\n\nPara 2", ["<p>Para 1</p>", "<p>Para 2</p>"]),
            ("Line 1\nLine 2", ["<br>", "Line 1", "Line 2"]),
        ],
    )
    def test_inline_and_paragraph_conversions(self, md, expected_substrings):
        """Test inline markdown, paragraph separation, and line breaks."""
        result = _markdown_to_html(md)
        for substring in expected_substrings:
            assert substring in result

    @pytest.mark.parametrize(
        "md,expected_substrings",
        [
            ("<script>alert(1)</script>", ["&lt;script&gt;"]),
            ("a & b", ["a &amp; b"]),
            ("a < b", ["a &lt; b"]),
            ("a > b", ["a &gt; b"]),
            # The markdown library does not escape double quotes (only & < >).
            # Quotes are safe in HTML body content and are XML-escaped by
            # feedgen when the description is embedded in the RSS output.
            ('say "hello"', ['say "hello"']),
        ],
    )
    def test_html_escaping(self, md, expected_substrings):
        """Parameterized test for HTML special character escaping."""
        result = _markdown_to_html(md)
        for substring in expected_substrings:
            assert substring in result
        # Verify raw HTML tags are not present (escaped)
        assert "<script>" not in result
        assert "<script" not in result

    @pytest.mark.parametrize(
        "md,expected",
        [
            # Single paragraph: outer <p> wrapper should be stripped
            ("Hello world", "Hello world"),
            ("Some text", "Some text"),
            ("**bold** text", "<strong>bold</strong> text"),
            ("*italic* text", "<em>italic</em> text"),
            # Headers: no <p> wrapper to strip
            ("# Heading", "<h1>Heading</h1>"),
            # Lists: no <p> wrapper to strip
            ("- item", "<ul>\n<li>item</li>\n</ul>"),
        ],
    )
    def test_p_wrapper_stripped_for_single_paragraph(self, md, expected):
        """Single-paragraph output should not have outer <p> wrapper."""
        assert _markdown_to_html(md) == expected

    @pytest.mark.parametrize(
        "md,expected_substrings",
        [
            # Multiple paragraphs: <p> tags should be preserved
            ("Para 1\n\nPara 2", ["<p>Para 1</p>", "<p>Para 2</p>"]),
            (
                "First paragraph.\n\nSecond paragraph.",
                ["<p>First paragraph.</p>", "<p>Second paragraph.</p>"],
            ),
        ],
    )
    def test_p_tags_preserved_for_multiple_paragraphs(self, md, expected_substrings):
        """Multiple-paragraph output should preserve <p> tags."""
        result = _markdown_to_html(md)
        for substring in expected_substrings:
            assert substring in result

    def test_p_wrapper_not_stripped_for_mixed_content(self):
        """Mixed content (header + paragraph) should preserve <p> tags."""
        result = _markdown_to_html("# Header\n\nParagraph")
        assert "<h1>Header</h1>" in result
        assert "<p>Paragraph</p>" in result

    def test_p_wrapper_stripped_for_inline_markdown(self):
        """Inline markdown in a single paragraph should have outer <p> stripped."""
        result = _markdown_to_html("**bold** and *italic*")
        assert result == "<strong>bold</strong> and <em>italic</em>"
        assert "<p>" not in result

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
# Per-folder feed generation via generate_all_feeds()
# ---------------------------------------------------------------------------


def write_feed_json(data_dir: Path, folder: str, title: str, description: str) -> Path:
    """Write a feed.json file to a subfolder within data_dir."""
    folder_path = data_dir / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    feed_json_path = folder_path / "feed.json"
    feed_json_path.write_text(
        json.dumps({"title": title, "description": description}),
        encoding="utf-8",
    )
    return feed_json_path


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

    def test_stale_root_feeds_removed_without_feed_json(self, tmp_path):
        """Stale root-level feeds should be removed when no feed.json exists at root.

        If a previous run generated root-level podcast.xml and articles.xml,
        a subsequent run without a root feed.json should clean them up so
        they don't appear in the index.html.
        """
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        # Create stale root-level feed files (simulating a previous run)
        (tmp_path / "podcast.xml").write_text("<rss></rss>", encoding="utf-8")
        (tmp_path / "articles.xml").write_text("<rss></rss>", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        # Root feeds should not be generated (no feed.json)
        assert results["podcast"] == []
        assert results["articles"] == []
        # Stale root-level feeds should be removed
        assert not (tmp_path / "podcast.xml").exists()
        assert not (tmp_path / "articles.xml").exists()

    def test_stale_root_feeds_not_removed_when_feed_json_exists(self, tmp_path):
        """Stale root-level feeds should NOT be removed when feed.json exists at root.

        When a feed.json exists, root-level feeds are regenerated (overwritten),
        so stale files are naturally replaced — no explicit cleanup needed.
        """
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )
        # Create stale root-level feed file (simulating a previous run)
        (tmp_path / "podcast.xml").write_text("<rss></rss>", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        results = fg.generate_all_feeds()

        # Root feeds should be regenerated (overwriting stale file)
        assert len(results["podcast"]) == 1
        assert (tmp_path / "podcast.xml").exists()
        # The stale file should have been overwritten with valid content
        content = (tmp_path / "podcast.xml").read_text(encoding="utf-8")
        assert "<item>" in content

    @pytest.mark.parametrize(
        "filetype,expected_podcast,expected_articles,folder",
        [
            ("pdf", False, True, ""),  # PDF only: podcast skipped, articles generated
            ("mp4", True, False, ""),  # MP4 only: podcast generated, articles skipped
            ("pdf", False, True, "Course1/"),  # Per-folder PDF: podcast skipped, articles generated
            ("mp4", True, False, "Course1/"),  # Per-folder MP4: podcast generated, articles skipped
        ],
    )
    def test_conditional_feed_generation(
        self, tmp_path, filetype, expected_podcast, expected_articles, folder
    ):
        """Feeds should only be generated if applicable file types exist (root and per-folder)."""
        write_metadata_file(tmp_path, f"{folder}file.{filetype}", filetype=filetype)
        if folder:
            write_feed_json(tmp_path, folder.rstrip("/"), "Course Feed", "Description")
        else:
            (tmp_path / "feed.json").write_text(
                json.dumps({"title": "Root Feed", "description": "Root Description"}),
                encoding="utf-8",
            )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        feed_dir = tmp_path if not folder else tmp_path / folder.rstrip("/")
        assert (feed_dir / "podcast.xml").exists() == expected_podcast
        assert (feed_dir / "articles.xml").exists() == expected_articles

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

    def test_per_folder_feed_uses_title_from_feed_json(self, tmp_path):
        """Per-folder feed title should come from feed.json."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "My Course Feed", "My Description")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<title>My Course Feed</title>" in content
        assert "<description>My Description</description>" in content

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

    def test_nested_feeds_generate_and_contain_correct_files(self, tmp_path):
        """Nested feed.json files should generate feeds with correct file scope."""
        self._setup_nested_structure(tmp_path)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # All feed files should exist in respective folders
        assert (tmp_path / "podcast.xml").exists()
        assert (tmp_path / "articles.xml").exists()
        assert (tmp_path / "Course1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "articles.xml").exists()
        assert (tmp_path / "Course1" / "Unit1" / "podcast.xml").exists()
        assert (tmp_path / "Course1" / "Unit1" / "articles.xml").exists()

        # Root podcast feed should contain both MP4 files
        root_podcast = parse_rss_file(tmp_path / "podcast.xml")
        assert "Course1/lecture1.mp4" in root_podcast
        assert "Course1/Unit1/lecture2.mp4" in root_podcast

        # Root articles feed should contain both PDF/HTML files
        root_articles = parse_rss_file(tmp_path / "articles.xml")
        assert "Course1/notes.pdf" in root_articles
        assert "Course1/Unit1/article.html" in root_articles

        # Course1 podcast feed should contain both MP4 files
        course1_podcast = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "Course1/lecture1.mp4" in course1_podcast
        assert "Course1/Unit1/lecture2.mp4" in course1_podcast

        # Course1 articles feed should contain both PDF/HTML files
        course1_articles = parse_rss_file(tmp_path / "Course1" / "articles.xml")
        assert "Course1/notes.pdf" in course1_articles
        assert "Course1/Unit1/article.html" in course1_articles

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

    def test_index_html_structure_and_content(self, tmp_path):
        """index.html should contain expected structure, head, sections, and CSS."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        # Structural elements
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "<head>" in content
        assert "<body>" in content
        # Head and sections
        assert "<title>Feed Index</title>" in content
        assert "<h1>Feed Index</h1>" in content
        assert "<h2>Podcast Feeds</h2>" in content
        assert "<h2>Articles Feeds</h2>" in content
        # Meta and CSS
        assert '<meta charset="utf-8">' in content
        assert "font-family: Arial, sans-serif" in content
        assert "color: #333" in content
        assert "list-style: none" in content
        assert "color: #0066cc" in content

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

    @pytest.mark.parametrize(
        "filetype,should_list_podcast,should_list_articles",
        [
            ("mp4", True, False),
            ("pdf", False, True),
        ],
    )
    def test_index_html_only_lists_applicable_feeds(
        self, tmp_path, filetype, should_list_podcast, should_list_articles
    ):
        """index.html should only list feeds that exist based on file types."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, f"file.{filetype}", filetype=filetype)
        (data_dir / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = (data_dir / "index.html").read_text(encoding="utf-8")
        if should_list_podcast:
            assert 'href="https://example.com/feeds/podcast.xml"' in content
        else:
            assert 'href="https://example.com/feeds/podcast.xml"' not in content
        if should_list_articles:
            assert 'href="https://example.com/feeds/articles.xml"' in content
        else:
            assert 'href="https://example.com/feeds/articles.xml"' not in content
        # Both sections should exist even if one feed is empty
        assert "<h2>Podcast Feeds</h2>" in content
        assert "<h2>Articles Feeds</h2>" in content

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

    @pytest.mark.parametrize(
        "base_url,expected_href",
        [
            ("https://example.com/feeds", 'href="https://example.com/feeds/podcast.xml"'),
            ("https://example.com/feeds/", 'href="https://example.com/feeds/podcast.xml"'),
        ],
    )
    def test_index_html_url_uses_base_url_without_double_slash(
        self, tmp_path, base_url, expected_href
    ):
        """index.html links should use the configured BASE_URL without double slashes."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, "videos/lecture.mp4")
        (data_dir / "feed.json").write_text(
            json.dumps({"title": "Root", "description": "Root Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=data_dir, base_url=base_url)
        fg.generate_all_feeds()

        content = (data_dir / "index.html").read_text(encoding="utf-8")
        assert expected_href in content
        # articles.xml should NOT exist since no PDF/HTML files
        assert 'href="https://example.com/feeds/articles.xml"' not in content
        assert "//podcast.xml" not in content

    def test_index_html_lists_feeds_from_root_and_subfolders(self, tmp_path):
        """index.html should list feeds from both root and subfolders."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_metadata_file(tmp_path, "Course1/doc.pdf")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
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


# ---------------------------------------------------------------------------
# Feed link handling (from feed.json vs BASE_URL)
# ---------------------------------------------------------------------------


class TestFeedLinkHandling:
    """Tests for feed-level link derivation from BASE_URL and feed.json."""

    def test_feed_link_defaults_to_base_url_when_no_feed_json(self, tmp_path):
        """Feed-level channel link should default to BASE_URL when no feed.json link is present."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # The channel link (not rel="self") should be BASE_URL
        assert "<link>https://example.com/feeds</link>" in content

    def test_feed_link_uses_feed_json_link_when_present(self, tmp_path):
        """Feed-level channel link should use feed.json link when present."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Root Feed",
                    "description": "Root Description",
                    "link": "https://example.com/custom-homepage",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # The channel link should use the feed.json link, not BASE_URL
        assert "<link>https://example.com/custom-homepage</link>" in content
        assert "<link>https://example.com/feeds</link>" not in content

    def test_feed_link_uses_feed_json_link_for_articles(self, tmp_path):
        """Articles feed should also use feed.json link when present."""
        write_metadata_file(tmp_path, "documents/doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Root Feed",
                    "description": "Root Description",
                    "link": "https://example.com/articles-home",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<link>https://example.com/articles-home</link>" in content

    def test_feed_link_falls_back_to_base_url_when_link_omitted(self, tmp_path):
        """Feed link should fall back to BASE_URL when feed.json has no link field."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")
        # feed.json without a link field
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<link>https://example.com/feeds</link>" in content

    def test_per_folder_feed_link_uses_feed_json_link(self, tmp_path):
        """Per-folder feed should use its feed.json link when present."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4", filetype="mp4")
        write_feed_json_with_link(
            tmp_path, "Course1", "Course 1", "Desc", "https://example.com/course1"
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<link>https://example.com/course1</link>" in content

    def test_per_folder_feed_link_falls_back_to_base_url(self, tmp_path):
        """Per-folder feed should fall back to BASE_URL when no link in feed.json."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4", filetype="mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Desc")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<link>https://example.com/feeds</link>" in content

    def test_item_links_always_use_base_url_plus_filepath(self, tmp_path):
        """Item links should always be BASE_URL + filepath, regardless of feed.json link."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(data_dir, "videos/lecture.mp4", filetype="mp4")
        (data_dir / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Root Feed",
                    "description": "Root Description",
                    "link": "https://example.com/custom-homepage",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(data_dir / "podcast.xml")
        # Item link should be BASE_URL + filepath, NOT the feed.json link
        assert "<link>https://example.com/feeds/videos/lecture.mp4</link>" in content

    def test_link_parameter_overrides_base_url_directly(self, tmp_path):
        """generate_podcast_feed link parameter should override base_url for channel link."""
        write_metadata_file(tmp_path, "videos/lecture.mp4", filetype="mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed(link="https://custom.example.com")

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<link>https://custom.example.com</link>" in content
        assert "<link>https://example.com/feeds</link>" not in content


def write_feed_json_with_link(
    data_dir: Path,
    folder: str,
    title: str,
    description: str,
    link: str,
) -> Path:
    """Write a feed.json file with a link field to a subfolder within data_dir."""
    folder_path = data_dir / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    feed_json_path = folder_path / "feed.json"
    feed_json_path.write_text(
        json.dumps({"title": title, "description": description, "link": link}),
        encoding="utf-8",
    )
    return feed_json_path


# ---------------------------------------------------------------------------
# Hash-based GUID tests
# ---------------------------------------------------------------------------


class TestHashBasedGuid:
    """Tests for hash-based guid generation (FR-020).

    When the ``hash`` field is present in metadata, the guid should be the
    last 12 characters of the hash.  When the hash field is absent (legacy
    files), the guid should fall back to the absolute URL of the filepath.
    """

    @pytest.mark.parametrize(
        "filetype,filepath",
        [
            ("mp4", "videos/lecture.mp4"),
            ("pdf", "documents/doc.pdf"),
            ("html", "pages/article.html"),
        ],
    )
    def test_guid_uses_hash_when_present(self, tmp_path, filetype, filepath):
        """GUID should use the last 12 chars of hash when hash is present."""
        test_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        write_metadata_file(
            tmp_path,
            filepath,
            filetype=filetype,
            hash=test_hash,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        if filetype == "mp4":
            fg.generate_podcast_feed()
            xml_name = "podcast.xml"
        else:
            fg.generate_articles_feed()
            xml_name = "articles.xml"

        content = parse_rss_file(tmp_path / xml_name)
        expected_guid = test_hash[-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    @pytest.mark.parametrize(
        "filetype,xml_name,filepath",
        [
            ("mp4", "podcast.xml", "videos/lecture.mp4"),
            ("pdf", "articles.xml", "documents/doc.pdf"),
        ],
    )
    def test_guid_falls_back_to_filepath_when_no_hash(self, tmp_path, filetype, xml_name, filepath):
        """GUID should fall back to absolute URL when hash is not present."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(
            data_dir,
            filepath,
            filetype=filetype,
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        if filetype == "mp4":
            fg.generate_podcast_feed()
        else:
            fg.generate_articles_feed()

        content = parse_rss_file(data_dir / xml_name)
        expected_guid = f"https://example.com/feeds/{filepath}"
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_guid_uses_hash_for_mp4(self, tmp_path):
        """Podcast feed GUID should use hash[-12:] when hash is present."""
        test_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(
            data_dir,
            "videos/lecture.mp4",
            filetype="mp4",
            hash=test_hash,
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(data_dir / "podcast.xml")
        expected_guid = test_hash[-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content
        # Should NOT contain the filepath-based guid
        guid_section = content.split("<guid")[1].split("</guid>")[0]
        assert "https://example.com/feeds/videos/lecture.mp4" not in guid_section

    def test_guid_uses_hash_for_pdf(self, tmp_path):
        """Articles feed GUID should use hash[-12:] when hash is present."""
        test_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            hash=test_hash,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        expected_guid = test_hash[-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_guid_uses_hash_for_html(self, tmp_path):
        """Articles feed GUID should use hash[-12:] for HTML files when hash is present."""
        test_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        write_metadata_file(
            tmp_path,
            "pages/article.html",
            filetype="html",
            hash=test_hash,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        expected_guid = test_hash[-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_guid_last_12_chars_of_hash(self, tmp_path):
        """GUID should be exactly the last 12 characters of the hash."""
        test_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            hash=test_hash,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        expected_guid = test_hash[-12:]
        assert expected_guid == "456789abcdef"
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_guid_fallback_for_mixed_hash_and_no_hash(self, tmp_path):
        """When some items have hash and others don't, each should use its own guid."""
        hash_with = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_metadata_file(
            data_dir,
            "videos/with_hash.mp4",
            filetype="mp4",
            hash=hash_with,
        )
        write_metadata_file(
            data_dir,
            "videos/without_hash.mp4",
            filetype="mp4",
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(data_dir / "podcast.xml")
        # First item should use hash[-12:]
        assert f'<guid isPermaLink="false">{hash_with[-12:]}</guid>' in content
        # Second item should use filepath URL
        assert (
            '<guid isPermaLink="false">https://example.com/feeds/videos/without_hash.mp4</guid>'
            in content
        )


class TestComputeGuid:
    """Tests for the _compute_guid() helper method."""

    def test_compute_guid_with_hash(self, tmp_path):
        """_compute_guid should return hash[-12:] when hash is present."""
        test_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        metadata = make_metadata(
            "videos/lecture.mp4",
            filetype="mp4",
            hash=test_hash,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        guid = fg._compute_guid(metadata)

        assert guid == test_hash[-12:]

    def test_compute_guid_without_hash(self, tmp_path):
        """_compute_guid should return absolute URL when hash is not present."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metadata = make_metadata(
            "videos/lecture.mp4",
            filetype="mp4",
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        guid = fg._compute_guid(metadata)

        assert guid == "https://example.com/feeds/videos/lecture.mp4"

    def test_compute_guid_with_empty_hash(self, tmp_path):
        """_compute_guid should return absolute URL when hash is empty string."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metadata = make_metadata(
            "videos/lecture.mp4",
            filetype="mp4",
            hash="",
        )

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        guid = fg._compute_guid(metadata)

        assert guid == "https://example.com/feeds/videos/lecture.mp4"


# ---------------------------------------------------------------------------
# Hash update before feed generation
# ---------------------------------------------------------------------------


class TestUpdateMissingHashes:
    """Tests for _update_missing_hashes() and integration with generate_all_feeds()."""

    def test_json_without_hash_gets_updated(self, tmp_path):
        """JSON metadata files without a hash field should get one added."""
        # Create a source file and JSON metadata without hash
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
                    "description": "",
                    "summary": "Summary text",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 1
        # Read back the JSON and verify hash was added
        updated_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in updated_data
        assert len(updated_data["hash"]) == 64  # SHA256 hex digest length

    def test_json_with_hash_not_modified(self, tmp_path):
        """JSON metadata files that already have a hash should not be modified."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        existing_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        json_path = tmp_path / "lecture.mp4.json"
        original_data = {
            "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
            "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
            "title": {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            "description": "",
            "summary": "Summary text",
            "filename": "lecture.mp4",
            "filepath": "lecture.mp4",
            "filetype": "mp4",
            "date": 1728931200,
            "size": 1000,
            "hash": existing_hash,
        }
        json_path.write_text(json.dumps(original_data), encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 0
        # Verify hash was not changed
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["hash"] == existing_hash

    def test_hash_computed_correctly(self, tmp_path):
        """The hash should be the actual SHA256 of the source file content."""
        import hashlib

        source = tmp_path / "doc.pdf"
        source.write_bytes(b"hello world")
        json_path = tmp_path / "doc.pdf.json"
        json_path.write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 11, "output_size": 11},
                    "title": {"filename": "doc", "extracted": None, "generated": "Doc"},
                    "description": "desc",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 11,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg._update_missing_hashes()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256(b"hello world").hexdigest()
        assert data["hash"] == expected_hash

    def test_orphaned_json_skipped(self, tmp_path):
        """JSON files without a corresponding source file should be skipped."""
        # Create only JSON, no source file
        json_path = tmp_path / "orphan.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": "Orphan",
                    "description": "",
                    "filename": "orphan.mp4",
                    "filepath": "orphan.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 0
        # Verify no hash was added
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" not in data

    def test_feed_json_not_modified(self, tmp_path):
        """feed.json configuration files should not be treated as metadata."""
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "My Feed", "description": "Desc"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 0
        # feed.json should be unchanged
        data = json.loads((tmp_path / "feed.json").read_text(encoding="utf-8"))
        assert "hash" not in data

    def test_invalid_json_skipped_with_warning(self, tmp_path, caplog):
        """Invalid JSON files should be skipped with a warning."""
        source = tmp_path / "bad.mp4"
        source.write_bytes(b"data")
        json_path = tmp_path / "bad.mp4.json"
        json_path.write_text("{invalid json", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 0
        assert any(
            "Failed to read JSON" in record.message
            for record in caplog.records
            if record.levelno >= 30
        )

    def test_hash_added_to_ineligible_file(self, tmp_path):
        """Hash should be added even to files that are not yet feed-eligible."""
        source = tmp_path / "incomplete.mp4"
        source.write_bytes(b"data")
        json_path = tmp_path / "incomplete.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "in_progress", "input_size": 1000, "output_size": 0},
                    "title": {"filename": "incomplete", "extracted": None, "generated": None},
                    "description": "",
                    "filename": "incomplete.mp4",
                    "filepath": "incomplete.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 1
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in data
        assert len(data["hash"]) == 64

    def test_nonexistent_data_dir_returns_zero(self, tmp_path):
        """_update_missing_hashes should return 0 for nonexistent directory."""
        fg = FeedGenerator(
            data_dir=tmp_path / "nonexistent",
            base_url="https://example.com/feeds",
        )
        updated = fg._update_missing_hashes()
        assert updated == 0

    def test_multiple_files_updated(self, tmp_path):
        """Multiple JSON files without hashes should all be updated."""
        for name, content in [("a.mp4", b"aaa"), ("b.pdf", b"bbb")]:
            source = tmp_path / name
            source.write_bytes(content)
            json_path = tmp_path / (name + ".json")
            stem = Path(name).stem
            if name.endswith(".mp4"):
                step_data = {
                    "transcribe": {"status": "complete", "input_size": 3, "output_size": 3},
                    "summarize": {"status": "complete", "input_size": 3, "output_size": 3},
                }
                filetype = "mp4"
            else:
                step_data = {
                    "extract": {"status": "complete", "input_size": 3, "output_size": 3},
                }
                filetype = "pdf"
            json_path.write_text(
                json.dumps(
                    {
                        **step_data,
                        "title": {"filename": stem, "extracted": None, "generated": stem},
                        "description": "",
                        "filename": name,
                        "filepath": name,
                        "filetype": filetype,
                        "date": 1728931200,
                        "size": 3,
                    }
                ),
                encoding="utf-8",
            )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 2
        for name in ["a.mp4", "b.pdf"]:
            data = json.loads((tmp_path / (name + ".json")).read_text(encoding="utf-8"))
            assert "hash" in data
            assert len(data["hash"]) == 64

    def test_empty_hash_treated_as_missing(self, tmp_path):
        """JSON files with an empty string hash should be updated."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
                    "description": "",
                    "summary": "Summary text",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                    "hash": "",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        updated = fg._update_missing_hashes()

        assert updated == 1
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["hash"]  # Should now be non-empty
        assert len(data["hash"]) == 64

    def test_hash_added_to_html_file(self, tmp_path):
        """Hash should be added to HTML file metadata as well."""
        source = tmp_path / "page.html"
        source.write_text("<html><body>Test</body></html>", encoding="utf-8")
        json_path = tmp_path / "page.html.json"
        json_path.write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 100, "output_size": 50},
                    "title": {"filename": "page", "extracted": None, "generated": "Page"},
                    "description": "desc",
                    "filename": "page.html",
                    "filepath": "page.html",
                    "filetype": "html",
                    "date": 1728931200,
                    "size": 100,
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg._update_missing_hashes()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in data
        assert len(data["hash"]) == 64


class TestHashUpdateIntegration:
    """Integration tests for hash update in generate_all_feeds()."""

    def test_generate_all_feeds_updates_missing_hash(self, tmp_path):
        """generate_all_feeds() should add hash and feeds should use hash-based guid."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
                    "description": "",
                    "summary": "Summary text",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )
        # Create root feed.json so root-level feeds are generated
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Verify JSON was updated with hash
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in data
        assert len(data["hash"]) == 64

        # Verify the feed uses hash-based guid (last 12 chars of hash)
        content = parse_rss_file(tmp_path / "podcast.xml")
        expected_guid = data["hash"][-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_generate_all_feeds_existing_hash_not_changed(self, tmp_path):
        """generate_all_feeds() should not change JSON files that already have a hash."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        existing_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
                    "description": "",
                    "summary": "Summary text",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                    "hash": existing_hash,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Verify hash was not changed
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["hash"] == existing_hash

        # Verify the feed uses the existing hash-based guid
        content = parse_rss_file(tmp_path / "podcast.xml")
        expected_guid = existing_hash[-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_generate_all_feeds_hash_for_articles(self, tmp_path):
        """generate_all_feeds() should add hash to PDF metadata and use it in articles feed."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        json_path = tmp_path / "doc.pdf.json"
        json_path.write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "doc", "extracted": None, "generated": "My Doc"},
                    "description": "Description",
                    "filename": "doc.pdf",
                    "filepath": "doc.pdf",
                    "filetype": "pdf",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Verify JSON was updated with hash
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in data
        assert len(data["hash"]) == 64

        # Verify the articles feed uses hash-based guid
        content = parse_rss_file(tmp_path / "articles.xml")
        expected_guid = data["hash"][-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content

    def test_generate_all_feeds_skips_orphaned_json(self, tmp_path):
        """generate_all_feeds() should skip orphaned JSON files during hash update."""
        # Create only JSON, no source file
        json_path = tmp_path / "orphan.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "orphan", "extracted": None, "generated": "Orphan"},
                    "description": "",
                    "summary": "Summary",
                    "filename": "orphan.mp4",
                    "filepath": "orphan.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )
        # Create a real file with hash for comparison
        source = tmp_path / "real.mp4"
        source.write_bytes(b"real data")
        real_json = tmp_path / "real.mp4.json"
        real_json.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "real", "extracted": None, "generated": "Real"},
                    "description": "",
                    "summary": "Summary",
                    "filename": "real.mp4",
                    "filepath": "real.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Orphaned JSON should not have hash
        orphan_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" not in orphan_data

        # Real file should have hash
        real_data = json.loads(real_json.read_text(encoding="utf-8"))
        assert "hash" in real_data

    def test_generate_all_feeds_idempotent_hash_update(self, tmp_path):
        """Running generate_all_feeds() twice should not change hashes."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {"status": "complete", "input_size": 500, "output_size": 200},
                    "title": {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
                    "description": "",
                    "summary": "Summary text",
                    "filename": "lecture.mp4",
                    "filepath": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "size": 1000,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")

        # First run: should add hash
        fg.generate_all_feeds()
        data1 = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in data1

        # Second run: should not change hash
        fg.generate_all_feeds()
        data2 = json.loads(json_path.read_text(encoding="utf-8"))
        assert data2["hash"] == data1["hash"]


# ---------------------------------------------------------------------------
# Summarization attribution (_get_summarization_attribution)
# ---------------------------------------------------------------------------


class TestSummarizationAttribution:
    """Tests for _get_summarization_attribution() helper function."""

    @pytest.fixture
    def mock_config(self):
        """Mock get_config to return a config with llm_model set."""
        with mock.patch("local_feed_generator.feeds.get_config") as mock_get_config:
            mock_config_obj = mock.MagicMock()
            mock_config_obj.llm_model = "test-llm-model"
            mock_get_config.return_value = mock_config_obj
            yield mock_get_config

    def test_mp4_uses_summarize_model_and_end_time(self, mock_config):
        """MP4 files should use summarize.model and summarize.end_time."""
        metadata = make_metadata(
            filepath="videos/lecture.mp4",
            filetype="mp4",
            summary="Summary text",
        )
        # Override summarize step with model and end_time
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="whisper-large-v3",
            end_time=1728931200.0,
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by whisper-large-v3" in result
        assert "2024-10-14" in result  # datetime from 1728931200.0

    def test_pdf_with_summarize_uses_summarize_model(self, mock_config):
        """PDF with summarize step should use summarize.model and summarize.end_time."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="gpt-4o-mini",
            end_time=1719211200.0,  # 2024-06-24 12:00:00 UTC (noon to avoid tz date boundary)
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by gpt-4o-mini" in result
        assert "2024-06-24" in result  # datetime from 1719211200.0

    def test_pdf_without_summarize_uses_config_llm_model(self, mock_config):
        """PDF without summarize should use config.llm_model and metadata.date."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
        )
        # No summarize step; falls back to config.llm_model and metadata.date
        metadata.summarize = None
        metadata.extract = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            end_time=1728931200.0,
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by test-llm-model" in result
        assert "2024-10-14" in result  # from metadata.date

    def test_html_without_summarize_uses_config_llm_model(self, mock_config):
        """HTML without summarize should use config.llm_model and metadata.date."""
        metadata = make_metadata(
            filepath="pages/page.html",
            filetype="html",
            description="Description",
        )
        metadata.summarize = None
        metadata.extract = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            end_time=1728931200.0,
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by test-llm-model" in result
        assert "2024-10-14" in result  # from metadata.date

    def test_falls_back_to_metadata_date_when_end_time_none(self, mock_config):
        """Should fall back to metadata.date when step.end_time is None."""
        metadata = make_metadata(
            filepath="videos/lecture.mp4",
            filetype="mp4",
            summary="Summary",
            date=1728931200,
        )
        # summarize has no end_time
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="whisper-model",
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by whisper-model" in result
        assert "2024-10-14" in result  # from metadata.date

    def test_falls_back_to_publication_date(self, mock_config):
        """Should fall back to publication_date when summarize.end_time is None and date is None."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
            date=None,
            publication_date=1719211200.0,  # 2024-06-24 12:00:00 UTC
        )
        metadata.summarize = None
        metadata.extract = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by test-llm-model" in result
        assert "2024-06-24" in result  # from publication_date

    def test_returns_empty_when_no_timestamp_available(self, mock_config):
        """Should return empty string when no timestamp is available."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
            date=None,
            publication_date=None,
        )
        metadata.summarize = None
        metadata.extract = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
        )

        result = _get_summarization_attribution(metadata)
        assert result == ""

    def test_returns_empty_when_no_model_available(self, mock_config):
        """Should return empty string when no model can be determined."""
        # Mock config with no llm_model
        mock_config.return_value.llm_model = None

        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
        )
        # summarize.model is None, and config.llm_model is None

        result = _get_summarization_attribution(metadata)
        assert result == ""

    def test_returns_empty_when_config_unavailable(self):
        """Should return empty string when get_config() raises an exception."""
        with mock.patch(
            "local_feed_generator.feeds.get_config",
            side_effect=Exception("Config not loaded"),
        ):
            metadata = make_metadata(
                filepath="videos/lecture.mp4",
                filetype="mp4",
                summary="Summary",
            )
            metadata.summarize = StepMap(
                status="complete",
                input_size=100,
                output_size=50,
                model=None,  # model is None, so falls back to config which fails
            )

            result = _get_summarization_attribution(metadata)
            assert result == ""

    def test_datetime_uses_local_timezone(self, mock_config):
        """Datetime should be converted to local timezone."""
        metadata = make_metadata(
            filepath="videos/lecture.mp4",
            filetype="mp4",
            summary="Summary",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="test-model",
            end_time=1728931200.0,  # 2024-10-14 12:00:00 UTC
        )

        result = _get_summarization_attribution(metadata)
        # The result should contain a timezone abbreviation (local tz)
        # and the date should be in the expected format
        assert "Summarized by test-model at" in result
        # Verify datetime format: YYYY-MM-DD HH:MM:SS TZ
        import re

        pattern = r"Summarized by test-model at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \S+"
        assert re.match(pattern, result)

    def test_mp4_fallback_to_config_model_when_summarize_model_none(self, mock_config):
        """MP4 with summarize.model=None should fall back to config.llm_model."""
        metadata = make_metadata(
            filepath="videos/lecture.mp4",
            filetype="mp4",
            summary="Summary",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model=None,  # None, should fall back to config
            end_time=1728931200.0,
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by test-llm-model" in result

    def test_pdf_with_summarize_fallback_to_config_model(self, mock_config):
        """PDF with summarize.model=None should fall back to config.llm_model."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model=None,
            end_time=1728931200.0,
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by test-llm-model" in result

    def test_pdf_without_summarize_falls_back_to_date(self, mock_config):
        """PDF without summarize and without summarize.end_time should use metadata.date."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description",
            date=1728931200,
        )
        metadata.summarize = None
        metadata.extract = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            # end_time is None
        )

        result = _get_summarization_attribution(metadata)
        assert "Summarized by test-llm-model" in result
        assert "2024-10-14" in result


class TestAttributionInPodcastFeed:
    """Tests for attribution appearing in generated podcast feeds."""

    @pytest.fixture
    def mock_config(self):
        """Mock get_config to return a config with llm_model set."""
        with mock.patch("local_feed_generator.feeds.get_config") as mock_get_config:
            mock_config_obj = mock.MagicMock()
            mock_config_obj.llm_model = "podcast-llm-model"
            mock_get_config.return_value = mock_config_obj
            yield mock_get_config

    def test_podcast_feed_includes_attribution(self, tmp_path, mock_config):
        """Podcast feed should include summarization attribution in description."""
        metadata = make_metadata(
            filepath="videos/lecture.mp4",
            filetype="mp4",
            summary="Key takeaways from the lecture.",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="whisper-large-v3",
            end_time=1728931200.0,
        )
        # Write metadata file
        source = tmp_path / "videos" / "lecture.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"video data")
        json_path = tmp_path / "videos" / "lecture.mp4.json"
        json_path.write_text(metadata.to_json(), encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "Key takeaways from the lecture." in content
        assert "Summarized by whisper-large-v3" in content

    def test_podcast_feed_attribution_separated_by_paragraph(self, tmp_path, mock_config):
        """Attribution should be separated from description by paragraph break."""
        metadata = make_metadata(
            filepath="videos/lecture.mp4",
            filetype="mp4",
            summary="Summary text",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="test-model",
            end_time=1728931200.0,
        )
        source = tmp_path / "videos" / "lecture.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"video data")
        json_path = tmp_path / "videos" / "lecture.mp4.json"
        json_path.write_text(metadata.to_json(), encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # Both summary and attribution should be present
        assert "Summary text" in content
        assert "Summarized by test-model" in content

    def test_podcast_feed_no_attribution_when_config_unavailable(self, tmp_path):
        """Podcast feed should still work when config is unavailable (no attribution)."""
        with mock.patch(
            "local_feed_generator.feeds.get_config",
            side_effect=Exception("Config not loaded"),
        ):
            metadata = make_metadata(
                filepath="videos/lecture.mp4",
                filetype="mp4",
                summary="Summary text",
            )
            metadata.summarize = StepMap(
                status="complete",
                input_size=100,
                output_size=50,
                model=None,
            )
            source = tmp_path / "videos" / "lecture.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video data")
            json_path = tmp_path / "videos" / "lecture.mp4.json"
            json_path.write_text(metadata.to_json(), encoding="utf-8")

            fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
            fg.generate_podcast_feed()

            content = parse_rss_file(tmp_path / "podcast.xml")
            assert "Summary text" in content
            assert "Summarized by" not in content


class TestAttributionInArticlesFeed:
    """Tests for attribution appearing in generated articles feeds."""

    @pytest.fixture
    def mock_config(self):
        """Mock get_config to return a config with llm_model set."""
        with mock.patch("local_feed_generator.feeds.get_config") as mock_get_config:
            mock_config_obj = mock.MagicMock()
            mock_config_obj.llm_model = "articles-llm-model"
            mock_get_config.return_value = mock_config_obj
            yield mock_get_config

    def test_articles_feed_includes_attribution_for_pdf_with_summarize(self, tmp_path, mock_config):
        """Articles feed should include attribution for PDF with summarize step."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description text",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model="gpt-4o",
            end_time=1728931200.0,
        )
        source = tmp_path / "docs" / "doc.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"pdf data")
        json_path = tmp_path / "docs" / "doc.pdf.json"
        json_path.write_text(metadata.to_json(), encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "Summarized by gpt-4o" in content

    def test_articles_feed_includes_attribution_for_pdf_with_summarize_model_none(
        self, tmp_path, mock_config
    ):
        """Articles feed should include attribution for PDF with summarize.model=None,
        falling back to config.llm_model."""
        metadata = make_metadata(
            filepath="docs/doc.pdf",
            filetype="pdf",
            description="Description text",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model=None,  # None, should fall back to config
            end_time=1728931200.0,
        )
        source = tmp_path / "docs" / "doc.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"pdf data")
        json_path = tmp_path / "docs" / "doc.pdf.json"
        json_path.write_text(metadata.to_json(), encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "Summarized by articles-llm-model" in content

    def test_articles_feed_includes_attribution_for_html_with_summarize_model_none(
        self, tmp_path, mock_config
    ):
        """Articles feed should include attribution for HTML with summarize.model=None,
        falling back to config.llm_model."""
        metadata = make_metadata(
            filepath="pages/page.html",
            filetype="html",
            description="Description text",
        )
        metadata.summarize = StepMap(
            status="complete",
            input_size=100,
            output_size=50,
            model=None,  # None, should fall back to config
            end_time=1728931200.0,
        )
        source = tmp_path / "pages" / "page.html"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("<html></html>", encoding="utf-8")
        json_path = tmp_path / "pages" / "page.html.json"
        json_path.write_text(metadata.to_json(), encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "Summarized by articles-llm-model" in content

    def test_articles_feed_no_attribution_when_config_unavailable(self, tmp_path):
        """Articles feed should work without attribution when config unavailable."""
        with mock.patch(
            "local_feed_generator.feeds.get_config",
            side_effect=Exception("Config not loaded"),
        ):
            metadata = make_metadata(
                filepath="docs/doc.pdf",
                filetype="pdf",
                description="Description text",
            )
            metadata.summarize = StepMap(
                status="complete",
                input_size=100,
                output_size=50,
                model=None,
            )
            source = tmp_path / "docs" / "doc.pdf"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"pdf data")
            json_path = tmp_path / "docs" / "doc.pdf.json"
            json_path.write_text(metadata.to_json(), encoding="utf-8")

            fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
            fg.generate_articles_feed()

            content = parse_rss_file(tmp_path / "articles.xml")
            assert "Description text" in content
            assert "Summarized by" not in content
