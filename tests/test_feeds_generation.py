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
import re
from typing import Any
from unittest import mock

import pytest

from focus_feed.feeds import (
    MEDIA_FILETYPES,
    FeedGenerator,
    get_mime_type,
)
from focus_feed.metadata import Metadata, StepMap

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
    summarize_end_time: float | None = 1728931200.0,
) -> Metadata:
    """Create a Metadata object for testing.

    Args:
        filepath: Relative path from data directory root.
        filetype: A supported media type, 'pdf', or 'html'.
        title: Title string (defaults to filepath stem).
        description: Description string.
        summary: Summary string (for media files).
        transcript: Transcript string (for media files).
        size: File size in bytes.
        feed_eligible: If True, set status to "complete" for all applicable steps.
            If False, leave status as None.
        error: Error message (if provided, feed_eligible is set to False).
         date: UNIX epoch date (seconds).
         publication_date: Publication date from document metadata (seconds).
             If None, the date field is used for pubDate in articles feed.
         hash: SHA256 hash string (optional; used for guid generation).
         summarize_end_time: End time for the summarize step (seconds).
             Used as the pubDate source for feed items. Pass None to simulate
             a missing end_time (items will be skipped in feed generation).

    Returns:
        A validated Metadata instance.
    """
    filename = Path(filepath).name
    kwargs: dict[str, Any] = {
        "title": title or filename,
        "filename": filename,
        "filetype": filetype,
        "date": date,
        "size": size,
    }
    if hash is not None:
        kwargs["hash"] = hash
    if filetype in MEDIA_FILETYPES:
        # For media, transcript maps to content, summary holds the LLM key
        # takeaways, and description is left blank.
        kwargs["content"] = transcript
        kwargs["description"] = ""  # Media: description is blank
        kwargs["summary"] = summary  # Media: summary holds LLM key takeaways
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
                end_time=summarize_end_time,
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
                end_time=summarize_end_time,
            )
    if error:
        kwargs["error"] = error
    if publication_date is not None:
        kwargs["publication_date"] = publication_date
    metadata = Metadata(**kwargs)
    metadata._source_filepath = filepath
    return metadata


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


@pytest.fixture
def mock_config(request):
    """Mock feed config with the model used by the requesting test class."""
    models = {
        "TestSummarizationAttribution": "test-llm-model",
        "TestAttributionInPodcastFeed": "podcast-llm-model",
        "TestAttributionInArticlesFeed": "articles-llm-model",
    }
    with mock.patch("focus_feed.feeds.get_config") as mock_get_config:
        mock_config_obj = mock.MagicMock()
        mock_config_obj.llm_model = models[request.node.cls.__name__]
        mock_get_config.return_value = mock_config_obj
        yield mock_get_config


# ---------------------------------------------------------------------------
# FeedGenerator construction
# ---------------------------------------------------------------------------


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


_ATOM_FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Not RSS</title>
</feed>
"""


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


_NO_CHANNEL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
</rss>
"""


_MALFORMED_XML = "<rss><channel><item><title>Unclosed</item></channel>"


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


class TestFeedGeneratorConstruction:
    """Tests for FeedGenerator initialization."""

    def test_uses_injected_config_base_url_when_base_url_omitted(self, tmp_path):
        """An injected config should supply the default base URL."""
        config = mock.MagicMock()
        config.base_url = "https://injected.example.com/feeds/"

        fg = FeedGenerator(data_dir=tmp_path, config=config)

        assert fg._config is config
        assert fg.base_url == "https://injected.example.com/feeds"

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
        with caplog.at_level(logging.CRITICAL, logger="focus_feed"):
            FeedGenerator(data_dir=nonexistent, base_url="https://example.com")
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) == 1
        assert "Data directory does not exist" in critical_records[0].message


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
        assert result[0].source_filepath == "videos/lecture.mp4"

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
        filepaths = {m.source_filepath for m in result}
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
        assert result[0].source_filepath == "lecture.mp4"
        assert not any(
            "Failed to parse" in record.message
            for record in caplog.records
            if record.levelno >= 30  # WARNING
        )

    def test_eligible_paths_filters_discovered_metadata(self, tmp_path):
        """An explicit source allow-list excludes all other metadata."""
        included = write_metadata_file(tmp_path, "included.mp4")
        write_metadata_file(tmp_path, "excluded.mp4")

        fg = FeedGenerator(
            data_dir=tmp_path,
            base_url="https://example.com",
            eligible_paths=[included.with_suffix("")],
        )

        result = fg._discover_metadata_files()

        assert [metadata.source_filepath for metadata in result] == ["included.mp4"]


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
        filepaths = [m.source_filepath for m in result]
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
        metadata._source_filepath = filepath
        key = FeedGenerator._sort_key(metadata)
        assert key == expected_key


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
        # An extended leading filename code replaces the parent folder prefix.
        parent_dir = Path(filepath).parent.name
        filename_token = re.match(r"[A-Za-z0-9]+", Path(filepath).stem)
        effective_prefix = parent_dir
        if (
            filename_token
            and len(filename_token.group()) > len(parent_dir)
            and filename_token.group().lower().startswith(parent_dir.lower())
        ):
            effective_prefix = filename_token.group()
        title_normalized = title.replace("_", " ").lower()
        if title_normalized.startswith(effective_prefix.lower()):
            expected_title = title
        else:
            expected_title = f"{effective_prefix}: {title}"
        assert f"<title>{expected_title}</title>" in content

    def test_articles_feed_uses_extended_filename_code_as_title_prefix(self, tmp_path):
        """Articles feed titles should use an extended filename code as prefix."""
        write_metadata_file(
            tmp_path,
            "H103/H103RA_Decline_of_Napoleon_(FS).docx",
            filetype="docx",
            title="Decline of Napoleon",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<title>H103RA: Decline of Napoleon</title>" in content
        assert "<title>H103: Decline of Napoleon</title>" not in content

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
                        "end_time": 1728931200.0,
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
        "filetype,expected_mime_type",
        [
            ("mp3", "audio/mpeg"),
            ("m4a", "audio/mp4"),
            ("wav", "audio/wav"),
            ("webm", "audio/webm"),
            ("aac", "audio/aac"),
            ("flac", "audio/flac"),
            ("ogg", "audio/ogg"),
            ("opus", "audio/opus"),
            ("mp4", "audio/mp4"),
            ("mpeg", "audio/mpeg"),
            ("mpga", "audio/mpeg"),
            ("mkv", "audio/mp4"),
            ("mov", "audio/mp4"),
            ("avi", "audio/mp4"),
            ("wmv", "audio/mp4"),
        ],
    )
    def test_podcast_feed_supports_media_types_and_mime_types(
        self, tmp_path, filetype, expected_mime_type
    ):
        """Every supported media type is included with its mapped enclosure MIME type."""
        filepath = f"videos/lecture.{filetype}"
        write_metadata_file(tmp_path, filepath, filetype=filetype)

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        assert [item.filetype for item in items] == [filetype]
        content = parse_rss_file(tmp_path / "podcast.xml")
        assert f'type="{expected_mime_type}"' in content
        assert get_mime_type(filetype) == expected_mime_type

    def test_get_mime_type_unknown_type_uses_binary_fallback(self):
        assert get_mime_type("unknown") == "application/octet-stream"

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

    def test_podcast_feed_pubdate_uses_summarize_end_time(self, tmp_path):
        """Podcast feed pubDate should use summarize.end_time."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,  # 2024-10-14
            summarize_end_time=1719200000.0,  # 2024-06-24
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # pubDate should reflect summarize.end_time (June 2024)
        from datetime import UTC, datetime

        expected = datetime.fromtimestamp(1719200000.0, tz=UTC).strftime("%a, %d %b %Y")
        assert expected in content

    def test_podcast_feed_skips_item_when_summarize_end_time_missing(self, tmp_path):
        """Podcast feed should skip items with missing summarize.end_time."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            date=1728931200,
            summarize_end_time=None,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        # Feed file should be generated but the item should not be in it
        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<item>" not in content

    def test_podcast_feed_items_sorted(self, tmp_path):
        """Podcast feed items should be sorted by folder then filename (FR-021)."""
        write_metadata_file(tmp_path, "zebra/last.mp4")
        write_metadata_file(tmp_path, "alpha/first.mp4")
        write_metadata_file(tmp_path, "alpha/middle.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        items = fg.generate_podcast_feed()

        filepaths = [m.source_filepath for m in items]
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

        filepaths = [m.source_filepath for m in items]
        assert filepaths == ["alpha/first.html", "alpha/middle.pdf", "zebra/last.pdf"]

    def test_articles_feed_pubdate_uses_summarize_end_time(self, tmp_path):
        """Articles feed pubDate should use summarize.end_time."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            date=1728931200,  # 2024-10-14
            summarize_end_time=1719200000.0,  # 2024-06-24
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # pubDate should reflect summarize.end_time (June 2024)
        from datetime import UTC, datetime

        expected = datetime.fromtimestamp(1719200000.0, tz=UTC).strftime("%a, %d %b %Y")
        assert expected in content

    def test_articles_feed_skips_item_when_summarize_end_time_missing(self, tmp_path):
        """Articles feed should skip items with missing summarize.end_time."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            date=1728931200,
            summarize_end_time=None,
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        # Feed file should be generated but the item should not be in it
        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<item>" not in content

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
        write_metadata_file(tmp_path, "videos/lecture.mp4", size=9)
        write_metadata_file(tmp_path, "documents/doc.pdf", size=9)
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


class TestXmlEscaping:
    """Tests for proper XML escaping and CDATA wrapping of special characters.

    These tests verify that our feed generator passes content with special
    characters through to feedgen without crashing. Item descriptions are now
    wrapped in CDATA sections to preserve HTML formatting, while feedgen handles
    XML escaping for other elements.
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


class TestCDataWrapping:
    """Tests for CDATA wrapping of feed content."""

    def test_podcast_description_uses_cdata(self, tmp_path):
        """Podcast item description should be wrapped in CDATA."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            summary="Key takeaways with **bold** and [link](https://example.com)",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<![CDATA[" in content
        assert "<strong>bold</strong>" in content
        assert "<a href=" in content

    def test_articles_description_uses_cdata(self, tmp_path):
        """Articles item description should be wrapped in CDATA."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="Description with **formatting** & special chars",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<![CDATA[" in content
        assert "<strong>formatting</strong>" in content
        # Ampersand should be escaped by markdown but preserved in CDATA
        assert "&amp;" in content or "&" in content

    def test_cdata_content_not_double_escaped(self, tmp_path):
        """CDATA-wrapped HTML should not be double-escaped."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            summary="# Title\n\nParagraph with **bold** text",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<![CDATA[" in content
        # HTML tags should be present unescaped inside CDATA
        assert "<h1>Title</h1>" in content
        assert "<strong>bold</strong>" in content
        # Should NOT be double-escaped
        assert "&lt;strong&gt;" not in content

    def test_cdata_terminating_sequence_sanitized(self, tmp_path):
        """Content containing ]]> should be sanitized to avoid CDATA termination."""
        # Create content that would produce ]]> in the HTML output
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="Code sample: ]]> sequence",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # The ]]> should be replaced with ]]&gt; to prevent CDATA termination
        assert "]]&gt;" in content
        # Should still have CDATA wrapper
        assert "<![CDATA[" in content

    def test_cdata_valid_xml(self, tmp_path):
        """CDATA-wrapped feed should be valid XML."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description='Description with <html> & "quotes"',
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        # Feed should parse without errors
        import xml.etree.ElementTree as ET

        content = parse_rss_file(tmp_path / "articles.xml")
        root = ET.fromstring(content)  # Should not raise
        assert root.tag == "rss"

    def test_empty_description_still_works(self, tmp_path):
        """Empty descriptions should not cause errors with CDATA wrapping."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # Should still be valid XML
        import xml.etree.ElementTree as ET

        root = ET.fromstring(content)
        assert root.tag == "rss"

    def test_html_with_ampersand_in_cdata(self, tmp_path):
        """Ampersands in HTML content should be escaped by markdown, then preserved in CDATA."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            summary="Tom & Jerry & Friends",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<![CDATA[" in content
        # Ampersand is escaped to &amp; by the markdown library, then preserved
        # as-is in CDATA (not double-escaped by feedgen)
        assert "Tom &amp; Jerry &amp; Friends" in content

    def test_content_encoded_element_not_duplicated(self, tmp_path):
        """When using content() with type='CDATA', <content:encoded> should not appear."""
        write_metadata_file(
            tmp_path,
            "videos/lecture.mp4",
            filetype="mp4",
            summary="Summary text",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # Should have CDATA in description
        assert "<![CDATA[" in content
        # Should NOT have a separate content:encoded element
        assert "content:encoded" not in content


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
