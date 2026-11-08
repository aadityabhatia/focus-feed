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
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from focus_feed.feeds import (
    MEDIA_FILETYPES,
    FeedGenerator,
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
            ("mp3", True, False, ""),  # Audio only: podcast generated
            ("mp3", True, False, "Course1/"),  # Per-folder audio: podcast generated
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
        root_podcast_paths = {m.source_filepath for m in results["podcast"]}
        assert "videos/lecture.mp4" in root_podcast_paths
        # Root articles should include root-level PDF
        root_article_paths = {m.source_filepath for m in results["articles"]}
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
        course1_paths = {m.source_filepath for m in course1_metadata}
        assert course1_paths == {
            "Course1/lecture1.mp4",
            "Course1/Unit1/lecture2.mp4",
            "Course1/Unit1/Sub/deep.mp4",
        }

        # Filter for Unit1 should only include the two files under Unit1
        unit1_metadata = fg._filter_metadata_for_folder(
            all_metadata, tmp_path / "Course1" / "Unit1"
        )
        unit1_paths = {m.source_filepath for m in unit1_metadata}
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


class TestGenerateIndexHtml:
    """Tests for _generate_index_html() and index.html generation."""

    def test_index_html_created_after_generate_all_feeds(self, tmp_path):
        """generate_all_feeds should create index.html in data_dir."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        write_metadata_file(tmp_path, "documents/doc.pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Test Feed", "description": "Test Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert (tmp_path / "index.html").exists()

    def test_nested_index_urls_are_relative_to_data_root(self, tmp_path):
        """Nested feed URLs should include the folder beneath the data root."""
        feed_dir = tmp_path / "ile"
        feed_dir.mkdir()
        (feed_dir / "feed.json").write_text(
            json.dumps({"title": "ILE", "description": "ILE feed"}),
            encoding="utf-8",
        )
        (feed_dir / "podcast.xml").write_text("<rss></rss>", encoding="utf-8")
        (feed_dir / "articles.xml").write_text("<rss></rss>", encoding="utf-8")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://feed.example.com")
        fg._generate_index_html(output_dir=feed_dir)

        content = (feed_dir / "index.html").read_text(encoding="utf-8")
        assert 'href="https://feed.example.com/ile/podcast.xml"' in content
        assert 'href="https://feed.example.com/ile/articles.xml"' in content

    def test_index_html_structure_and_content(self, tmp_path):
        """index.html should contain expected structure, head, sections, and CSS."""
        write_metadata_file(tmp_path, "videos/lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Test Feed", "description": "Test Description"}),
            encoding="utf-8",
        )

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
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Test Feed", "description": "Test Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        assert (tmp_path / "index.html").exists()
        content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<h1>Feed Index</h1>" in content
        assert "<h2>Podcast Feeds</h2>" in content
        assert "<h2>Articles Feeds</h2>" in content

    def test_index_html_overwrites_existing(self, tmp_path):
        """generate_all_feeds should overwrite existing index.html."""
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Test Feed", "description": "Test Description"}),
            encoding="utf-8",
        )
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
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Test Feed", "description": "Test Description"}),
            encoding="utf-8",
        )
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
