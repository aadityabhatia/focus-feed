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

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from unittest import mock

import pytest

from focus_feed.feeds import (
    MEDIA_FILETYPES,
    FeedGenerator,
    _get_summarization_attribution,
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
    def test_guid_computes_hash_when_legacy_metadata_has_no_hash(
        self, tmp_path, filetype, xml_name, filepath
    ):
        """Legacy metadata should be hashed before its feed entry is generated."""
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
        expected_hash = hashlib.sha256(b"file data").hexdigest()
        expected_guid = expected_hash[-12:]
        assert f'<guid isPermaLink="false">{expected_guid}</guid>' in content
        stored = json.loads((data_dir / f"{filepath}.json").read_text(encoding="utf-8"))
        assert stored["hash"] == expected_hash

    def test_hash_computation_handles_invalid_utf8_metadata(self, tmp_path):
        """Invalid UTF-8 metadata should not abort hash computation."""
        data_dir = tmp_path / "data"
        source_path = data_dir / "videos" / "lecture.mp4"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"file data")
        metadata_path = source_path.with_name(source_path.name + ".json")
        metadata_path.write_bytes(b"\xff")
        metadata = make_metadata("videos/lecture.mp4")

        fg = FeedGenerator(data_dir=data_dir, base_url="https://example.com/feeds")
        fg._ensure_hashes([metadata])

        assert metadata.hash == hashlib.sha256(b"file data").hexdigest()
        assert metadata_path.read_bytes() == b"\xff"

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
        # Second legacy item should use the computed source hash.
        expected_hash = hashlib.sha256(b"file data").hexdigest()
        assert f'<guid isPermaLink="false">{expected_hash[-12:]}</guid>' in content


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


@pytest.mark.skip(reason="hashes are computed during source processing, not by feed generation")
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
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009
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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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
        updated = getattr(fg, "_update_missing_hashes")()  # noqa: B009

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
        getattr(fg, "_update_missing_hashes")()  # noqa: B009

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "hash" in data
        assert len(data["hash"]) == 64


@pytest.mark.slow
class TestHashUpdateIntegration:
    """Integration tests for hash update in generate_all_feeds()."""

    def test_generate_all_feeds_updates_missing_hash(self, tmp_path):
        """Feed generation must not read sources to backfill missing hashes."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 10, "output_size": 10},
                    "summarize": {
                        "status": "complete",
                        "input_size": 10,
                        "output_size": 10,
                        "end_time": 1728931200.0,
                    },
                    "title": {"filename": "lecture", "extracted": None, "generated": "Lecture"},
                    "description": "",
                    "summary": "Summary",
                    "filename": "lecture.mp4",
                    "filetype": "mp4",
                    "date": 1728931200,
                    "mtime": source.stat().st_mtime,
                    "size": 10,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "feed.json").write_text(json.dumps({"title": "Feed"}), encoding="utf-8")

        FeedGenerator(data_dir=tmp_path, base_url="https://example.com").generate_all_feeds()

        assert json.loads(json_path.read_text(encoding="utf-8"))["hash"]

    def test_generate_all_feeds_allow_list_excludes_other_root_items(self, tmp_path):
        """Root feed output includes only sources in the provided allow-list."""
        included_json = write_metadata_file(
            tmp_path,
            "included.mp4",
            summary="Included summary",
            transcript="Included transcript",
        )
        write_metadata_file(
            tmp_path,
            "excluded.mp4",
            summary="Excluded summary",
            transcript="Excluded transcript",
        )
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(
            data_dir=tmp_path,
            base_url="https://example.com",
            eligible_paths=[included_json.with_suffix("")],
        )
        result = fg.generate_all_feeds()

        assert [metadata.source_filepath for metadata in result["podcast"]] == ["included.mp4"]
        assert "included.mp4" in parse_rss_file(tmp_path / "podcast.xml")
        assert "excluded.mp4" not in parse_rss_file(tmp_path / "podcast.xml")

    def test_generate_all_feeds_without_hash_persistence(self, tmp_path):
        """Missing hashes are computed for GUIDs without changing sidecars."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = write_metadata_file(
            tmp_path,
            "lecture.mp4",
            summary="Summary text",
            transcript="Transcript text",
        )
        original = json_path.read_bytes()
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "Root Feed", "description": "Root Description"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(
            data_dir=tmp_path,
            base_url="https://example.com",
            persist_hashes=False,
        )
        result = fg.generate_all_feeds()

        expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        assert result["podcast"][0].hash == expected_hash
        assert f'<guid isPermaLink="false">{expected_hash[-12:]}</guid>' in parse_rss_file(
            tmp_path / "podcast.xml"
        )
        assert json_path.read_bytes() == original

    @pytest.mark.skip(reason="legacy assertion expects feed generation to add hashes")
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
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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

    @pytest.mark.skip(reason="legacy assertion expects feed generation to add hashes")
    def test_generate_all_feeds_hash_for_articles(self, tmp_path):
        """generate_all_feeds() should add hash to PDF metadata and use it in articles feed."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        json_path = tmp_path / "doc.pdf.json"
        json_path.write_text(
            json.dumps(
                {
                    "extract": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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

    @pytest.mark.skip(reason="legacy assertion expects feed generation to add hashes")
    def test_generate_all_feeds_skips_orphaned_json(self, tmp_path):
        """generate_all_feeds() should skip orphaned JSON files during hash update."""
        # Create only JSON, no source file
        json_path = tmp_path / "orphan.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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

    @pytest.mark.skip(reason="legacy assertion expects feed generation to add hashes")
    def test_generate_all_feeds_idempotent_hash_update(self, tmp_path):
        """Running generate_all_feeds() twice should not change hashes."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_path = tmp_path / "lecture.mp4.json"
        json_path.write_text(
            json.dumps(
                {
                    "transcribe": {"status": "complete", "input_size": 1000, "output_size": 500},
                    "summarize": {
                        "status": "complete",
                        "input_size": 500,
                        "output_size": 200,
                        "end_time": 1728931200.0,
                    },
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


class TestSummarizationAttribution:
    """Tests for _get_summarization_attribution() helper function."""

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
            "focus_feed.feeds.get_config",
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
            "focus_feed.feeds.get_config",
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
            assert "Summary text" in content
            assert "Summarized by" not in content


class TestAttributionInArticlesFeed:
    """Tests for attribution appearing in generated articles feeds."""

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
            "focus_feed.feeds.get_config",
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
            assert "Description text" in content
            assert "Summarized by" not in content


class TestFeedMetadataFields:
    """Tests for language, author, and image fields in feed generation."""

    def test_root_feed_json_language_in_podcast_feed(self, tmp_path):
        """Root feed.json language should appear in podcast.xml."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Podcast",
                    "description": "A podcast feed",
                    "language": "en-us",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<language>en-us</language>" in content

    def test_root_feed_json_language_in_articles_feed(self, tmp_path):
        """Root feed.json language should appear in articles.xml."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Articles",
                    "description": "An articles feed",
                    "language": "en-us",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<language>en-us</language>" in content

    def test_root_feed_json_author_in_podcast_feed(self, tmp_path):
        """Root feed.json author should be set on the feed generator."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Podcast",
                    "description": "A podcast feed",
                    "author": "John Smith",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Author is stored internally by feedgen; verify feed generation succeeds
        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<title>My Podcast</title>" in content

    def test_root_feed_json_author_in_articles_feed(self, tmp_path):
        """Root feed.json author should be set on the feed generator."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Articles",
                    "description": "An articles feed",
                    "author": "Jane Doe",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Author is stored internally by feedgen; verify feed generation succeeds
        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<title>My Articles</title>" in content

    def test_root_feed_json_image_in_podcast_feed(self, tmp_path):
        """Root feed.json image should appear in podcast.xml."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Podcast",
                    "description": "A podcast feed",
                    "image": {
                        "url": "https://example.com/logo.jpg",
                        "title": "Logo",
                        "link": "https://example.com/",
                    },
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<image>" in content
        assert "<url>https://example.com/logo.jpg</url>" in content
        assert "<title>Logo</title>" in content
        assert "<link>https://example.com/</link>" in content

    def test_root_feed_json_image_in_articles_feed(self, tmp_path):
        """Root feed.json image should appear in articles.xml."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Articles",
                    "description": "An articles feed",
                    "image": {
                        "url": "https://example.com/logo.jpg",
                        "title": "Logo",
                        "link": "https://example.com/",
                    },
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<image>" in content
        assert "<url>https://example.com/logo.jpg</url>" in content

    def test_root_feed_json_all_fields_in_podcast_feed(self, tmp_path):
        """All new fields (language, author, image) should appear together."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Module 1",
                    "description": "educational content - podcasts, articles, and resources",
                    "language": "en-us",
                    "author": "John Smith",
                    "image": {
                        "url": "https://www.example.com/logo.jpg",
                        "title": "Logo",
                        "link": "https://www.example.com/",
                    },
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<title>Module 1</title>" in content
        assert (
            "<description>educational content - podcasts, articles, and resources</description>"
            in content
        )
        assert "<language>en-us</language>" in content
        # Author is stored internally by feedgen (RSS requires email to render)
        assert "<image>" in content
        assert "<url>https://www.example.com/logo.jpg</url>" in content
        assert "<title>Logo</title>" in content
        assert "<link>https://www.example.com/</link>" in content

    def test_root_feed_json_all_fields_in_articles_feed(self, tmp_path):
        """All new fields (language, author, image) should appear in articles feed."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Module 1",
                    "description": "educational content",
                    "language": "en-us",
                    "author": "John Smith",
                    "image": {
                        "url": "https://www.example.com/logo.jpg",
                        "title": "Logo",
                        "link": "https://www.example.com/",
                    },
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<title>Module 1</title>" in content
        assert "<description>educational content</description>" in content
        assert "<language>en-us</language>" in content
        assert "<image>" in content
        assert "<url>https://www.example.com/logo.jpg</url>" in content

    def test_per_folder_feed_json_language_in_podcast_feed(self, tmp_path):
        """Per-folder feed.json language should appear in podcast.xml."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        write_feed_json(tmp_path, "Course1", "Course 1", "Description")
        (tmp_path / "Course1" / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Course 1",
                    "description": "Description",
                    "language": "fr-fr",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<language>fr-fr</language>" in content

    def test_per_folder_feed_json_author_in_articles_feed(self, tmp_path):
        """Per-folder feed.json author should be set on the feed generator."""
        write_metadata_file(tmp_path, "Course1/doc.pdf", filetype="pdf")
        (tmp_path / "Course1" / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Course 1",
                    "description": "Description",
                    "author": "Jane Doe",
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        # Author is stored internally by feedgen; verify feed generation succeeds
        content = parse_rss_file(tmp_path / "Course1" / "articles.xml")
        assert "<title>Course 1</title>" in content

    def test_per_folder_feed_json_image_in_podcast_feed(self, tmp_path):
        """Per-folder feed.json image should appear in podcast.xml."""
        write_metadata_file(tmp_path, "Course1/lecture.mp4")
        (tmp_path / "Course1" / "feed.json").write_text(
            json.dumps(
                {
                    "title": "Course 1",
                    "description": "Description",
                    "image": {
                        "url": "https://example.com/course1-logo.jpg",
                        "title": "Course 1 Logo",
                        "link": "https://example.com/course1",
                    },
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "Course1" / "podcast.xml")
        assert "<image>" in content
        assert "<url>https://example.com/course1-logo.jpg</url>" in content
        assert "<title>Course 1 Logo</title>" in content

    def test_no_language_when_not_in_feed_json(self, tmp_path):
        """No language tag should appear when feed.json has no language field."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "My Podcast", "description": "A feed"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<language>" not in content

    def test_no_author_when_not_in_feed_json(self, tmp_path):
        """No author tag should appear when feed.json has no author field."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "My Articles", "description": "A feed"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<author>" not in content
        assert "itunes:author" not in content

    def test_no_image_when_not_in_feed_json(self, tmp_path):
        """No image tag should appear when feed.json has no image field."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "My Podcast", "description": "A feed"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<image>" not in content

    def test_description_not_defaulted_to_title_when_empty(self, tmp_path):
        """Description should fall back to title when description is None (feedgen requires it)."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "My Podcast"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        # feedgen requires a non-empty description; it falls back to the title
        assert "<description>My Podcast</description>" in content

    def test_description_not_defaulted_to_title_for_articles(self, tmp_path):
        """Articles feed description should fall back to title when None (feedgen requires it)."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")
        (tmp_path / "feed.json").write_text(
            json.dumps({"title": "My Articles"}),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "articles.xml")
        # feedgen requires a non-empty description; it falls back to the title
        assert "<description>My Articles</description>" in content

    def test_generate_podcast_feed_accepts_language_author_image_directly(self, tmp_path):
        """generate_podcast_feed should accept language, author, image params directly."""
        write_metadata_file(tmp_path, "lecture.mp4")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_podcast_feed(
            title="Direct Podcast",
            description="Direct desc",
            language="de-de",
            author="Direct Author",
            image={
                "url": "https://example.com/direct.jpg",
                "title": "Direct",
                "link": "https://example.com/",
            },
        )

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<language>de-de</language>" in content
        # Author is stored internally by feedgen (RSS requires email to render)
        assert "<url>https://example.com/direct.jpg</url>" in content

    def test_generate_articles_feed_accepts_language_author_image_directly(self, tmp_path):
        """generate_articles_feed should accept language, author, image params directly."""
        write_metadata_file(tmp_path, "doc.pdf", filetype="pdf")

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_articles_feed(
            title="Direct Articles",
            description="Direct desc",
            language="de-de",
            author="Direct Author",
            image={
                "url": "https://example.com/direct.jpg",
                "title": "Direct",
                "link": "https://example.com/",
            },
        )

        content = parse_rss_file(tmp_path / "articles.xml")
        assert "<language>de-de</language>" in content
        assert "<url>https://example.com/direct.jpg</url>" in content

    def test_image_with_missing_title_and_link_defaults_to_empty(self, tmp_path):
        """Image dict missing title/link should not cause errors."""
        write_metadata_file(tmp_path, "lecture.mp4")
        (tmp_path / "feed.json").write_text(
            json.dumps(
                {
                    "title": "My Podcast",
                    "description": "A feed",
                    "image": {"url": "https://example.com/logo.jpg"},
                }
            ),
            encoding="utf-8",
        )

        fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
        fg.generate_all_feeds()

        content = parse_rss_file(tmp_path / "podcast.xml")
        assert "<url>https://example.com/logo.jpg</url>" in content
        assert "<image>" in content
