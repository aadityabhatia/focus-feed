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
    FeedValidationResult,
    _markdown_to_html,
    get_mime_type,
    validate_feed,
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
                lambda xml: xml.replace(' type="audio/mp4"', ' type="application/octet-stream"'),
                "expected a supported media MIME type",
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

    @pytest.mark.parametrize("mime_type", sorted({get_mime_type(t) for t in MEDIA_FILETYPES}))
    def test_podcast_enclosure_accepts_supported_media_mime_types(self, mime_type):
        xml = _VALID_PODCAST_XML.replace('type="audio/mp4"', f'type="{mime_type}"')
        result = validate_feed(xml, feed_name="podcast.xml", is_podcast=True)
        assert result.is_valid

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

    def test_three_space_indented_list_is_nested(self):
        """Three-space-indented bullets should nest under ordered list items."""
        md = "1. **Historical Contextual Analysis**\n   - Examines military events"

        result = _markdown_to_html(md)

        assert "<li><strong>Historical Contextual Analysis</strong><ul>" in result
        assert "<li>Examines military events</li>" in result
        assert "</ul>\n</li>\n</ol>" in result

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

    def test_consecutive_horizontal_rules_are_not_parsed_as_setext_heading(self):
        """Repeated standalone dashes in generated summaries should remain rules."""
        result = _markdown_to_html("---\n---\n")

        assert result == "<hr />\n<hr />"
        assert "<h2>---</h2>" not in result

    def test_single_dash_line_preserves_valid_setext_heading(self):
        """A single dash underline remains a valid level-two heading."""
        assert _markdown_to_html("Heading\n---") == "<h2>Heading</h2>"

    def test_markdown_cdata_wrapping(self, tmp_path):
        """HTML output from _markdown_to_html should be CDATA-wrapped in feed output."""
        write_metadata_file(
            tmp_path,
            "documents/doc.pdf",
            filetype="pdf",
            description="**Bold** and *italic* text with <tags>",
        )

        # Mock get_config to ensure no attribution is added (deterministic test).
        # Without this, other tests' config mocks can leak and add an attribution
        # paragraph, which would prevent <p> tag stripping for single paragraphs.
        with mock.patch("focus_feed.feeds.get_config", side_effect=Exception):
            fg = FeedGenerator(data_dir=tmp_path, base_url="https://example.com/feeds")
            fg.generate_articles_feed()

        content = parse_rss_file(tmp_path / "articles.xml")
        # The HTML tags from markdown should be inside CDATA (not XML-escaped)
        assert "<![CDATA[" in content
        assert "<strong>Bold</strong>" in content
        assert "<em>italic</em>" in content
        # The raw <tags> from the input should be escaped by the markdown converter
        # but NOT double-escaped by feedgen (since it's in CDATA)
        assert "&lt;tags&gt;" in content
        # Verify the CDATA section contains the HTML without <p> wrapper
        # (single paragraph <p> tags are stripped by _markdown_to_html)
        assert (
            "<![CDATA[<strong>Bold</strong> and <em>italic</em> text with &lt;tags&gt;]]>"
            in content
        )
