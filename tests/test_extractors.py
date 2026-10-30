"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest import mock

from pypdf import PdfWriter
import pytest

from focus_feed.services.extractors import (
    HTMLExtractor,
    PDFExtractor,
    _clean_title,
    _sanitize_content,
    get_audio_metadata,
)

# ---------------------------------------------------------------------------
# Helper functions for creating test fixtures
# ---------------------------------------------------------------------------


def create_test_pdf(
    path: Path,
    title: str | None = "Test PDF Title",
    author: str | None = "Test Author",
    creation_date: datetime | None = None,
) -> None:
    """Create a minimal PDF file with optional metadata.

    Args:
        path: Where to write the PDF file.
        title: Title to embed in PDF metadata, or None to omit.
        author: Author to embed in PDF metadata, or None to omit.
        creation_date: Creation date to embed, or None to omit.
            Converted to PDF date format string (D:YYYYMMDDHHMMSS).
    """
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    metadata: dict[str, Any] = {}
    if title is not None:
        metadata["/Title"] = title
    if author is not None:
        metadata["/Author"] = author
    if creation_date is not None:
        # PDF date format: D:YYYYMMDDHHMMSS
        metadata["/CreationDate"] = creation_date.strftime("D:%Y%m%d%H%M%S")

    if metadata:
        writer.add_metadata(metadata)

    with open(path, "wb") as f:
        writer.write(f)


def create_test_pdf_with_text(
    path: Path,
    text: str,
    title: str | None = "Test PDF Title",
    author: str | None = "Test Author",
) -> None:
    """Create a PDF file with text content and optional metadata.

    Uses pypdf's PdfWriter to create a page with a content stream
    containing the given text so that ``extract_text()`` can read it back.

    Args:
        path: Where to write the PDF file.
        text: The text content to embed in the PDF page.
        title: Title to embed in PDF metadata, or None to omit.
        author: Author to embed in PDF metadata, or None to omit.
    """
    from pypdf import PageObject
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    page = PageObject.create_blank_page(width=200, height=200)

    # Build a minimal content stream with the text.
    content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject("/Contents")] = stream

    # Add a Helvetica font resource so text is extractable.
    font_dict = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_indirect = writer._add_object(font_dict)
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): font_indirect,
                }
            ),
        }
    )
    page[NameObject("/Resources")] = resources

    writer.add_page(page)

    metadata: dict[str, Any] = {}
    if title is not None:
        metadata["/Title"] = title
    if author is not None:
        metadata["/Author"] = author
    if metadata:
        writer.add_metadata(metadata)

    with open(path, "wb") as f:
        writer.write(f)


def create_test_html(
    path: Path,
    title: str | None = "Test HTML Title",
    description: str | None = "Test description from meta tag",
    date: str | None = "2026-10-14T12:00:00Z",
    pubdate: str | None = None,
    time_datetime: str | None = None,
) -> None:
    """Create an HTML file with optional metadata.

    Args:
        path: Where to write the HTML file.
        title: Content of the <title> tag, or None to omit.
        description: Content of <meta name="description">, or None to omit.
        date: Content of <meta name="date">, or None to omit.
        pubdate: Content of <meta name="pubdate">, or None to omit.
        time_datetime: Value of <time datetime="...">, or None to omit.
    """
    html_parts = ["<!DOCTYPE html>", "<html>", "<head>"]

    if title is not None:
        html_parts.append(f"<title>{title}</title>")

    if description is not None:
        html_parts.append(f'<meta name="description" content="{description}">')

    if date is not None:
        html_parts.append(f'<meta name="date" content="{date}">')

    if pubdate is not None:
        html_parts.append(f'<meta name="pubdate" content="{pubdate}">')

    html_parts.append("</head>")
    html_parts.append("<body>")

    if time_datetime is not None:
        html_parts.append(f'<time datetime="{time_datetime}">October 14, 2026</time>')

    html_parts.append("<p>Test content</p>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    path.write_text("\n".join(html_parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# _clean_title helper tests
# ---------------------------------------------------------------------------


class TestCleanTitle:
    """Tests for the _clean_title helper function."""

    def test_clean_title_replaces_single_underscore(self):
        """Single underscore should be replaced with a space."""
        assert _clean_title("my_title") == "my title"

    def test_clean_title_replaces_multiple_underscores(self):
        """Multiple underscores should each be replaced with spaces."""
        assert _clean_title("my__title") == "my  title"

    def test_clean_title_replaces_leading_trailing_underscores(self):
        """Leading and trailing underscores should be replaced with spaces."""
        assert _clean_title("_title_") == " title "

    def test_clean_title_no_underscores(self):
        """Title without underscores should be returned unchanged."""
        assert _clean_title("My Title") == "My Title"

    def test_clean_title_none_returns_none(self):
        """None input should return None."""
        assert _clean_title(None) is None

    def test_clean_title_empty_string(self):
        """Empty string should return empty string."""
        assert _clean_title("") == ""

    def test_clean_title_only_underscores(self):
        """String of only underscores should become spaces."""
        assert _clean_title("___") == "   "


class TestSanitizeContent:
    """Tests for the _sanitize_content helper function."""

    def test_sanitize_removes_null_bytes(self):
        """Null bytes should be removed from content."""
        assert _sanitize_content("hello\x00world") == "helloworld"

    def test_sanitize_removes_control_chars(self):
        """Control characters (except tab/newline/CR) should be removed."""
        # 0x01 (SOH), 0x1d (GS), 0x03 (ETX) should be removed
        assert _sanitize_content("hello\x01world\x1dtest\x03") == "helloworldtest"

    def test_sanitize_preserves_tab_newline_cr(self):
        """Tab, newline, and carriage return should be preserved."""
        assert _sanitize_content("hello\tworld\nfoo\rbar") == "hello\tworld\nfoo\rbar"

    def test_sanitize_removes_del_char(self):
        """DEL character (0x7f) should be removed."""
        assert _sanitize_content("hello\x7fworld") == "helloworld"

    def test_sanitize_none_returns_none(self):
        """None input should return None."""
        assert _sanitize_content(None) is None

    def test_sanitize_empty_string(self):
        """Empty string should return empty string."""
        assert _sanitize_content("") == ""

    def test_sanitize_no_control_chars(self):
        """Normal text should be unchanged."""
        assert _sanitize_content("Hello, World! This is a test.") == "Hello, World! This is a test."

    def test_sanitize_preserves_unicode(self):
        """Unicode characters should be preserved."""
        assert _sanitize_content("Hello \u2014 World \u00e9") == "Hello \u2014 World \u00e9"

    def test_sanitize_real_pdf_binary_content(self):
        """Content with binary garbage from PDFs should be cleaned."""
        # Simulate the binary content found in C102RF_FM_5-0 PDF
        dirty = "solved.\nPIN\x1d 082115\x10000\x03"
        result = _sanitize_content(dirty)
        assert result is not None
        assert "\x1d" not in result
        assert "\x10" not in result
        assert "\x03" not in result
        assert result == "solved.\nPIN 082115000"


# ---------------------------------------------------------------------------
# Shared parameterized tests for PDF and HTML extractors
# ---------------------------------------------------------------------------
# These tests cover behavior that is identical across both PDFExtractor and
# HTMLExtractor (file size, date type, required keys, string path, etc.).
# Rather than duplicating the same test logic in both TestPDFExtractor and
# TestHTMLExtractor, we parameterize across the two extractor classes.


class TestExtractorSharedBehavior:
    """Parameterized tests covering behavior common to PDF and HTML extractors."""

    @pytest.fixture(
        params=[
            pytest.param(("pdf", PDFExtractor, create_test_pdf, "document.pdf"), id="pdf"),
            pytest.param(("html", HTMLExtractor, create_test_html, "article.html"), id="html"),
        ]
    )
    def extractor_info(self, request):
        """Provide (filetype, extractor_class, create_func, filename) for each extractor."""
        return request.param

    def test_extract_returns_all_required_keys(self, extractor_info, tmp_path):
        """Result dict should contain all required keys."""
        _filetype, extractor_class, create_func, filename = extractor_info
        path = tmp_path / filename
        create_func(path, title="Keys Test")

        extractor = extractor_class()
        result = extractor.extract(path)

        required_keys = {
            "title",
            "extracted_title",
            "description",
            "date",
            "publication_date",
            "size",
            "filetype",
            "content",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_extract_file_size_in_bytes(self, extractor_info, tmp_path):
        """File size should be the actual file size in bytes."""
        _filetype, extractor_class, create_func, filename = extractor_info
        path = tmp_path / filename
        create_func(path, title="Size Test")

        extractor = extractor_class()
        result = extractor.extract(path)

        expected_size = path.stat().st_size
        assert result["size"] == expected_size
        assert isinstance(result["size"], int)

    def test_extract_date_is_iso8601(self, extractor_info, tmp_path):
        """Date should be stored as UNIX epoch (float with ms precision)."""
        filetype, extractor_class, create_func, _filename = extractor_info
        path = tmp_path / f"date_test.{filetype}"
        if filetype == "pdf":
            create_func(
                path,
                title="Date Test",
                author=None,
                creation_date=datetime(2026, 6, 15, 8, 30, 0),
            )
        else:
            create_func(path, title="Date Test", date="2026-06-15T08:30:00Z")

        extractor = extractor_class()
        result = extractor.extract(path)

        assert isinstance(result["date"], float)
        assert result["date"] > 0

    def test_extract_fallback_title_to_filename(self, extractor_info, tmp_path):
        """Without a title, should fall back to filename (stem, underscores cleaned)."""
        filetype, extractor_class, create_func, _filename = extractor_info
        path = tmp_path / f"my_document.{filetype}"
        if filetype == "pdf":
            create_func(path, title=None, author=None, creation_date=None)
        else:
            create_func(path, title=None, description="Some description")

        extractor = extractor_class()
        result = extractor.extract(path)

        assert result["title"] == "my document"
        assert result["filetype"] == filetype

    def test_extract_fallback_date_to_mtime(self, extractor_info, tmp_path):
        """Without a date, should fall back to file modification time."""
        filetype, extractor_class, create_func, _filename = extractor_info
        path = tmp_path / f"nodate.{filetype}"
        if filetype == "pdf":
            create_func(path, title="No Date", author=None, creation_date=None)
        else:
            create_func(path, title="No Date", date=None)

        extractor = extractor_class()
        result = extractor.extract(path)

        mtime = int(path.stat().st_mtime)
        assert isinstance(result["date"], float)
        assert abs(result["date"] - mtime) < 5

    def test_extract_empty_title_falls_back_to_filename(self, extractor_info, tmp_path):
        """Empty title string should fall back to filename (underscores cleaned)."""
        filetype, extractor_class, create_func, _filename = extractor_info
        path = tmp_path / f"empty_title.{filetype}"
        if filetype == "pdf":
            create_func(path, title="", author="Author")
        else:
            path.write_text(
                "<html><head><title></title></head><body>Test</body></html>",
                encoding="utf-8",
            )

        extractor = extractor_class()
        result = extractor.extract(path)

        assert result["title"] == "empty title"

    def test_extract_cleans_underscores_from_fallback_filename(self, extractor_info, tmp_path):
        """Underscores in fallback filename should be replaced with spaces."""
        filetype, extractor_class, create_func, _filename = extractor_info
        path = tmp_path / f"my_document.{filetype}"
        if filetype == "pdf":
            create_func(path, title=None, author=None)
        else:
            create_func(path, title=None, description=None)

        extractor = extractor_class()
        result = extractor.extract(path)

        assert result["title"] == "my document"

    def test_extract_accepts_string_path(self, extractor_info, tmp_path):
        """Extractor should accept a string path, not just Path."""
        filetype, extractor_class, create_func, _filename = extractor_info
        path = tmp_path / f"string_path.{filetype}"
        create_func(path, title="String Path Test")

        extractor = extractor_class()
        result = extractor.extract(str(path))

        assert result["title"] == "string path"
        assert result["extracted_title"] == "String Path Test"
        assert result["filetype"] == filetype

    def test_extract_with_full_metadata(self, extractor_info, tmp_path):
        """With full metadata, should extract all fields."""
        filetype, extractor_class, create_func, filename = extractor_info
        path = tmp_path / filename
        if filetype == "pdf":
            create_func(
                path,
                title="My Title",
                author="Author Name",
                creation_date=datetime(2026, 10, 14, 12, 0, 0),
            )
            expected_description = "Author Name"
        else:
            create_func(
                path,
                title="My Title",
                description="An interesting article",
                date="2026-10-14T12:00:00Z",
            )
            expected_description = "An interesting article"

        extractor = extractor_class()
        result = extractor.extract(path)

        assert result["title"] == filename.split(".")[0]
        assert result["extracted_title"] == "My Title"
        assert result["description"] == expected_description
        assert result["filetype"] == filetype
        assert result["size"] > 0
        assert isinstance(result["date"], float)
        assert result["date"] > 0


# ---------------------------------------------------------------------------
# PDFExtractor tests
# ---------------------------------------------------------------------------


class TestPDFExtractor:
    """Tests for the PDFExtractor class.

    NOTE: Tests covering behavior shared with HTMLExtractor (file size, date
    type, required keys, string path, fallback title, etc.) are consolidated
    in TestExtractorSharedBehavior above to avoid duplication.
    """

    def test_extract_handles_corrupted_pdf(self, tmp_path):
        """Corrupted PDF should fall back to filename and mtime without crashing."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"not a real pdf file content")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "corrupted"
        assert result["extracted_title"] is None
        assert result["description"] is None
        assert result["filetype"] == "pdf"
        assert result["size"] > 0
        # Date should fall back to file mtime (as epoch float)
        assert isinstance(result["date"], float)
        assert result["date"] > 0
        # publication_date should be None when no document metadata date
        assert result["publication_date"] is None

    def test_extract_publication_date_from_pdf_metadata(self, tmp_path):
        """publication_date should be extracted from PDF CreationDate metadata."""
        pdf_path = tmp_path / "dated.pdf"
        create_test_pdf(
            pdf_path,
            title="Dated PDF",
            author="Author",
            creation_date=datetime(2026, 6, 15, 8, 30, 0),
        )

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["publication_date"] is not None
        assert isinstance(result["publication_date"], float)
        # Verify the date matches the creation date (2026-06-15 08:30:00 UTC)
        parsed = datetime.fromtimestamp(result["publication_date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 6
        assert parsed.day == 15

    def test_extract_publication_date_none_when_no_creation_date(self, tmp_path):
        """publication_date should be None when PDF has no CreationDate."""
        pdf_path = tmp_path / "no_date.pdf"
        create_test_pdf(pdf_path, title="No Date", author=None, creation_date=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["publication_date"] is None
        # date should fall back to file mtime
        assert result["date"] is not None
        assert isinstance(result["date"], float)

    def test_extract_description_is_none_when_no_author(self, tmp_path):
        """Description should be None when PDF has no author."""
        pdf_path = tmp_path / "no_author.pdf"
        create_test_pdf(pdf_path, title="No Author", author=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["description"] is None

    def test_extract_cleans_underscores_from_pdf_title(self, tmp_path):
        """Underscores in PDF metadata title should be replaced with spaces in extracted_title."""
        pdf_path = tmp_path / "document.pdf"
        create_test_pdf(pdf_path, title="my_pdf_title", author=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        # Title is always filename-derived
        assert result["title"] == "document"
        # Extracted title should have underscores cleaned
        assert result["extracted_title"] == "my pdf title"

    def test_extract_content_from_pdf_with_text(self, tmp_path):
        """PDF with text content should extract plaintext into the content field."""
        pdf_path = tmp_path / "with_text.pdf"
        create_test_pdf_with_text(pdf_path, text="Hello World from PDF")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["content"] is not None
        assert "Hello World from PDF" in result["content"]

    def test_extract_content_is_none_for_pdf_without_text(self, tmp_path):
        """PDF with no text content should have content=None."""
        pdf_path = tmp_path / "no_text.pdf"
        create_test_pdf(pdf_path, title="No Text")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["content"] is None

    def test_extract_content_is_none_for_corrupted_pdf(self, tmp_path):
        """Corrupted PDF should have content=None (graceful fallback)."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"not a real pdf file content")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["content"] is None


# ---------------------------------------------------------------------------
# HTMLExtractor tests
# ---------------------------------------------------------------------------


class TestHTMLExtractor:
    """Tests for the HTMLExtractor class.

    NOTE: Tests covering behavior shared with PDFExtractor (file size, date
    type, required keys, string path, fallback title, etc.) are consolidated
    in TestExtractorSharedBehavior above to avoid duplication.
    """

    def test_extract_description_from_meta(self, tmp_path):
        """Description should come from <meta name='description'>."""
        html_path = tmp_path / "desc.html"
        create_test_html(
            html_path,
            title="Desc Test",
            description="This is a meta description",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["description"] == "This is a meta description"

    def test_extract_description_none_when_no_meta(self, tmp_path):
        """Description should be None when no meta description is present."""
        html_path = tmp_path / "no_desc.html"
        create_test_html(html_path, title="No Desc", description=None)

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["description"] is None

    def test_extract_minimal_html(self, tmp_path):
        """Minimal HTML with only a title should work correctly."""
        html_path = tmp_path / "minimal.html"
        html_path.write_text(
            "<!DOCTYPE html><html><head><title>Minimal</title></head><body></body></html>",
            encoding="utf-8",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "minimal"
        assert result["extracted_title"] == "Minimal"
        assert result["description"] is None
        assert result["filetype"] == "html"
        assert result["size"] > 0

    def test_extract_date_from_pubdate_meta(self, tmp_path):
        """Date should be extracted from <meta name='pubdate'> when present."""
        html_path = tmp_path / "pubdate.html"
        create_test_html(
            html_path,
            title="Pubdate Test",
            date=None,
            pubdate="2026-09-01T10:00:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        parsed = datetime.fromtimestamp(result["date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 9
        assert parsed.day == 1

    def test_extract_date_from_time_tag(self, tmp_path):
        """Date should be extracted from <time datetime='...'> when present."""
        html_path = tmp_path / "time_tag.html"
        create_test_html(
            html_path,
            title="Time Tag Test",
            date=None,
            time_datetime="2026-08-20T14:30:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        parsed = datetime.fromtimestamp(result["date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 8
        assert parsed.day == 20

    def test_extract_handles_malformed_html(self, tmp_path):
        """Malformed HTML should fall back to filename and mtime without crashing."""
        html_path = tmp_path / "malformed.html"
        html_path.write_text(
            "<html><head><title>Broken",
            encoding="utf-8",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "malformed"
        assert result["filetype"] == "html"
        assert result["size"] > 0
        # publication_date should be None when no date metadata is available
        assert result["publication_date"] is None

    def test_extract_publication_date_from_meta_date(self, tmp_path):
        """publication_date should be extracted from <meta name='date'>."""
        html_path = tmp_path / "pub_date.html"
        create_test_html(
            html_path,
            title="Pub Date Test",
            date="2026-06-15T08:30:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["publication_date"] is not None
        assert isinstance(result["publication_date"], float)
        parsed = datetime.fromtimestamp(result["publication_date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 6
        assert parsed.day == 15

    def test_extract_publication_date_from_pubdate_meta(self, tmp_path):
        """publication_date should be extracted from <meta name='pubdate'>."""
        html_path = tmp_path / "pubdate.html"
        create_test_html(
            html_path,
            title="Pubdate Test",
            date=None,
            pubdate="2026-09-01T10:00:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["publication_date"] is not None
        parsed = datetime.fromtimestamp(result["publication_date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 9
        assert parsed.day == 1

    def test_extract_publication_date_from_time_tag(self, tmp_path):
        """publication_date should be extracted from <time datetime='...'>."""
        html_path = tmp_path / "time_date.html"
        create_test_html(
            html_path,
            title="Time Date Test",
            date=None,
            time_datetime="2026-08-20T14:30:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["publication_date"] is not None
        parsed = datetime.fromtimestamp(result["publication_date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 8
        assert parsed.day == 20

    def test_extract_publication_date_none_when_no_date_metadata(self, tmp_path):
        """publication_date should be None when no date metadata is available."""
        html_path = tmp_path / "no_date.html"
        create_test_html(html_path, title="No Date", date=None, pubdate=None, time_datetime=None)

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["publication_date"] is None
        # date should fall back to file mtime
        assert result["date"] is not None
        assert isinstance(result["date"], float)

    def test_extract_date_date_only_format(self, tmp_path):
        """Date in YYYY-MM-DD format should be parsed correctly."""
        html_path = tmp_path / "date_only.html"
        create_test_html(
            html_path,
            title="Date Only",
            date="2026-10-14",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        parsed = datetime.fromtimestamp(result["date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 10
        assert parsed.day == 14

    def test_extract_date_with_slash_format(self, tmp_path):
        """Date in YYYY/MM/DD format should be parsed correctly."""
        html_path = tmp_path / "slash_date.html"
        create_test_html(
            html_path,
            title="Slash Date",
            date="2026/10/14",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        parsed = datetime.fromtimestamp(result["date"], tz=UTC)
        assert parsed.year == 2026
        assert parsed.month == 10
        assert parsed.day == 14

    def test_extract_cleans_underscores_from_html_title(self, tmp_path):
        """Underscores in <title> tag should be replaced with spaces."""
        html_path = tmp_path / "article.html"
        create_test_html(
            html_path,
            title="my_article_title",
            description="Test description",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "article"
        assert result["extracted_title"] == "my article title"

    def test_extract_content_from_html(self, tmp_path):
        """HTML content should be extracted as plaintext (tags removed)."""
        html_path = tmp_path / "content.html"
        html_path.write_text(
            "<html><head><title>Test</title></head>"
            "<body><h1>Heading</h1><p>This is a paragraph.</p></body></html>",
            encoding="utf-8",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["content"] is not None
        assert "Heading" in result["content"]
        assert "This is a paragraph." in result["content"]
        # Tags should be stripped
        assert "<h1>" not in result["content"]
        assert "<p>" not in result["content"]

    def test_extract_content_none_for_malformed_html(self, tmp_path):
        """Malformed HTML should still extract content if possible."""
        html_path = tmp_path / "malformed.html"
        html_path.write_text(
            "<html><head><title>Broken",
            encoding="utf-8",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        # BeautifulSoup should still extract whatever text it can
        assert result["content"] is not None
        assert "Broken" in result["content"]


# ---------------------------------------------------------------------------
# _parse_date helper tests
# ---------------------------------------------------------------------------


class TestParseDate:
    """Tests for the HTMLExtractor._parse_date static method."""

    def test_parse_iso8601_with_z(self):
        """ISO8601 with Z suffix should parse correctly."""
        result = HTMLExtractor._parse_date("2026-10-14T12:00:00Z")
        assert result is not None
        assert result.year == 2026
        assert result.month == 10
        assert result.day == 14
        assert result.tzinfo is not None

    def test_parse_iso8601_with_offset(self):
        """ISO8601 with timezone offset should parse correctly."""
        result = HTMLExtractor._parse_date("2026-10-14T12:00:00+05:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_parse_iso8601_naive(self):
        """ISO8601 without timezone should be made timezone-aware."""
        result = HTMLExtractor._parse_date("2026-10-14T12:00:00")
        assert result is not None
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_parse_date_only(self):
        """Date-only format YYYY-MM-DD should parse correctly."""
        result = HTMLExtractor._parse_date("2026-10-14")
        assert result is not None
        assert result.year == 2026
        assert result.month == 10
        assert result.day == 14

    def test_parse_slash_format(self):
        """YYYY/MM/DD format should parse correctly."""
        result = HTMLExtractor._parse_date("2026/10/14")
        assert result is not None
        assert result.year == 2026

    def test_parse_month_name_format(self):
        """'Month DD, YYYY' format should parse correctly."""
        result = HTMLExtractor._parse_date("October 14, 2026")
        assert result is not None
        assert result.year == 2026
        assert result.month == 10
        assert result.day == 14

    def test_parse_abbreviated_month_format(self):
        """'Mon DD, YYYY' format should parse correctly."""
        result = HTMLExtractor._parse_date("Oct 14, 2026")
        assert result is not None
        assert result.year == 2026
        assert result.month == 10
        assert result.day == 14

    def test_parse_invalid_returns_none(self):
        """Invalid date string should return None."""
        result = HTMLExtractor._parse_date("not a date")
        assert result is None

    def test_parse_empty_string_returns_none(self):
        """Empty string should return None."""
        result = HTMLExtractor._parse_date("")
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests with Metadata model
# ---------------------------------------------------------------------------


class TestMetadataCompatibility:
    """Tests verifying extractor output is compatible with the Metadata model."""

    def test_pdf_result_compatible_with_metadata(self, tmp_path):
        """PDF extractor output should be usable to create a Metadata object."""
        from focus_feed.metadata import Metadata, StepMap

        pdf_path = tmp_path / "doc.pdf"
        create_test_pdf(pdf_path, title="Test Doc", author="Author Name")

        extractor = PDFExtractor()
        extracted = extractor.extract(pdf_path)

        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=extracted["size"],
                output_size=(
                    len(extracted["content"].encode("utf-8")) if extracted["content"] else 0
                ),
            ),
            summarize=StepMap(
                status="complete",
                input_size=extracted["size"],
                output_size=0,
            ),
            title=extracted["title"],
            description=extracted["description"] or "",
            filename=pdf_path.name,
            filetype=extracted["filetype"],
            date=extracted["date"],
            size=extracted["size"],
            content=extracted["content"],
        )

        assert metadata.final_title == "doc"
        assert metadata.filetype == "pdf"
        assert metadata.size == extracted["size"]
        assert metadata.content == extracted["content"]
        assert metadata.feed_eligible is True

    def test_html_result_compatible_with_metadata(self, tmp_path):
        """HTML extractor output should be usable to create a Metadata object."""
        from focus_feed.metadata import Metadata, StepMap

        html_path = tmp_path / "page.html"
        create_test_html(
            html_path,
            title="Test Page",
            description="A test page description",
        )

        extractor = HTMLExtractor()
        extracted = extractor.extract(html_path)

        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=extracted["size"],
                output_size=(
                    len(extracted["content"].encode("utf-8")) if extracted["content"] else 0
                ),
            ),
            summarize=StepMap(
                status="complete",
                input_size=extracted["size"],
                output_size=0,
            ),
            title=extracted["title"],
            description=extracted["description"] or "",
            filename=html_path.name,
            filetype=extracted["filetype"],
            date=extracted["date"],
            size=extracted["size"],
            content=extracted["content"],
        )

        assert metadata.final_title == "page"
        assert metadata.filetype == "html"
        assert metadata.description == "A test page description"
        assert metadata.content == extracted["content"]
        assert metadata.feed_eligible is True

    def test_pdf_result_serializes_to_json(self, tmp_path):
        """PDF extractor output should be JSON serializable."""
        pdf_path = tmp_path / "serialize.pdf"
        create_test_pdf(pdf_path, title="Serialize Test")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        # Should not raise
        json_str = json.dumps(result)
        restored = json.loads(json_str)
        assert restored["title"] == "serialize"
        assert restored["extracted_title"] == "Serialize Test"
        assert restored["filetype"] == "pdf"

    def test_html_result_serializes_to_json(self, tmp_path):
        """HTML extractor output should be JSON serializable."""
        html_path = tmp_path / "serialize.html"
        create_test_html(html_path, title="Serialize Test")

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        # Should not raise
        json_str = json.dumps(result)
        restored = json.loads(json_str)
        assert restored["title"] == "serialize"
        assert restored["extracted_title"] == "Serialize Test"
        assert restored["filetype"] == "html"


# ---------------------------------------------------------------------------
# get_audio_metadata tests
# ---------------------------------------------------------------------------


class TestGetAudioMetadata:
    """Tests for the get_audio_metadata function."""

    @pytest.fixture
    def mock_probe(self, tmp_path):
        """Set up a mock audio file and return helper for patching."""
        audio_path = tmp_path / "test.mp4"
        audio_path.write_bytes(b"fake audio data")
        return audio_path

    def test_returns_none_when_ffprobe_not_found(self, mock_probe):
        """Should return None when ffprobe is not on PATH."""
        with mock.patch("shutil.which", return_value=None):
            result = get_audio_metadata(mock_probe)
        assert result is None

    def test_returns_metadata_dict_when_ffprobe_succeeds(self, mock_probe):
        """Should return a dict with expected keys when ffprobe succeeds."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "3600.500000",
                "bit_rate": "128000",
                "tags": {
                    "creation_time": "2024-01-15T10:30:00.000000Z",
                },
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "3600.500000",
                    "bit_rate": "128000",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["type"] == "mov,mp4,m4a,3gp,3g2,mj2"
        assert result["duration"] == 3600.5
        assert result["creation_time"] == "2024-01-15T10:30:00.000000Z"
        assert result["bit_rate"] == 128000

    @pytest.mark.parametrize(
        "side_effect,expected",
        [
            (subprocess.CalledProcessError(1, "ffprobe"), None),
            (OSError("command not found"), None),
        ],
    )
    def test_returns_none_on_subprocess_error(self, mock_probe, side_effect, expected):
        """Should return None when ffprobe command fails."""
        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = side_effect
            result = get_audio_metadata(mock_probe)
        assert result is expected

    def test_returns_none_on_json_decode_error(self, mock_probe):
        """Should return None when ffprobe output is not valid JSON."""
        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout="not valid json",
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)
        assert result is None

    def test_handles_missing_creation_time(self, mock_probe):
        """Should return None for creation_time when tag is absent."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "120.000000",
                "tags": {},
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "120.000000",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["creation_time"] is None
        assert result["duration"] == 120.0
        assert result["bit_rate"] is None

    def test_falls_back_to_format_duration_when_no_audio_stream(self, mock_probe):
        """Should use format-level duration when no audio stream exists."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "600.000000",
                "bit_rate": "256000",
                "tags": {
                    "creation_time": "2023-06-01T08:00:00Z",
                },
            },
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "600.000000",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["duration"] == 600.0
        assert result["bit_rate"] == 256000
        assert result["creation_time"] == "2023-06-01T08:00:00Z"

    def test_handles_empty_streams_list(self, mock_probe):
        """Should return 0 duration when no streams and no format duration."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["duration"] == 0.0
        assert result["creation_time"] is None
        assert result["bit_rate"] is None

    def test_accepts_string_path(self, mock_probe):
        """Should accept a string path, not just Path."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "100.0",
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(str(mock_probe))

        assert result is not None
        assert result["type"] == "mov,mp4,m4a,3gp,3g2,mj2"
        # Verify the path was passed to ffprobe as a string
        call_args = mock_run.call_args
        assert str(mock_probe) in call_args[0][0]

    @pytest.mark.parametrize(
        "duration_str,expected_duration",
        [
            ("180.5", 180.5),
            ("100.0", 100.0),
            ("0.0", 0.0),
        ],
    )
    def test_duration_is_float(self, mock_probe, duration_str, expected_duration):
        """Duration should be a float."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": duration_str,
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert isinstance(result["duration"], float)
        assert result["duration"] == expected_duration

    @pytest.mark.parametrize(
        "bit_rate_str,expected_bit_rate",
        [
            ("320000", 320000),
            ("128000", 128000),
        ],
    )
    def test_bit_rate_is_int(self, mock_probe, bit_rate_str, expected_bit_rate):
        """Bit rate should be an int."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "bit_rate": bit_rate_str,
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert isinstance(result["bit_rate"], int)
        assert result["bit_rate"] == expected_bit_rate

    def test_returns_all_required_keys(self, mock_probe):
        """Result dict should contain all required keys."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "100.0",
                "bit_rate": "128000",
                "tags": {
                    "creation_time": "2024-01-15T10:30:00.000000Z",
                },
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "100.0",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        required_keys = {"type", "duration", "creation_time", "bit_rate"}
        assert required_keys.issubset(set(result.keys()))
