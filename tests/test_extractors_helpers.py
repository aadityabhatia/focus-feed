"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfWriter
import pytest

from focus_feed.services.extractors import (
    HTMLExtractor,
    PDFExtractor,
    _clean_title,
    _sanitize_content,
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
            "extracted",
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
        assert result["extracted"] == "String Path Test"
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
        assert result["extracted"] == "My Title"
        assert result["description"] == expected_description
        assert result["filetype"] == filetype
        assert result["size"] > 0
        assert isinstance(result["date"], float)
        assert result["date"] > 0
