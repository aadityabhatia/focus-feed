"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfWriter

from focus_feed.services.extractors import (
    HTMLExtractor,
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
        assert result["extracted"] == "Minimal"
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
        assert result["extracted"] == "my article title"

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
