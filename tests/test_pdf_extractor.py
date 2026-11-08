"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfWriter

from focus_feed.services.extractors import (
    PDFExtractor,
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
        assert result["extracted"] is None
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
        """Underscores in PDF metadata title should be replaced with spaces in extracted."""
        pdf_path = tmp_path / "document.pdf"
        create_test_pdf(pdf_path, title="my_pdf_title", author=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        # Title is always filename-derived
        assert result["title"] == "document"
        # Extracted title should have underscores cleaned
        assert result["extracted"] == "my pdf title"

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
