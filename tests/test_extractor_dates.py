"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from pypdf import PdfWriter

from focus_feed.services.extractors import (
    HTMLExtractor,
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
        assert restored["extracted"] == "Serialize Test"
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
        assert restored["extracted"] == "Serialize Test"
        assert restored["filetype"] == "html"
