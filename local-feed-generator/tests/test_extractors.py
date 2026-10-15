"""Tests for the PDF and HTML metadata extractors."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from local_feed_generator.services.extractors import HTMLExtractor, PDFExtractor, _clean_title
from pypdf import PdfWriter

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


# ---------------------------------------------------------------------------
# PDFExtractor tests
# ---------------------------------------------------------------------------


class TestPDFExtractor:
    """Tests for the PDFExtractor class."""

    def test_extract_with_full_metadata(self, tmp_path):
        """PDF with title, author, and creation date should extract all fields."""
        pdf_path = tmp_path / "document.pdf"
        # PDF dates use D:YYYYMMDDhhmmss format; pypdf expects naive datetime
        create_test_pdf(
            pdf_path,
            title="My PDF Title",
            author="John Doe",
            creation_date=datetime(2026, 10, 14, 12, 0, 0),
        )

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "My PDF Title"
        assert result["description"] == "John Doe"
        assert result["filetype"] == "pdf"
        assert result["size"] > 0
        # Date should be UNIX epoch (int)
        assert isinstance(result["date"], int)
        assert result["date"] > 0

    def test_extract_fallback_title_to_filename(self, tmp_path):
        """PDF without title should fall back to filename (stem, underscores cleaned)."""
        pdf_path = tmp_path / "my_document.pdf"
        create_test_pdf(
            pdf_path,
            title=None,
            author=None,
            creation_date=None,
        )

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "my document"
        assert result["description"] is None
        assert result["filetype"] == "pdf"

    def test_extract_fallback_date_to_mtime(self, tmp_path):
        """PDF without creation date should fall back to file modification time."""
        pdf_path = tmp_path / "nodate.pdf"
        create_test_pdf(pdf_path, title="No Date", author=None, creation_date=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        # Date should be close to file modification time (as epoch int)
        mtime = int(pdf_path.stat().st_mtime)
        assert isinstance(result["date"], int)
        assert abs(result["date"] - mtime) < 5

    def test_extract_file_size_in_bytes(self, tmp_path):
        """File size should be the actual file size in bytes."""
        pdf_path = tmp_path / "sized.pdf"
        create_test_pdf(pdf_path, title="Size Test")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        expected_size = pdf_path.stat().st_size
        assert result["size"] == expected_size
        assert isinstance(result["size"], int)

    def test_extract_date_is_iso8601(self, tmp_path):
        """Date should be stored as UNIX epoch (int)."""
        pdf_path = tmp_path / "date_test.pdf"
        create_test_pdf(
            pdf_path,
            title="Date Test",
            author=None,
            creation_date=datetime(2026, 6, 15, 8, 30, 0),
        )

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        # Should be an integer (UNIX epoch)
        assert isinstance(result["date"], int)
        assert result["date"] > 0

    def test_extract_handles_corrupted_pdf(self, tmp_path):
        """Corrupted PDF should fall back to filename and mtime without crashing."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"not a real pdf file content")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "corrupted"
        assert result["description"] is None
        assert result["filetype"] == "pdf"
        assert result["size"] > 0
        # Date should fall back to file mtime (as epoch int)
        assert isinstance(result["date"], int)
        assert result["date"] > 0

    def test_extract_empty_title_falls_back_to_filename(self, tmp_path):
        """PDF with empty title string should fall back to filename (underscores cleaned)."""
        pdf_path = tmp_path / "empty_title.pdf"
        create_test_pdf(pdf_path, title="", author="Author")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "empty title"
        assert result["description"] == "Author"

    def test_extract_accepts_string_path(self, tmp_path):
        """Extractor should accept a string path, not just Path."""
        pdf_path = tmp_path / "string_path.pdf"
        create_test_pdf(pdf_path, title="String Path Test")

        extractor = PDFExtractor()
        result = extractor.extract(str(pdf_path))

        assert result["title"] == "String Path Test"
        assert result["filetype"] == "pdf"

    def test_extract_returns_all_required_keys(self, tmp_path):
        """Result dict should contain all required keys."""
        pdf_path = tmp_path / "keys.pdf"
        create_test_pdf(pdf_path, title="Keys Test")

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        required_keys = {"title", "description", "date", "size", "filetype"}
        assert required_keys.issubset(set(result.keys()))

    def test_extract_description_is_none_when_no_author(self, tmp_path):
        """Description should be None when PDF has no author."""
        pdf_path = tmp_path / "no_author.pdf"
        create_test_pdf(pdf_path, title="No Author", author=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["description"] is None

    def test_extract_cleans_underscores_from_pdf_title(self, tmp_path):
        """Underscores in PDF metadata title should be replaced with spaces."""
        pdf_path = tmp_path / "document.pdf"
        create_test_pdf(pdf_path, title="my_pdf_title", author=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "my pdf title"

    def test_extract_cleans_underscores_from_fallback_filename(self, tmp_path):
        """Underscores in fallback filename should be replaced with spaces."""
        pdf_path = tmp_path / "my_document.pdf"
        create_test_pdf(pdf_path, title=None, author=None)

        extractor = PDFExtractor()
        result = extractor.extract(pdf_path)

        assert result["title"] == "my document"


# ---------------------------------------------------------------------------
# HTMLExtractor tests
# ---------------------------------------------------------------------------


class TestHTMLExtractor:
    """Tests for the HTMLExtractor class."""

    def test_extract_with_full_metadata(self, tmp_path):
        """HTML with title, description, and date should extract all fields."""
        html_path = tmp_path / "article.html"
        create_test_html(
            html_path,
            title="My Article Title",
            description="An interesting article about ILE",
            date="2026-10-14T12:00:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "My Article Title"
        assert result["description"] == "An interesting article about ILE"
        assert result["filetype"] == "html"
        assert result["size"] > 0
        # Date should be UNIX epoch (int)
        assert isinstance(result["date"], int)
        assert result["date"] > 0

    def test_extract_fallback_title_to_filename(self, tmp_path):
        """HTML without title tag should fall back to filename (stem, underscores cleaned)."""
        html_path = tmp_path / "no_title.html"
        create_test_html(
            html_path,
            title=None,
            description="Some description",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "no title"
        assert result["filetype"] == "html"

    def test_extract_fallback_date_to_mtime(self, tmp_path):
        """HTML without date meta should fall back to file modification time."""
        html_path = tmp_path / "no_date.html"
        create_test_html(html_path, title="No Date", date=None)

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        mtime = int(html_path.stat().st_mtime)
        assert isinstance(result["date"], int)
        assert abs(result["date"] - mtime) < 5

    def test_extract_file_size_in_bytes(self, tmp_path):
        """File size should be the actual file size in bytes."""
        html_path = tmp_path / "sized.html"
        create_test_html(html_path, title="Size Test")

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        expected_size = html_path.stat().st_size
        assert result["size"] == expected_size
        assert isinstance(result["size"], int)

    def test_extract_date_is_iso8601(self, tmp_path):
        """Date should be stored as UNIX epoch (int)."""
        html_path = tmp_path / "date_test.html"
        create_test_html(
            html_path,
            title="Date Test",
            date="2026-06-15T08:30:00Z",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        # Should be an integer (UNIX epoch)
        assert isinstance(result["date"], int)
        assert result["date"] > 0

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

        parsed = datetime.fromisoformat(result["date"])
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

        parsed = datetime.fromisoformat(result["date"])
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

        assert result["title"] == "Broken"
        assert result["filetype"] == "html"
        assert result["size"] > 0

    def test_extract_empty_title_falls_back_to_filename(self, tmp_path):
        """HTML with empty title tag should fall back to filename (underscores cleaned)."""
        html_path = tmp_path / "empty_title.html"
        html_path.write_text(
            "<html><head><title></title></head><body>Test</body></html>",
            encoding="utf-8",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "empty title"

    def test_extract_accepts_string_path(self, tmp_path):
        """Extractor should accept a string path, not just Path."""
        html_path = tmp_path / "string_path.html"
        create_test_html(html_path, title="String Path Test")

        extractor = HTMLExtractor()
        result = extractor.extract(str(html_path))

        assert result["title"] == "String Path Test"
        assert result["filetype"] == "html"

    def test_extract_returns_all_required_keys(self, tmp_path):
        """Result dict should contain all required keys."""
        html_path = tmp_path / "keys.html"
        create_test_html(html_path, title="Keys Test")

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        required_keys = {"title", "description", "date", "size", "filetype"}
        assert required_keys.issubset(set(result.keys()))

    def test_extract_minimal_html(self, tmp_path):
        """Minimal HTML with only a title should work correctly."""
        html_path = tmp_path / "minimal.html"
        html_path.write_text(
            "<!DOCTYPE html><html><head><title>Minimal</title></head><body></body></html>",
            encoding="utf-8",
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "Minimal"
        assert result["description"] is None
        assert result["filetype"] == "html"
        assert result["size"] > 0

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

        parsed = datetime.fromisoformat(result["date"])
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

        parsed = datetime.fromisoformat(result["date"])
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

        assert result["title"] == "my article title"

    def test_extract_cleans_underscores_from_fallback_filename(self, tmp_path):
        """Underscores in fallback filename should be replaced with spaces."""
        html_path = tmp_path / "my_page.html"
        create_test_html(
            html_path,
            title=None,
            description=None,
        )

        extractor = HTMLExtractor()
        result = extractor.extract(html_path)

        assert result["title"] == "my page"


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
        from local_feed_generator.metadata import Metadata

        pdf_path = tmp_path / "doc.pdf"
        create_test_pdf(pdf_path, title="Test Doc", author="Author Name")

        extractor = PDFExtractor()
        extracted = extractor.extract(pdf_path)

        metadata = Metadata(
            extract_status="complete",
            title=extracted["title"],
            description=extracted["description"] or "",
            filename=pdf_path.name,
            filepath=str(pdf_path.relative_to(tmp_path)),
            filetype=extracted["filetype"],
            date=extracted["date"],
            size=extracted["size"],
            processing_date=datetime.now(UTC),
        )

        assert metadata.title == "Test Doc"
        assert metadata.filetype == "pdf"
        assert metadata.size == extracted["size"]
        assert metadata.feed_eligible is True

    def test_html_result_compatible_with_metadata(self, tmp_path):
        """HTML extractor output should be usable to create a Metadata object."""
        from local_feed_generator.metadata import Metadata

        html_path = tmp_path / "page.html"
        create_test_html(
            html_path,
            title="Test Page",
            description="A test page description",
        )

        extractor = HTMLExtractor()
        extracted = extractor.extract(html_path)

        metadata = Metadata(
            extract_status="complete",
            title=extracted["title"],
            description=extracted["description"] or "",
            filename=html_path.name,
            filepath=str(html_path.relative_to(tmp_path)),
            filetype=extracted["filetype"],
            date=extracted["date"],
            size=extracted["size"],
            processing_date=datetime.now(UTC),
        )

        assert metadata.title == "Test Page"
        assert metadata.filetype == "html"
        assert metadata.description == "A test page description"
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
        assert restored["title"] == "Serialize Test"
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
        assert restored["title"] == "Serialize Test"
        assert restored["filetype"] == "html"
