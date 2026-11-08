"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
from pypdf import PdfWriter
import pytest

from focus_feed.services.extractors import (
    DoclingExtractor,
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


class TestDoclingExtractor:
    """Tests for document conversion through the Docling API."""

    @pytest.mark.parametrize("extension", ["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls"])
    def test_extract_posts_document_and_returns_markdown(self, tmp_path, extension):
        path = tmp_path / f"my_document.{extension}"
        path.write_bytes(b"document data")
        response = mock.Mock()
        response.json.return_value = {"document": {"md_content": "# Extracted Markdown"}}

        with (
            mock.patch("focus_feed.services.extractors.get_config") as get_config,
            mock.patch("focus_feed.services.extractors.httpx.post", return_value=response) as post,
        ):
            get_config.return_value.docling_api = "http://docling:5001/"
            result = DoclingExtractor().extract(path)

        assert result == {
            "title": "my document",
            "extracted": None,
            "description": None,
            "date": int(path.stat().st_mtime * 1000) / 1000.0,
            "publication_date": None,
            "size": path.stat().st_size,
            "filetype": extension,
            "content": "# Extracted Markdown",
        }
        assert post.call_args.args[0] == "http://docling:5001/v1/convert/file"
        assert post.call_args.kwargs["data"] == {"to_formats": "md"}
        assert post.call_args.kwargs["files"]["files"][0] == path.name

    def test_extract_uses_injected_config(self, tmp_path):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"document data")
        response = mock.Mock()
        response.json.return_value = {"document": {"md_content": "# Injected config"}}
        config = mock.MagicMock()
        config.docling_api = "http://injected-docling:5001"

        with (
            mock.patch("focus_feed.services.extractors.get_config") as get_config,
            mock.patch("focus_feed.services.extractors.httpx.post", return_value=response) as post,
        ):
            result = DoclingExtractor(config=config).extract(path)

        get_config.assert_not_called()
        assert post.call_args.args[0] == "http://injected-docling:5001/v1/convert/file"
        assert result["content"] == "# Injected config"

    def test_extract_requires_docling_api_configuration(self, tmp_path):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"document data")

        with mock.patch("focus_feed.services.extractors.get_config") as get_config:
            get_config.return_value.docling_api = None
            with pytest.raises(RuntimeError, match="DOCLING_API is not configured"):
                DoclingExtractor().extract(path)

    def test_extract_raises_connection_error_when_api_unreachable(self, tmp_path):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"document data")
        request = httpx.ConnectError("connection refused")

        with (
            mock.patch("focus_feed.services.extractors.get_config") as get_config,
            mock.patch("focus_feed.services.extractors.httpx.post", side_effect=request),
        ):
            get_config.return_value.docling_api = "http://docling:5001"
            with pytest.raises(ConnectionError, match="connection refused"):
                DoclingExtractor().extract(path)

    def test_extract_returns_none_content_on_bad_response(self, tmp_path):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"document data")
        response = mock.Mock()
        response.json.return_value = {"status": "failure"}

        with (
            mock.patch("focus_feed.services.extractors.get_config") as get_config,
            mock.patch("focus_feed.services.extractors.httpx.post", return_value=response),
        ):
            get_config.return_value.docling_api = "http://docling:5001"
            result = DoclingExtractor().extract(path)

        assert result["content"] is None
