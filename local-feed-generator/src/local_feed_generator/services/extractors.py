"""Metadata extractors for PDF and HTML files.

This module provides extractor classes that read metadata from PDF and HTML
source files, returning structured dictionaries compatible with the Metadata
model (design doc section 3.3).

Each extractor's ``extract`` method returns a dict with the following keys:
    - title: str (always present; falls back to filename without extension)
    - description: str | None (optional; may be None if not available)
    - date: int (UNIX epoch seconds; falls back to file modification time)
    - size: int (file size in bytes)
    - filetype: str (always "pdf" or "html")
"""

from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

_logger = logging.getLogger("local_feed_generator")


def _clean_title(title: str | None) -> str | None:
    """Clean up a title by replacing special characters with spaces.

    Replaces underscores with spaces so that titles extracted from
    filenames or metadata (which may use underscores as separators)
    are human-readable.

    Args:
        title: The title string to clean, or None.

    Returns:
        The cleaned title with underscores replaced by spaces, or
        None if the input was None.
    """
    if title is None:
        return None
    # Replace underscores with spaces
    return title.replace("_", " ")


class PDFExtractor:
    """Extracts metadata from PDF files using pypdf.

    Extracts title, author, creation date, and file size from PDF metadata.
    Falls back to the filename (without extension) if no title is found in
    the PDF metadata, and to the file modification time if no creation date
    is available.
    """

    def extract(self, filepath: str | Path) -> dict[str, Any]:
        """Extract metadata from a PDF file.

        Args:
            filepath: Path to the PDF file.

        Returns:
            A dictionary with keys: title, description, date, size, filetype.

            - title: PDF metadata title, or filename without extension as fallback.
            - description: PDF metadata author, or None if not available.
            - date: Creation date as UNIX epoch (seconds), or file modification time.
            - size: File size in bytes.
            - filetype: Always "pdf".
        """
        filepath = Path(filepath)
        filename = filepath.stem
        stat_result = filepath.stat()
        size = stat_result.st_size
        mtime = int(stat_result.st_mtime)

        title = filename
        description: str | None = None
        date: datetime | None = None

        try:
            reader = PdfReader(str(filepath))
            metadata = reader.metadata

            if metadata is not None:
                if metadata.title:
                    title = str(metadata.title)
                if metadata.author:
                    description = str(metadata.author)
                if metadata.creation_date:
                    date = metadata.creation_date
        except Exception as e:
            _logger.warning("Failed to extract PDF metadata from %s: %s", filepath, e)

        # Fallback date to file modification time
        if date is None:
            date = datetime.fromtimestamp(mtime, tz=UTC)

        # Ensure date is timezone-aware for consistent epoch conversion
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        # Clean up title by replacing underscores with spaces
        title = _clean_title(title)

        return {
            "title": title,
            "description": description,
            "date": int(date.timestamp()),
            "size": size,
            "filetype": "pdf",
        }


class HTMLExtractor:
    """Extracts metadata from HTML files using BeautifulSoup.

    Extracts title from the ``<title>`` tag, description from
    ``<meta name="description">``, and date from ``<meta name="date">``,
    ``<meta name="pubdate">``, or ``<time datetime="...">`` elements.
    Falls back to the filename (without extension) if no title is found,
    and to the file modification time if no date is available.
    """

    def extract(self, filepath: str | Path) -> dict[str, Any]:
        """Extract metadata from an HTML file.

        Args:
            filepath: Path to the HTML file.

        Returns:
            A dictionary with keys: title, description, date, size, filetype.

            - title: Content of ``<title>`` tag, or filename without extension as fallback.
            - description: Content of ``<meta name="description">``, or None if not available.
            - date: Date as UNIX epoch (seconds), or file modification time.
            - size: File size in bytes.
            - filetype: Always "html".
        """
        filepath = Path(filepath)
        filename = filepath.stem
        stat_result = filepath.stat()
        size = stat_result.st_size
        mtime = int(stat_result.st_mtime)

        title = filename
        description: str | None = None
        date: datetime | None = None

        try:
            content = filepath.read_text(encoding="utf-8")
            soup = BeautifulSoup(content, "html.parser")

            # Extract title from <title> tag
            title_tag = soup.title
            if title_tag:
                title_text = title_tag.string or title_tag.get_text()
                if title_text and title_text.strip():
                    title = title_text.strip()

            # Extract description from <meta name="description">
            desc_meta = soup.find("meta", attrs={"name": "description"})
            desc_content = desc_meta.get("content") if desc_meta else None
            if desc_content:
                description = str(desc_content).strip()

            # Extract date from various sources
            date = self._extract_date(soup)
        except Exception as e:
            _logger.warning("Failed to extract HTML metadata from %s: %s", filepath, e)

        # Fallback date to file modification time
        if date is None:
            date = datetime.fromtimestamp(mtime, tz=UTC)

        # Ensure date is timezone-aware for consistent epoch conversion
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        # Clean up title by replacing underscores with spaces
        title = _clean_title(title)

        return {
            "title": title,
            "description": description,
            "date": int(date.timestamp()),
            "size": size,
            "filetype": "html",
        }

    def _extract_date(self, soup: BeautifulSoup) -> datetime | None:
        """Extract a date from HTML meta tags or time elements.

        Tries the following sources in order:
        1. ``<meta name="date" content="...">``
        2. ``<meta name="pubdate" content="...">``
        3. ``<time datetime="...">``

        Args:
            soup: A BeautifulSoup parsed HTML document.

        Returns:
            A timezone-aware datetime if a date was found and parsed, otherwise None.
        """
        # Try <meta name="date">
        date_meta = soup.find("meta", attrs={"name": "date"})
        date_content = date_meta.get("content") if date_meta else None
        if date_content:
            parsed = self._parse_date(str(date_content).strip())
            if parsed:
                return parsed

        # Try <meta name="pubdate">
        pubdate_meta = soup.find("meta", attrs={"name": "pubdate"})
        pubdate_content = pubdate_meta.get("content") if pubdate_meta else None
        if pubdate_content:
            parsed = self._parse_date(str(pubdate_content).strip())
            if parsed:
                return parsed

        # Try <time datetime="...">
        time_tag = soup.find("time")
        time_datetime = time_tag.get("datetime") if time_tag else None
        if time_datetime:
            parsed = self._parse_date(str(time_datetime).strip())
            if parsed:
                return parsed

        return None

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Try to parse a date string into a timezone-aware datetime.

        Tries ISO 8601 format first, then several common date formats.

        Args:
            date_str: A date string in an unknown format.

        Returns:
            A timezone-aware datetime if parsing succeeds, otherwise None.
        """
        # Try ISO 8601 format (handles "Z" suffix in Python 3.11+)
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError, TypeError:
            pass

        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
            except ValueError, TypeError:
                continue

        return None
