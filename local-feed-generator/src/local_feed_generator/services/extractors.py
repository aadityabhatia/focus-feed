"""Metadata extractors for PDF, HTML, and audio (MP4) files.

This module provides extractor classes that read metadata from PDF and HTML
source files, returning structured dictionaries compatible with the Metadata
model (design doc section 3.3).

It also provides a :func:`get_audio_metadata` function that uses ``ffprobe``
to extract metadata from MP4/audio files (type, duration, creation_time,
bit_rate).

    Each extractor's ``extract`` method returns a dict with the following keys:
    - title: str — filename without extension, underscores replaced with spaces
        (used as the actual/final title)
    - extracted_title: str | None — title extracted from source metadata
        (PDF metadata title, HTML <title> tag), or None if not available.
        This is stored separately and NOT used as the actual title.
    - description: str | None (optional; may be None if not available)
    - date: float (UNIX epoch with millisecond precision; falls back to file
        modification time)
    - publication_date: float | None (UNIX epoch with millisecond precision;
        document metadata date if available, None otherwise)
    - size: int (file size in bytes)
    - filetype: str (always "pdf" or "html")
    - content: str | None (plaintext content extracted from the source file,
      or None if extraction fails or no text is found)
"""

import contextlib
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

_logger = logging.getLogger("local_feed_generator")

# Pattern matching control characters that are invalid in XML (everything
# below 0x20 except tab \x09, newline \x0a, and carriage return \x0d).
# These characters can leak into extracted text from PDFs and other binary
# formats, causing feed generation to fail.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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


def _sanitize_content(text: str | None) -> str | None:
    """Remove control characters that are invalid in XML from extracted text.

    PDFs and other binary formats can produce text containing control
    characters (e.g. ``\\x00``-``\\x1f`` except tab/newline/CR, and
    ``\\x7f``) that are not valid in XML and cause feed generation to fail.
    This function strips those characters while preserving common
    whitespace (``\\t``, ``\\n``, ``\\r``).

    Args:
        text: The extracted text, or None.

    Returns:
        The sanitized text with control characters removed, or None if
        the input was None.
    """
    if text is None:
        return None
    return _CONTROL_CHAR_PATTERN.sub("", text)


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
            A dictionary with keys: title, extracted_title, description,
            date, publication_date, size, filetype, content.

            - title: Filename without extension (underscores replaced with
              spaces). This is the actual/final title used.
            - extracted_title: PDF metadata title, or None if not available.
              Stored separately; not used as the actual title.
            - description: PDF metadata author, or None if not available.
            - date: Creation date as UNIX epoch (float, ms precision), or
              file modification time.
            - publication_date: Creation date from PDF metadata as UNIX epoch
              (float, ms precision), or None if not available.
            - size: File size in bytes.
            - filetype: Always "pdf".
            - content: Plaintext content extracted from all PDF pages, or None
              if no text could be extracted.
        """
        filepath = Path(filepath)
        filename = filepath.stem
        stat_result = filepath.stat()
        size = stat_result.st_size
        mtime = int(stat_result.st_mtime)

        # Title is always the filename (without extension, underscores→spaces).
        # The extracted title from PDF metadata is stored separately.
        title: str | None = _clean_title(filename)
        extracted_title: str | None = None
        description: str | None = None
        date: datetime | None = None
        publication_date: datetime | None = None
        content: str | None = None

        try:
            reader = PdfReader(str(filepath))
            metadata = reader.metadata

            if metadata is not None:
                if metadata.title:
                    extracted_title = _clean_title(str(metadata.title))
                if metadata.author:
                    description = str(metadata.author)
                if metadata.creation_date:
                    date = metadata.creation_date
                    publication_date = metadata.creation_date

            # Extract plaintext content from all pages.
            content_parts: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            content = "\n".join(content_parts) if content_parts else None
            # Remove control characters that are invalid in XML and would
            # cause feed generation to fail.
            content = _sanitize_content(content)
        except Exception as e:
            _logger.warning("Failed to extract PDF metadata from %s: %s", filepath, e)

        # Fallback date to file modification time
        if date is None:
            date = datetime.fromtimestamp(mtime, tz=UTC)

        # Ensure date is timezone-aware for consistent epoch conversion
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        # publication_date is only set if document metadata provided a date.
        publication_date_epoch: float | None = None
        if publication_date is not None:
            if publication_date.tzinfo is None:
                publication_date = publication_date.replace(tzinfo=UTC)
            publication_date_epoch = int(publication_date.timestamp() * 1000) / 1000.0

        return {
            "title": title,
            "extracted_title": extracted_title,
            "description": description,
            "date": int(date.timestamp() * 1000) / 1000.0,
            "publication_date": publication_date_epoch,
            "size": size,
            "filetype": "pdf",
            "content": content,
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
            A dictionary with keys: title, extracted_title, description,
            date, publication_date, size, filetype, content.

            - title: Filename without extension (underscores replaced with
              spaces). This is the actual/final title used.
            - extracted_title: Content of ``<title>`` tag, or None if not
              available. Stored separately; not used as the actual title.
            - description: Content of ``<meta name="description">``, or None
              if not available.
            - date: Date as UNIX epoch (float, ms precision), or file
              modification time.
            - publication_date: Date from HTML metadata (``<meta name="date">``,
              ``<meta name="pubdate">``, or ``<time datetime="...">``) as UNIX
              epoch (float, ms precision), or None if not available.
            - size: File size in bytes.
            - filetype: Always "html".
            - content: Plaintext content extracted from the HTML document body
              (tags removed), or None if parsing fails.
        """
        filepath = Path(filepath)
        filename = filepath.stem
        stat_result = filepath.stat()
        size = stat_result.st_size
        mtime = int(stat_result.st_mtime)

        # Title is always the filename (without extension, underscores→spaces).
        # The extracted title from <title> tag is stored separately.
        title: str | None = _clean_title(filename)
        extracted_title: str | None = None
        description: str | None = None
        date: datetime | None = None
        publication_date: datetime | None = None
        text_content: str | None = None

        try:
            html_content = filepath.read_text(encoding="utf-8")
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title from <title> tag (stored as extracted_title, not used as actual title)
            title_tag = soup.title
            if title_tag:
                title_text = title_tag.string or title_tag.get_text()
                if title_text and title_text.strip():
                    extracted_title = _clean_title(title_text.strip())

            # Extract description from <meta name="description">
            desc_meta = soup.find("meta", attrs={"name": "description"})
            desc_content = desc_meta.get("content") if desc_meta else None
            if desc_content:
                description = str(desc_content).strip()

            # Extract plaintext content from the HTML body (tags removed)
            text_content = soup.get_text()
            # Remove control characters that are invalid in XML.
            text_content = _sanitize_content(text_content)

            # Extract date from various sources
            date = self._extract_date(soup)
            publication_date = date
        except Exception as e:
            _logger.warning("Failed to extract HTML metadata from %s: %s", filepath, e)

        # Fallback date to file modification time
        if date is None:
            date = datetime.fromtimestamp(mtime, tz=UTC)

        # Ensure date is timezone-aware for consistent epoch conversion
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        # publication_date is only set if document metadata provided a date.
        publication_date_epoch: float | None = None
        if publication_date is not None:
            if publication_date.tzinfo is None:
                publication_date = publication_date.replace(tzinfo=UTC)
            publication_date_epoch = int(publication_date.timestamp() * 1000) / 1000.0

        return {
            "title": title,
            "extracted_title": extracted_title,
            "description": description,
            "date": int(date.timestamp() * 1000) / 1000.0,
            "publication_date": publication_date_epoch,
            "size": size,
            "filetype": "html",
            "content": text_content,
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


# ---------------------------------------------------------------------------
# Audio metadata extraction via ffprobe
# ---------------------------------------------------------------------------


def get_audio_metadata(filepath: str | Path) -> dict[str, Any] | None:
    """Get audio file metadata using ``ffprobe``.

    Runs the ``ffprobe`` command-line tool to extract container and stream
    metadata from an audio/video file (typically MP4).  The function returns
    a structured dictionary suitable for inclusion in the JSON metadata file.

    Returns a dict with the following keys:
        - type: str — the format name reported by ffprobe (e.g.
          ``"mov,mp4,m4a,3gp,3g2,mj2"``).
        - duration: float — duration in seconds, or ``0`` if unavailable.
        - creation_time: str | None — ISO-8601 creation time string from
          the file's metadata tags, or ``None`` if not present.
        - bit_rate: int | None — bit rate in bits per second, or ``None``
          if not available.
        - title: str | None — title from the file's metadata tags, or
          ``None`` if not present. Underscores are replaced with spaces.

    If ``ffprobe`` is not installed or any error occurs during probing,
    ``None`` is returned so that callers can fall back to existing behaviour
    (e.g. file modification time).

    Args:
        filepath: Path to the audio/video file to probe.

    Returns:
        A metadata dictionary, or ``None`` if probing fails.
    """
    filepath = Path(filepath)

    # Bail out early if ffprobe is not on the system PATH.
    if shutil.which("ffprobe") is None:
        _logger.warning("ffprobe not found on PATH; cannot extract audio metadata")
        return None

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(filepath),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        probe_data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as e:
        _logger.warning("ffprobe failed for %s: %s", filepath, e)
        return None

    format_info = probe_data.get("format", {})
    streams = probe_data.get("streams", [])

    # Prefer the first audio stream for metadata, but fall back to the
    # container-level format info if no audio stream is found.
    duration: float | None = None
    bit_rate: int | None = None

    # Try to get duration from an audio stream first, then from format.
    for stream in streams:
        if stream.get("codec_type") == "audio":
            stream_duration = stream.get("duration")
            if stream_duration is not None:
                duration = float(stream_duration)
            stream_bit_rate = stream.get("bit_rate")
            if stream_bit_rate is not None:
                with contextlib.suppress(ValueError, TypeError):
                    bit_rate = int(stream_bit_rate)
            break

    # Fall back to format-level values.
    if duration is None:
        fmt_duration = format_info.get("duration")
        if fmt_duration is not None:
            with contextlib.suppress(ValueError, TypeError):
                duration = float(fmt_duration)

    if bit_rate is None:
        fmt_bit_rate = format_info.get("bit_rate")
        if fmt_bit_rate is not None:
            with contextlib.suppress(ValueError, TypeError):
                bit_rate = int(fmt_bit_rate)

    tags = format_info.get("tags", {})

    # Extract title from metadata tags if available (MP4 files may have a
    # 'title' tag in their container metadata).
    title: str | None = tags.get("title")
    if title:
        title = _clean_title(title)

    return {
        "type": format_info.get("format_name"),
        "duration": duration if duration is not None else 0.0,
        "creation_time": tags.get("creation_time"),
        "bit_rate": bit_rate,
        "title": title,
    }
