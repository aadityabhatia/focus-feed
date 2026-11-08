"""Shared, dependency-free utility helpers for Focus Feed."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re

from focus_feed.filetypes import DIRECT_TRANSCRIBE_EXTENSIONS, VIDEO_EXTRACTION_EXTENSIONS

# Pattern matching control characters that are invalid in XML (everything
# below 0x20 except tab \x09, newline \x0a, and carriage return \x0d).
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Characters that are problematic in filenames for XML/URL contexts.
SPECIAL_CHAR_PATTERN = re.compile(r'[<>&"\'\x00-\x1f]')

# Filenames that should be ignored entirely (not treated as metadata or source).
_IGNORE_FILENAMES = frozenset(
    {"podcast.xml", "articles.xml", "podcast.atom", "articles.atom", "index.html", "feed.json"}
)


def is_audio_filetype(filetype: str) -> bool:
    """Return whether *filetype* is a directly transcribable audio format."""
    normalized_filetype = filetype.lower().lstrip(".")
    return (
        f".{normalized_filetype}" in DIRECT_TRANSCRIBE_EXTENSIONS and normalized_filetype != "mp4"
    )


def is_video_filetype(filetype: str) -> bool:
    """Return whether *filetype* is a video format, case-insensitively."""
    normalized_filetype = filetype.lower().lstrip(".")
    return f".{normalized_filetype}" in (".mp4", *VIDEO_EXTRACTION_EXTENSIONS)


def should_ignore_filename(filename: str) -> bool:
    """Return whether *filename* is a generated, hidden, or temporary file."""
    return (
        filename.lower() in _IGNORE_FILENAMES
        or filename.startswith(".")
        or filename.lower().endswith(".tmp")
    )


def now_timestamp() -> float:
    """Return the current UNIX epoch time with millisecond precision."""
    return int(datetime.now(UTC).timestamp() * 1000) / 1000.0


def clean_title(title: str | None) -> str | None:
    """Replace underscores with spaces in *title*; return None for None."""
    if title is None:
        return None
    return title.replace("_", " ")


def sanitize_content(text: str | None) -> str | None:
    """Remove control characters that are invalid in XML from *text*."""
    if text is None:
        return None
    return _CONTROL_CHAR_PATTERN.sub("", text)


def metadata_sidecar_path(source_path: str | Path) -> Path:
    """Return the adjacent ``<source_filename>.json`` metadata path."""
    path = Path(source_path)
    return path.with_name(path.name + ".json")
