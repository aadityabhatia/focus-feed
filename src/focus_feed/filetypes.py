"""Canonical file type and extension constants for Focus Feed.

Dependency-free leaf module: importing this must not pull in any other
``focus_feed`` module or heavy third-party packages.
"""

from __future__ import annotations

from typing import Literal

# Audio formats supported directly by the transcription API.
DIRECT_TRANSCRIBE_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".m4a",
    ".wav",
    ".webm",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mpeg",
    ".mpga",
)

# Video formats that require audio extraction before transcription.
VIDEO_EXTRACTION_EXTENSIONS: tuple[str, ...] = (".mkv", ".mov", ".avi", ".wmv")

# All media extensions handled through the transcription pipeline.
MEDIA_EXTENSIONS: tuple[str, ...] = (
    *DIRECT_TRANSCRIBE_EXTENSIONS,
    *VIDEO_EXTRACTION_EXTENSIONS,
)

# Document formats handled by Docling (PDF also has a local extractor fallback).
DOCLING_EXTENSIONS: tuple[str, ...] = (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls")

# Document extensions including PDF and HTML.
DOCUMENT_EXTENSIONS: tuple[str, ...] = (".pdf", ".html", *DOCLING_EXTENSIONS)

# Every source extension the scanner discovers (case-insensitive). The exact
# ordering below matches the previous scanner definition (and test_scanner.py).
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    *DIRECT_TRANSCRIBE_EXTENSIONS,
    *VIDEO_EXTRACTION_EXTENSIONS,
    ".html",
    ".doc",
    ".docx",
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
)

# Bare filetypes (no leading dot) for media files.
MEDIA_FILETYPES: frozenset[str] = frozenset(ext.lstrip(".") for ext in MEDIA_EXTENSIONS)

# Bare filetypes for Docling-supported documents.
DOCLING_FILETYPES: frozenset[str] = frozenset(ext.lstrip(".") for ext in DOCLING_EXTENSIONS)

# Bare filetypes for all documents (PDF, HTML, Docling).
DOCUMENT_FILETYPES: frozenset[str] = frozenset({"pdf", "html"}) | DOCLING_FILETYPES

# All supported bare filetypes.
SUPPORTED_FILETYPES: frozenset[str] = MEDIA_FILETYPES | DOCUMENT_FILETYPES

# Literal types for static typing.
MediaFiletype = Literal[
    "mp3",
    "m4a",
    "wav",
    "webm",
    "aac",
    "flac",
    "ogg",
    "opus",
    "mp4",
    "mpeg",
    "mpga",
    "mkv",
    "mov",
    "avi",
    "wmv",
]
DoclingFiletype = Literal["docx", "doc", "pptx", "ppt", "xlsx", "xls"]
DocumentFiletype = Literal["pdf", "html", "docx", "doc", "pptx", "ppt", "xlsx", "xls"]
