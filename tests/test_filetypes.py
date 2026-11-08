"""Tests for canonical filetype constants and compatibility exports."""

from focus_feed.filetypes import (
    DIRECT_TRANSCRIBE_EXTENSIONS,
    DOCLING_FILETYPES,
    DOCUMENT_FILETYPES,
    MEDIA_EXTENSIONS,
    MEDIA_FILETYPES,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILETYPES,
    VIDEO_EXTRACTION_EXTENSIONS,
)
from focus_feed.producer import (
    DOCLING_FILETYPES as PRODUCER_DOCLING_FILETYPES,
)
from focus_feed.producer import (
    DOCUMENT_FILETYPES as PRODUCER_DOCUMENT_FILETYPES,
)
from focus_feed.producer import (
    MEDIA_FILETYPES as PRODUCER_MEDIA_FILETYPES,
)
from focus_feed.producer import (
    SUPPORTED_FILETYPES as PRODUCER_SUPPORTED_FILETYPES,
)
from focus_feed.scanner import (
    _DOC_FILETYPES,
)
from focus_feed.scanner import (
    DIRECT_TRANSCRIBE_EXTENSIONS as SCANNER_DIRECT_TRANSCRIBE_EXTENSIONS,
)
from focus_feed.scanner import (
    SUPPORTED_EXTENSIONS as SCANNER_SUPPORTED_EXTENSIONS,
)
from focus_feed.scanner import (
    VIDEO_EXTRACTION_EXTENSIONS as SCANNER_VIDEO_EXTRACTION_EXTENSIONS,
)
from focus_feed.services.audio_extractor import (
    DIRECT_TRANSCRIBE_FORMATS,
    NEARLY_SUPPORTED_FORMATS,
)


def test_canonical_filetype_sets_include_all_supported_types() -> None:
    assert frozenset(ext[1:] for ext in MEDIA_EXTENSIONS) == MEDIA_FILETYPES
    assert frozenset({"pdf", "html"}) | DOCLING_FILETYPES == DOCUMENT_FILETYPES
    assert SUPPORTED_FILETYPES == MEDIA_FILETYPES | DOCUMENT_FILETYPES
    assert (
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
    ) == SUPPORTED_EXTENSIONS


def test_legacy_modules_reexport_canonical_filetype_constants() -> None:
    assert SCANNER_DIRECT_TRANSCRIBE_EXTENSIONS is DIRECT_TRANSCRIBE_EXTENSIONS
    assert SCANNER_VIDEO_EXTRACTION_EXTENSIONS is VIDEO_EXTRACTION_EXTENSIONS
    assert SCANNER_SUPPORTED_EXTENSIONS is SUPPORTED_EXTENSIONS
    assert _DOC_FILETYPES is DOCUMENT_FILETYPES
    assert PRODUCER_MEDIA_FILETYPES is MEDIA_FILETYPES
    assert PRODUCER_DOCLING_FILETYPES is DOCLING_FILETYPES
    assert PRODUCER_DOCUMENT_FILETYPES is DOCUMENT_FILETYPES
    assert PRODUCER_SUPPORTED_FILETYPES is SUPPORTED_FILETYPES
    assert DIRECT_TRANSCRIBE_FORMATS is DIRECT_TRANSCRIBE_EXTENSIONS
    assert NEARLY_SUPPORTED_FORMATS is VIDEO_EXTRACTION_EXTENSIONS
