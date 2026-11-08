"""Producer pipeline for Focus Feed.

Orchestrates file processing based on file type. For audio/video files, it
transcribes audio to markdown, then generates key-takeaways summary via
LLM. For PDF/HTML files, it extracts metadata from the source file,
summarizes the extracted content via LLM, and uses the filename
(without extension, underscores→spaces) as the ``title.filename``. For
all file types, it generates a descriptive title via LLM using the
filename, extracted title (if available), and content as context,
stored as ``title.generated``.  This title generation step is tracked via
the ``generate_title`` step map (status, timing, model, and token usage).

The Producer writes a JSON metadata file adjacent to each source file,
following the schema defined in design doc section 3.3. On errors, it
writes an error-state metadata file and logs the error to stderr.

Processing progress is tracked through separate step maps:
- ``transcribe``: StepMap for transcription (media files only).
- ``summarize``: StepMap for summarization (all file types).
- ``extract``: StepMap for metadata extraction (PDF/HTML only).
- ``generate_title``: StepMap for LLM title generation (all file types,
  when LLM title generation runs).

Date fields (``date``) are stored as UNIX epoch floats with millisecond
precision.  Per-step timing is tracked within each step map's
``start_time`` and ``end_time`` fields, also as epoch floats with
millisecond precision.

The title field is a dict with exactly three keys:
- ``filename``: sanitized filename (no extension, underscores→spaces)
- ``extracted``: title from source metadata (PDF metadata, HTML <title>,
  or media metadata where available)
- ``generated``: LLM-generated title (all file types when content is
  available)

Title precedence for the final title: generated > extracted > filename.
The module name prefix is applied to the final title after precedence
resolution.

When a file is processed for the first time (no existing JSON metadata or
incomplete processing), the SHA256 hash of the source file is computed and
stored in the ``hash`` field of the JSON metadata.  This hash is used by the
feed generator for guid generation (last 12 characters).

Design references:
- design.md section 2.2 (Producer Pipeline responsibility)
- design.md section 5.2 (Producer module responsibilities)
- design.md section 6.1 (Batch processing flow)
- design.md section 7.1 (Error handling flow)
"""

from focus_feed.config import get_config
from focus_feed.filetypes import (
    DOCLING_FILETYPES,
    DOCUMENT_FILETYPES,
    MEDIA_FILETYPES,
    SUPPORTED_FILETYPES,
    DoclingFiletype,
    MediaFiletype,
)
from focus_feed.services.extractors import get_audio_metadata
from focus_feed.utils import clean_title

from ._base import (
    ProducerBase,
    _check_reflection_md,
    _compute_file_hash,
    _now_timestamp,
    _sanitize_filename,
    _tag_step,
)
from ._document import DocumentProcessingMixin
from ._media import MediaProcessingMixin

_clean_title = clean_title

__all__ = [
    "DOCLING_FILETYPES",
    "DOCUMENT_FILETYPES",
    "MEDIA_FILETYPES",
    "SUPPORTED_FILETYPES",
    "DoclingFiletype",
    "MediaFiletype",
    "Producer",
    "_check_reflection_md",
    "_clean_title",
    "_compute_file_hash",
    "_now_timestamp",
    "_sanitize_filename",
    "_tag_step",
    "get_audio_metadata",
    "get_config",
]


class Producer(MediaProcessingMixin, DocumentProcessingMixin, ProducerBase):
    """Producer pipeline for Focus Feed (see package docstring)."""
