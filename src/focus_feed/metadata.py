"""Metadata model for Focus Feed JSON metadata files.

This module defines the Metadata Pydantic model that matches the JSON schema
specified in design doc section 3.3. Each processed source file has a
corresponding `.json` metadata file with the same name, containing metadata
used for feed generation.

Processing progress is tracked through separate step maps for each processing
step:

- ``transcribe``: Step map for transcription progress (MP4 files only).
- ``summarize``: Step map for summarization progress (all file types).
- ``extract``: Step map for metadata extraction progress (PDF/HTML only).
- ``generate_title``: Step map for LLM title generation progress (all file
  types, when LLM title generation runs).

    Each step map contains a ``status`` field that can be ``None`` (not started),
    ``"in_progress"``, or ``"complete"``.  The :attr:`feed_eligible` computed
    property returns ``True`` only when all applicable steps are ``"complete"``
    and there is no error.  For PDF/HTML files, both ``extract`` and
    ``summarize`` must be complete; for MP4 files, both ``transcribe`` and
    ``summarize`` must be complete.

Date fields (``date``) are stored as UNIX epoch floats with millisecond
precision.  Per-step timing is tracked within each step map's
``start_time`` and ``end_time`` fields, also as epoch floats with
millisecond precision.

    Schema:
        {
            "title": {
                "filename": "lecture",      # sanitized filename, no extension/underscores
                "extracted": "Lecture 1",    # from source metadata (PDF/HTML/MP4)
                "generated": "LLM Title"     # LLM-generated with filename and extracted title
            },
            "description": "string (optional, extracted from source for PDF/HTML, empty for MP4)",
            "summary": "string (optional, LLM-generated key takeaways for all file types)",
            "filename": "lecture.mp4",
            "filetype": "mp4|pdf|html",
            "date": 1728931200.123,          # UNIX epoch (float, millisecond precision)
            "mtime": 1728931200.123,         # Source file modification time
            "publication_date": 1728931200.123,  # Document metadata date (PDF/HTML, optional)
            "size": 12345678,
            "duration": 3600.5,          # Audio duration in seconds (MP4, optional)
            "creation_time": "2024-01-15T10:30:00.000000Z",  # Media creation time
            "content": "Plaintext content (transcript for MP4, extracted text for PDF/HTML)",
            "hash": "a1b2c3d4e5f6...",  # OPTIONAL: SHA256 hash of source file
            "transcribe": {
                "status": "complete",
                "start_time": 1792096505.123,
                "end_time": 1792096515.456,
                "input_size": 12345678,
                "output_size": 456789,
                "words": 452,
                "model": "large-v3-turbo",
                "input_tokens": 1234,
                "output_tokens": 567
            },
            "summarize": {
                "status": "complete",
                "start_time": 1792096515.123,
                "end_time": 1792096520.456,
                "input_size": 456789,
                "output_size": 1997,
                "words": 345,
                "model": "mistralai/Mistral-7B-Instruct",
                "input_tokens": 890,
                "output_tokens": 123
            },
            "extract": {
                "status": "complete",
                "start_time": 1792096500.123,
                "end_time": 1792096505.456,
                "input_size": 1790891,
                "output_size": 12345,
                "words": 543
            },
            "generate_title": {
                "status": "complete",
                "start_time": 1792096520.123,
                "end_time": 1792096525.456,
                "input_size": 12345,
                "output_size": 24,
                "words": 4,
                "model": "mistralai/Mistral-7B-Instruct",
                "input_tokens": 456,
                "output_tokens": 8
            },
            "error": "string (optional, only when processing failed)"
        }

    Note: Version tracking is handled via git commits, not explicit version
    numbers in metadata files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from focus_feed.errors import log_error

# Step status values.
_STEP_STATUS_COMPLETE = "complete"

_logger = logging.getLogger("focus_feed")

# Required keys for the title dict (design doc section 3.3).
_TITLE_DICT_KEYS: frozenset[str] = frozenset({"filename", "extracted", "generated"})


def _apply_module_prefix_to_title(title: str, filepath: str) -> str:
    """Prepend the module name to *title* if appropriate.

    The module name is extracted from the parent directory of *filepath*.
    The module name is prepended only when the title does not already start
    with the module name (case-insensitive, handling underscores and spaces).

    Args:
        title: The base title string (already resolved via precedence).
        filepath: Relative path from the data directory root.

    Returns:
        The title with module prefix if needed, otherwise the original title.
    """
    path = Path(filepath)
    parent = path.parent
    module_name = parent.name if parent.name else ""
    if not module_name:
        return title

    # Normalize: replace underscores with spaces, lowercase for comparison.
    module_normalized = module_name.replace("_", " ").lower()
    title_normalized = title.replace("_", " ").lower()

    # Check if title already starts with module name (case-insensitive).
    if title_normalized.startswith(module_normalized):
        return title

    return f"{module_name.replace('_', ' ')}: {title}"


class StepMap(BaseModel):
    """Metadata for a single processing step.

    Each processing step (transcribe, summarize, extract, generate_title) gets
    its own :class:`StepMap` instance that tracks the step's status, timing,
    sizes, and (for LLM-based steps) model and token usage.

    Attributes:
        status: ``None``, ``"in_progress"``, or ``"complete"``.
        start_time: UNIX epoch timestamp when the step started, or ``None``
            (millisecond precision).
        end_time: UNIX epoch timestamp when the step ended, or ``None``
            (millisecond precision).
        input_size: Input size in bytes.
        output_size: Output size in bytes.
        words: Output word count (for text output steps), or ``None``.
        model: Model identifier used (LLM-based steps only: transcribe,
            summarize, generate_title), or ``None``.
        input_tokens: Number of tokens input to LLM (LLM-based steps only),
            or ``None``.
        output_tokens: Number of tokens output from LLM (LLM-based steps
            only), or ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    status: str | None = Field(
        default=None,
        description="null, 'in_progress', or 'complete'",
    )
    start_time: float | None = Field(
        default=None,
        description="UNIX epoch start time (millisecond precision)",
    )
    end_time: float | None = Field(
        default=None,
        description="UNIX epoch end time (millisecond precision)",
    )
    input_size: int = Field(
        ge=0,
        description="Input size in bytes",
    )
    output_size: int = Field(
        ge=0,
        description="Output size in bytes",
    )
    words: int | None = Field(
        default=None,
        ge=0,
        description="Output word count",
    )
    model: str | None = Field(
        default=None,
        description="Model identifier (LLM steps only)",
    )
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Input tokens (LLM steps only)",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Output tokens (LLM steps only)",
    )


class Metadata(BaseModel):
    """Pydantic model for JSON metadata matching design doc section 3.3.

    Each processed source file has a corresponding `.json` file with the same
    name, containing metadata used for feed generation.

    Step progress is tracked through separate step maps:

    - ``transcribe``: :class:`StepMap` for transcription (MP4 only).
    - ``summarize``: :class:`StepMap` for summarization (all file types).
    - ``extract``: :class:`StepMap` for metadata extraction (PDF/HTML only).
    - ``generate_title``: :class:`StepMap` for LLM title generation (all file
      types, when LLM title generation runs).

    The :attr:`feed_eligible` computed property returns ``True`` only when all
    applicable steps are ``"complete"``.  Files with ``feed_eligible == False``
    are excluded from generated feeds.

    For PDF/HTML files, both ``extract`` and ``summarize`` must be complete.
    For MP4 files, both ``transcribe`` and ``summarize`` must be complete.

    Field Descriptions:
        title: Human-readable title as a dict with keys 'filename', 'extracted',
            and 'generated', or a plain string (backward compatibility, emits
            a deprecation warning). The 'filename' key holds the
            filename-derived title (underscores replaced with spaces). The
            'extracted' key holds the title from source metadata (PDF metadata,
            HTML <title>, or MP4 metadata where available). The 'generated' key
            holds the LLM-generated title (all file types). The
            :attr:`final_title` computed property resolves the final title
            using precedence: generated > extracted > filename.
        description: Description for feed item (optional; extracted from source
            for PDF/HTML files, empty for MP4 files where summary is used instead)
        summary: LLM-generated key takeaways summary (all file types, optional)
        filename: Original filename with extension
        filetype: Source file type: "mp4", "pdf", or "html"; None when
            unavailable for an error entry.
        date: Publication/creation date as UNIX epoch (float, millisecond precision)
        mtime: Source file modification time as a UNIX epoch float.
        publication_date: Publication date from document metadata (PDF/HTML
            only), or None if not available
        size: File size in bytes (for podcast enclosure length)
        duration: Audio duration in seconds (MP4 files only, optional)
        creation_time: ISO-8601 creation time string from media metadata
            (MP4 files only, optional)
        content: Plaintext content — transcript for MP4 files, extracted text
            for PDF/HTML files (optional)
        hash: SHA256 hash of the source file (optional; used for guid generation in
            feeds). When present, the last 12 characters are used as the feed item
            guid. Falls back to the inferred source path when not present.
        transcribe: Step map for transcription (MP4 only)
        summarize: Step map for summarization (all file types)
        extract: Step map for metadata extraction (PDF/HTML only)
        generate_title: Step map for LLM title generation (all file types,
            when LLM title generation runs)
        error: Error message (only present when processing failed)
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    title: dict[str, str | None] | str = Field(
        ...,
        description="Human-readable title. Must be a dict with keys "
        "'filename' (sanitized filename with no extension, no underscores), "
        "'extracted' (title from source metadata), and 'generated' "
        "(LLM-generated title). A plain string is accepted for backward "
        "compatibility but emits a deprecation warning. The final title "
        "uses the precedence: generated > extracted > filename.",
    )
    description: str | None = Field(
        default=None,
        description="Description for feed item (extracted from source for PDF/HTML, "
        "empty for MP4 where summary is used instead)",
    )
    summary: str | None = Field(
        default=None,
        description="LLM-generated key takeaways summary (all file types, optional)",
    )
    filename: str = Field(
        ...,
        description="Original filename with extension",
    )
    filetype: Literal["mp4", "pdf", "html"] | None = Field(
        ...,
        description="Source file type, or null when unavailable for an error entry",
    )
    date: float | None = Field(
        default=None,
        description="Publication/creation date as UNIX epoch (millisecond precision). "
        "For MP4, this is the media creation_time or file mtime. For PDF/HTML, "
        "this is the document metadata date or file mtime.",
    )
    mtime: float | None = Field(
        default=None,
        description="Source file modification time as a UNIX epoch float",
    )
    publication_date: float | None = Field(
        default=None,
        description="Publication date extracted from document metadata (PDF/HTML only) "
        "as UNIX epoch (millisecond precision). None when not available in "
        "document metadata. Falls back to ``date`` for feed generation.",
    )
    size: int = Field(
        ...,
        description="File size in bytes (for podcast enclosure length)",
        ge=0,
    )
    duration: float | None = Field(
        default=None,
        description="Audio duration in seconds (MP4 files only)",
        ge=0,
    )
    creation_time: str | None = Field(
        default=None,
        description="ISO-8601 creation time string from media metadata (MP4 files only)",
    )
    content: str | None = Field(
        default=None,
        description="Plaintext content (transcript for MP4, extracted text for PDF/HTML)",
    )
    transcribe: StepMap | None = Field(
        default=None,
        description="Step map for transcription (MP4 files only)",
    )
    summarize: StepMap | None = Field(
        default=None,
        description="Step map for summarization (all file types)",
    )
    extract: StepMap | None = Field(
        default=None,
        description="Step map for metadata extraction (PDF/HTML files only)",
    )
    generate_title: StepMap | None = Field(
        default=None,
        description="Step map for title generation (all file types, when "
        "LLM title generation runs)",
    )
    hash: str | None = Field(
        default=None,
        description="SHA256 hash of the source file (optional; used for guid generation in feeds)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (only present when processing failed)",
    )

    _source_filepath: str = PrivateAttr(default="")

    @field_validator("error")
    @classmethod
    def _validate_error(cls, v: str | None) -> str | None:
        """Ensure error is not an empty or whitespace-only string."""
        if v is not None and not v.strip():
            raise ValueError("error must not be empty or whitespace-only")
        return v

    @field_validator("title", mode="before")
    @classmethod
    def _validate_title(cls, v: Any) -> Any:
        """Validate the title field.

        - If *v* is a plain string, return it unchanged (backward
          compatibility).  A deprecation warning including the filename is
          emitted by :meth:`_warn_string_title` after model construction.
        - If *v* is a dict, ensure it has all three required keys
          (``filename``, ``extracted``, ``generated``).  Missing keys are
          added with empty-string values so that downstream code can rely
          on the dict always having exactly those keys.
        """
        if isinstance(v, str):
            return v

        if isinstance(v, dict):
            missing = _TITLE_DICT_KEYS - set(v.keys())
            if missing:
                _logger.warning(
                    "Title dict is missing keys %s; adding them with empty values.",
                    sorted(missing),
                )
                for key in missing:
                    v[key] = ""
            return v

        return v

    @model_validator(mode="after")
    def _warn_string_title(self) -> Metadata:
        """Emit a deprecation warning if title is a plain string.

        Unlike the field validator, this runs after model construction so
        that the source filename is available for inclusion in the warning
        message.
        """
        if isinstance(self.title, str):
            _logger.warning(
                "Title is a plain string; string titles are deprecated for file %s. "
                "Use a dict with 'filename', 'extracted', 'generated' keys.",
                self.filename,
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feed_eligible(self) -> bool:
        """Return ``True`` when the file should appear in generated feeds.

        A file is feed-eligible when **all applicable steps** are complete
        and there is no error:

        - **MP4 files**: ``transcribe`` is not ``None`` and
          ``transcribe.status == "complete"`` and ``summarize`` is not
          ``None`` and ``summarize.status == "complete"``.
        - **PDF/HTML files**: ``extract`` is not ``None`` and
          ``extract.status == "complete"`` and ``summarize`` is not
          ``None`` and ``summarize.status == "complete"``.

        If the ``error`` field is set, the file is never eligible.
        """
        if self.error is not None:
            return False
        if self.filetype == "mp4":
            return (
                self.transcribe is not None
                and self.transcribe.status == _STEP_STATUS_COMPLETE
                and self.summarize is not None
                and self.summarize.status == _STEP_STATUS_COMPLETE
            )
        if self.filetype in ("pdf", "html"):
            return (
                self.extract is not None
                and self.extract.status == _STEP_STATUS_COMPLETE
                and self.summarize is not None
                and self.summarize.status == _STEP_STATUS_COMPLETE
            )
        return False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def final_title(self) -> str:
        """Return the final title string from the title field.

        If ``title`` is a dict, the precedence is:

        1. ``generated`` — LLM-generated title (highest precedence)
        2. ``extracted`` — title from source metadata
        3. ``filename`` — filename-derived title (lowest precedence)

        The first non-empty value (after stripping whitespace) is returned.
        If ``title`` is a plain string, it is returned as-is (backward
        compatibility).  Returns an empty string when no suitable value
        is found.

        After precedence resolution, prepend the parent directory name of
        the inferred source path when the title does not already start with it.
        """
        if isinstance(self.title, dict):
            for key in ("generated", "extracted", "filename"):
                value = self.title.get(key)
                if value and value.strip():
                    base_title = value
                    break
            else:
                base_title = ""
        else:
            base_title = self.title

        return _apply_module_prefix_to_title(base_title, self._source_filepath)

    @property
    def source_filepath(self) -> str:
        """Return the inferred source path used for feed URLs and grouping."""
        return self._source_filepath

    # ------------------------------------------------------------------
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary.

        None values are excluded to match the exact schema from design doc
        section 3.3 (e.g., error field is not present when there is no error).
        None values nested inside the title dict are also excluded, since
        pydantic's ``exclude_none`` only applies to top-level fields.
        """
        data = self.model_dump(exclude_none=True, exclude={"feed_eligible", "final_title"})
        if isinstance(data.get("title"), dict):
            data["title"] = {k: v for k, v in data["title"].items() if v is not None}
        return data

    def to_json(self) -> str:
        """Serialize metadata to a JSON string.

        None values are excluded to match the exact schema from design doc
        section 3.3 (e.g., error field is not present when there is no error).
        None values nested inside the title dict are also excluded.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metadata:
        """Deserialize metadata from a dictionary.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            A validated Metadata instance.
        """
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> Metadata:
        """Deserialize metadata from a JSON string.

        Args:
            json_str: JSON string containing metadata fields.

        Returns:
            A validated Metadata instance.
        """
        data = json.loads(json_str)
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        """Write metadata to a JSON file.

        None values are excluded to match the exact schema from design doc
        section 3.3.

        Args:
            path: Path to the output JSON file. Parent directories are
                created if they do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> Metadata:
        """Read metadata from a JSON file.

        Args:
            path: Path to the JSON metadata file.

        Returns:
            A validated Metadata instance.

        Raises:
            json.JSONDecodeError: If the file contains invalid JSON.
            ValidationError: If the data does not match the Metadata schema.
            OSError: If the file cannot be read.
        """
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            _logger.warning("Failed to parse metadata file %s: %s", path, exc)
            log_error(exc, filepath=str(path), stage="metadata_parse")
            raise
        metadata = cls.model_validate(data)
        metadata._source_filepath = str(path.with_name(path.name.removesuffix(".json")))
        return metadata
