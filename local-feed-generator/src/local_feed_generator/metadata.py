"""Metadata model for local-feed-generator JSON metadata files.

This module defines the Metadata Pydantic model that matches the JSON schema
specified in design doc section 3.3. Each processed source file has a
corresponding `.json` metadata file with the same name, containing metadata
used for feed generation.

Processing progress is tracked through separate step maps for each processing
step:

- ``transcribe``: Step map for transcription progress (MP4 files only).
- ``summarize``: Step map for summarization progress (MP4 files only).
- ``extract``: Step map for metadata extraction progress (PDF/HTML only).

Each step map contains a ``status`` field that can be ``None`` (not started),
``"in_progress"``, or ``"complete"``.  The :attr:`feed_eligible` computed
property returns ``True`` only when all applicable steps are ``"complete"``
and there is no error.

Date fields (``date``) are stored as UNIX epoch integers (seconds since
1970-01-01 UTC).  Per-step timing is tracked within each step map's
``start_time`` and ``end_time`` fields.

Schema:
    {
        "title": "string",
        "description": "string (optional, extracted from source for PDF/HTML, empty for MP4)",
        "summary": "string (optional, LLM-generated key takeaways for MP4)",
        "filename": "lecture.mp4",
        "filepath": "videos/lecture.mp4",
        "filetype": "mp4|pdf|html",
        "date": 1728931200,          # UNIX epoch (seconds)
        "size": 12345678,
        "duration": 3600.5,          # Audio duration in seconds (MP4, optional)
        "creation_time": "2024-01-15T10:30:00.000000Z",  # Media creation time
        "content": "Plaintext content (transcript for MP4, extracted text for PDF/HTML)",
        "transcribe": {
            "status": "complete",
            "start_time": 1792096505,
            "end_time": 1792096515,
            "input_size": 12345678,
            "output_size": 456789,
            "words": 452,
            "model": "large-v3-turbo",
            "input_tokens": 1234,
            "output_tokens": 567
        },
        "summarize": {
            "status": "complete",
            "start_time": 1792096515,
            "end_time": 1792096520,
            "input_size": 456789,
            "output_size": 1997,
            "words": 345,
            "model": "mistralai/Mistral-7B-Instruct",
            "input_tokens": 890,
            "output_tokens": 123
        },
        "extract": {
            "status": "complete",
            "start_time": 1792096500,
            "end_time": 1792096505,
            "input_size": 1790891,
            "output_size": 12345,
            "words": 543
        },
        "error": "string (optional, only when processing failed)"
    }

    Note: Version tracking is handled via git commits, not explicit version
    numbers in metadata files.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# Step status values.
_STEP_STATUS_IN_PROGRESS = "in_progress"
_STEP_STATUS_COMPLETE = "complete"


class StepMap(BaseModel):
    """Metadata for a single processing step.

    Each processing step (transcribe, summarize, extract) gets its own
    :class:`StepMap` instance that tracks the step's status, timing, sizes,
    and (for LLM-based steps) model and token usage.

    Attributes:
        status: ``None``, ``"in_progress"``, or ``"complete"``.
        start_time: UNIX epoch timestamp when the step started, or ``None``.
        end_time: UNIX epoch timestamp when the step ended, or ``None``.
        input_size: Input size in bytes.
        output_size: Output size in bytes.
        words: Output word count (for text output steps), or ``None``.
        model: Model identifier used (LLM-based steps only: transcribe,
            summarize), or ``None``.
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
    start_time: int | None = Field(
        default=None,
        description="UNIX epoch start time",
    )
    end_time: int | None = Field(
        default=None,
        description="UNIX epoch end time",
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
    - ``summarize``: :class:`StepMap` for summarization (MP4 only).
    - ``extract``: :class:`StepMap` for metadata extraction (PDF/HTML only).

    The :attr:`feed_eligible` computed property returns ``True`` only when all
    applicable steps are ``"complete"``.  Files with ``feed_eligible == False``
    are excluded from generated feeds.

    Field Descriptions:
        title: Human-readable title (LLM-generated if not in source metadata)
        description: Description for feed item (optional; extracted from source
            for PDF/HTML files, empty for MP4 files where summary is used instead)
        summary: LLM-generated key takeaways summary (MP4 files only, optional)
        filename: Original filename with extension
        filepath: Relative path from data directory root
        filetype: Source file type: "mp4", "pdf", or "html"
        date: Publication/creation date as UNIX epoch (seconds)
        size: File size in bytes (for podcast enclosure length)
        duration: Audio duration in seconds (MP4 files only, optional)
        creation_time: ISO-8601 creation time string from media metadata
            (MP4 files only, optional)
        content: Plaintext content — transcript for MP4 files, extracted text
            for PDF/HTML files (optional)
        transcribe: Step map for transcription (MP4 only)
        summarize: Step map for summarization (MP4 only)
        extract: Step map for metadata extraction (PDF/HTML only)
        error: Error message (only present when processing failed)
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    title: str = Field(
        ...,
        description="Human-readable title (LLM-generated if not in source metadata)",
    )
    description: str | None = Field(
        default=None,
        description="Description for feed item (extracted from source for PDF/HTML, "
        "empty for MP4 where summary is used instead)",
    )
    summary: str | None = Field(
        default=None,
        description="LLM-generated key takeaways summary (MP4 files only)",
    )
    filename: str = Field(
        ...,
        description="Original filename with extension",
    )
    filepath: str = Field(
        ...,
        description="Relative path from data directory root",
    )
    filetype: Literal["mp4", "pdf", "html"] = Field(
        ...,
        description="Source file type",
    )
    date: int | None = Field(
        default=None,
        description="Publication/creation date as UNIX epoch (seconds)",
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
        description="Step map for summarization (MP4 files only)",
    )
    extract: StepMap | None = Field(
        default=None,
        description="Step map for metadata extraction (PDF/HTML files only)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (only present when processing failed)",
    )

    @field_validator("error")
    @classmethod
    def _validate_error(cls, v: str | None) -> str | None:
        """Ensure error is not an empty or whitespace-only string."""
        if v is not None and not v.strip():
            raise ValueError("error must not be empty or whitespace-only")
        return v

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def feed_eligible(self) -> bool:
        """Return ``True`` when the file should appear in generated feeds.

        A file is feed-eligible when **all applicable steps** are complete
        and there is no error:

        - **MP4 files**: ``transcribe`` is not ``None`` and
          ``transcribe.status == "complete"`` and ``summarize`` is not
          ``None`` and ``summarize.status == "complete"``.
        - **PDF/HTML files**: ``extract`` is not ``None`` and
          ``extract.status == "complete"``.

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
            )
        return False

    # ------------------------------------------------------------------
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary.

        None values are excluded to match the exact schema from design doc
        section 3.3 (e.g., error field is not present when there is no error).
        """
        return self.model_dump(exclude_none=True, exclude={"feed_eligible"})

    def to_json(self) -> str:
        """Serialize metadata to a JSON string.

        None values are excluded to match the exact schema from design doc
        section 3.3 (e.g., error field is not present when there is no error).
        """
        return self.model_dump_json(exclude_none=True, exclude={"feed_eligible"})

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
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
