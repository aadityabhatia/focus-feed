"""Metadata model for local-feed-generator JSON metadata files.

This module defines the Metadata Pydantic model that matches the JSON schema
specified in design doc section 3.3. Each processed source file has a
corresponding `.json` metadata file with the same name, containing metadata
used for feed generation.

Processing progress is tracked through a unified ``status`` dictionary that
maps step names to their current state:

- ``status["transcribe"]`` — tracks transcription progress (MP4 files only).
- ``status["summarize"]`` — tracks summarization progress (MP4 files only).
- ``status["extract"]`` — tracks metadata extraction progress (PDF/HTML only).

Each status value can be ``None`` (not started), ``"in_progress"``, or
``"complete"``.  For items with errors, the value is set to ``None`` or the
key is omitted.  The :attr:`feed_eligible` computed property returns ``True``
only when all applicable steps are ``"complete"`` and there is no error.

Date fields (``date``, ``processing_date``, ``started_at``, ``completed_at``)
are stored as UNIX epoch integers (seconds since 1970-01-01 UTC).

Schema:
    {
        "status": {
            "transcribe": "in_progress" | "complete" | null,  # MP4 only
            "summarize": "in_progress" | "complete" | null,   # MP4 only
            "extract": "in_progress" | "complete" | null,     # PDF/HTML only
        },
        "title": "string",
        "description": "string",
        "filename": "lecture.mp4",
        "filepath": "videos/lecture.mp4",
        "filetype": "mp4|pdf|html",
        "date": 1728931200,          # UNIX epoch (seconds)
        "size": 12345678,
        "transcript": "Markdown transcript... (mp4 only)",
        "summary": "Key takeaways... (mp4 only)",
        "processing_date": 1728931200,  # UNIX epoch (seconds)
        "transcription_model": "large-v3-turbo (optional, mp4 only)",
        "started_at": 1728931200,       # UNIX epoch, optional
        "completed_at": 1728931200,     # UNIX epoch, optional
        "error": "string (optional, only when processing failed)"
    }

    Note: Version tracking is handled via git commits, not explicit version
    numbers in metadata files.
"""

import json
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# Old deprecated field names from previous metadata schema versions.
# These are no longer part of the Metadata model and will be ignored
# (via extra="ignore") but a warning is printed when encountered.
OLD_FIELDS = {"transcribe_status", "summarize_status", "extract_status", "stage"}


def _warn_old_fields(data: dict[str, Any], filename: str | None = None) -> dict[str, Any]:
    """Check for old deprecated fields in metadata data and print warnings to stderr.

    Args:
        data: Dictionary containing metadata fields.
        filename: Optional filename for the warning message.

    Returns:
        The data dict, with old-style string ``status`` field removed if present
        (to prevent validation errors with the new dict-based schema).
    """
    location = f" in {filename}" if filename else ""
    for field in sorted(OLD_FIELDS):
        if field in data:
            print(
                f"Warning: Old JSON field '{field}' found{location}. "
                f"This field is deprecated. Please regenerate metadata using the new schema.",
                file=sys.stderr,
            )
    # Check for old-style status field (string instead of dict).
    # The new schema uses status as a dict, so a non-dict value indicates
    # the old single-value format.
    if "status" in data and not isinstance(data["status"], dict) and data["status"] is not None:
        print(
            f"Warning: Old JSON field 'status' found{location}. "
            f"This field is deprecated. Please regenerate metadata using the new schema.",
            file=sys.stderr,
        )
        # Remove old status field to prevent validation error
        data = {k: v for k, v in data.items() if k != "status"}
    return data


class Metadata(BaseModel):
    """Pydantic model for JSON metadata matching design doc section 3.3.

    Each processed source file has a corresponding `.json` file with the same
    name, containing metadata used for feed generation.

    Step progress is tracked through a unified ``status`` dictionary:

    - ``status["transcribe"]``: ``"in_progress"`` or ``"complete"`` (MP4 only).
    - ``status["summarize"]``: ``"in_progress"`` or ``"complete"`` (MP4 only).
    - ``status["extract"]``: ``"in_progress"`` or ``"complete"`` (PDF/HTML only).

    The :attr:`feed_eligible` computed property returns ``True`` only when all
    applicable steps are ``"complete"``.  Files with ``feed_eligible == False``
    are excluded from generated feeds.

    Field Descriptions:
        status: Dictionary mapping step names to their status. Keys: "transcribe",
            "summarize" (MP4 only), "extract" (PDF/HTML only). Values: None,
            "in_progress", or "complete".
        title: Human-readable title (LLM-generated if not in source metadata)
        description: Summary/description for feed item
        filename: Original filename with extension
        filepath: Relative path from data directory root
        filetype: Source file type: "mp4", "pdf", or "html"
        date: Publication/creation date as UNIX epoch (seconds)
        size: File size in bytes (for podcast enclosure length)
        transcript: Markdown transcript (MP4 files only, no timestamps)
        summary: Key takeaways from LLM (MP4 files only)
        processing_date: Processing date as UNIX epoch (seconds)
        transcription_model: Model used for transcription (MP4 files only)
        started_at: When processing started, UNIX epoch (seconds)
        completed_at: When processing completed, UNIX epoch (seconds)
        error: Error message (only present when processing failed)
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    status: dict[str, str | None] | None = Field(
        default=None,
        description="Step status map: keys are step names (transcribe, summarize, extract), "
        "values are None, 'in_progress', or 'complete'",
    )
    title: str = Field(
        ...,
        description="Human-readable title (LLM-generated if not in source metadata)",
    )
    description: str = Field(
        ...,
        description="Summary/description for feed item",
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
    transcript: str | None = Field(
        default=None,
        description="Markdown transcript (MP4 files only, no timestamps)",
    )
    summary: str | None = Field(
        default=None,
        description="Key takeaways from LLM (MP4 files only)",
    )
    processing_date: int | None = Field(
        default=None,
        description="Processing date as UNIX epoch (seconds)",
    )
    transcription_model: str | None = Field(
        default=None,
        description="Model used for transcription (MP4 files only)",
    )
    started_at: int | None = Field(
        default=None,
        description="When processing started, UNIX epoch (seconds)",
    )
    completed_at: int | None = Field(
        default=None,
        description="When processing completed, UNIX epoch (seconds)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (only present when processing failed)",
    )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def feed_eligible(self) -> bool:
        """Return ``True`` when the file should appear in generated feeds.

        A file is feed-eligible when **all applicable steps** are complete
        and there is no error:

        - **MP4 files**: ``status.get("transcribe") == "complete"`` **and**
          ``status.get("summarize") == "complete"``.
        - **PDF/HTML files**: ``status.get("extract") == "complete"``.

        If the ``error`` field is set, the file is never eligible.
        """
        if self.error is not None:
            return False
        if self.filetype == "mp4":
            return (
                (self.status or {}).get("transcribe") == "complete"
                and (self.status or {}).get("summarize") == "complete"
            )
        if self.filetype in ("pdf", "html"):
            return (self.status or {}).get("extract") == "complete"
        return False

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_error_field(self) -> Metadata:
        """Validate the ``error`` field.

        - When ``error`` is provided it must be a non-empty string.
        """
        if self.error is not None and not self.error.strip():
            raise ValueError("error field must be a non-empty string when provided")
        return self

    @model_validator(mode="after")
    def validate_conditional_fields(self) -> Metadata:
        """Validate that transcript/summary are only present for MP4 files.

        - transcript and summary fields are only allowed for filetype='mp4'
        """
        if self.filetype != "mp4":
            if self.transcript is not None:
                raise ValueError("transcript is only allowed for filetype='mp4'")
            if self.summary is not None:
                raise ValueError("summary is only allowed for filetype='mp4'")
        return self

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
        data = _warn_old_fields(data)
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
        data = _warn_old_fields(data)
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
        data = _warn_old_fields(data, path.name)
        return cls.model_validate(data)
