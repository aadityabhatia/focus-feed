"""Shared JSON metadata I/O and metadata-state helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from focus_feed.filetypes import (
    DOCUMENT_FILETYPES,
    MEDIA_FILETYPES,
    SUPPORTED_EXTENSIONS,
)
from focus_feed.utils import metadata_sidecar_path

_logger = logging.getLogger("focus_feed")
_TITLE_DICT_KEYS: frozenset[str] = frozenset({"filename", "extracted", "generated"})


def read_metadata_json(json_path: str | Path) -> dict[str, Any] | None:
    """Read a JSON metadata file without logging parse or access failures."""
    try:
        with Path(json_path).open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError, OSError, TypeError:
        return None
    return data if isinstance(data, dict) else None


def write_metadata_json(data: dict[str, Any], source_path: str | Path) -> None:
    """Write schema-complete metadata atomically beside its source file.

    The writer removes the legacy ``filepath`` key, supplies required
    ``Metadata`` fields and source file details when available, and writes the
    result through a temporary file before replacing the final JSON file.
    Null values in *data*, including nested title values, are preserved.

    Args:
        data: Metadata fields to normalize and write.
        source_path: Path to the source file the metadata describes.
    """
    path = Path(source_path)
    json_path = metadata_sidecar_path(path)
    sanitized_stem = path.stem.replace("_", " ")

    data.pop("filepath", None)
    data.setdefault("filename", path.name)
    data.setdefault("filetype", path.suffix.removeprefix(".").lower())

    title = data.get("title")
    if not isinstance(title, dict):
        title = {
            "filename": sanitized_stem,
            "extracted": None,
            "generated": None,
        }
        data["title"] = title
    else:
        for key in _TITLE_DICT_KEYS:
            title.setdefault(key, None)
        filename_title = title.get("filename")
        if not isinstance(filename_title, str) or not filename_title.strip():
            title["filename"] = sanitized_stem

    try:
        stat_result = path.stat()
    except OSError:
        data.setdefault("size", 0)
    else:
        data["size"] = stat_result.st_size
        data["mtime"] = stat_result.st_mtime

    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = json_path.with_name(json_path.name + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary_path.replace(json_path)


def is_orphaned_json(json_path: str | Path) -> bool:
    """Return whether a JSON metadata file has no supported source file."""
    path = Path(json_path)
    # feed.json is a configuration file, not orphaned metadata.
    if path.name == "feed.json":
        return False

    source_filename = path.name[: -len(".json")]
    source_path = path.parent / source_filename

    # The stripped name must have a supported extension to be a real source.
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return True

    return not source_path.exists()


def is_step_complete(metadata: dict[str, Any], step: str) -> bool:
    """Return whether a step is complete and has required textual output.

    Summarize steps require a non-empty ``summary`` and extract steps require
    non-empty ``content`` in addition to ``status == 'complete'``.
    """
    step_map = metadata.get(step)
    if not isinstance(step_map, dict) or step_map.get("status") != "complete":
        return False
    if step == "summarize":
        summary = metadata.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return False
    elif step == "extract":
        content = metadata.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def is_dict_feed_eligible(metadata: dict[str, Any], filetype: str) -> bool:
    """Check whether raw metadata satisfies all feed eligibility steps."""
    if metadata.get("error"):
        return False
    if not isinstance(filetype, str):
        return False

    normalized_filetype = filetype.lower().lstrip(".")
    if normalized_filetype in MEDIA_FILETYPES:
        transcribe = metadata.get("transcribe")
        summarize = metadata.get("summarize")
        if (
            isinstance(transcribe, dict)
            and transcribe.get("error") is not None
            and transcribe.get("status") != "complete"
        ):
            return False
        if (
            isinstance(summarize, dict)
            and summarize.get("error") is not None
            and summarize.get("status") != "complete"
        ):
            return False
        transcribe_complete = (
            isinstance(transcribe, dict) and transcribe.get("status") == "complete"
        )
        summarize_complete = isinstance(summarize, dict) and summarize.get("status") == "complete"
        if not transcribe_complete:
            transcribe_complete = metadata.get("transcribe_status") == "complete"
        if not summarize_complete:
            summarize_complete = metadata.get("summarize_status") == "complete"
        return transcribe_complete and summarize_complete

    if normalized_filetype in DOCUMENT_FILETYPES:
        extract = metadata.get("extract")
        if (
            isinstance(extract, dict)
            and extract.get("error") is not None
            and extract.get("status") != "complete"
        ):
            return False
        extract_complete = isinstance(extract, dict) and extract.get("status") == "complete"
        if not extract_complete:
            extract_complete = metadata.get("extract_status") == "complete"
        if not extract_complete:
            return False

        summarize = metadata.get("summarize")
        if (
            isinstance(summarize, dict)
            and summarize.get("error") is not None
            and summarize.get("status") != "complete"
        ):
            return False
        summarize_complete = isinstance(summarize, dict) and (summarize.get("status") == "complete")
        if not summarize_complete:
            summarize_complete = metadata.get("summarize_status") == "complete"
        return summarize_complete

    return False
