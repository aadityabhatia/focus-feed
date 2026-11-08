"""Private helper functions shared by the Focus Feed CLI."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click

from focus_feed.config import reload_config
from focus_feed.errors import log_error
from focus_feed.filetypes import DOCUMENT_FILETYPES, MEDIA_FILETYPES, SUPPORTED_EXTENSIONS
from focus_feed.metadata import Metadata, StepMap
from focus_feed.metadata_io import (
    is_dict_feed_eligible,
    is_step_complete,
    read_metadata_json,
    write_metadata_json,
)
from focus_feed.scanner import ScannedFile
from focus_feed.utils import (
    SPECIAL_CHAR_PATTERN,
    clean_title,
    now_timestamp,
    should_ignore_filename,
)

_clean_title = clean_title
_SPECIAL_CHAR_PATTERN = SPECIAL_CHAR_PATTERN
_should_ignore_filename = should_ignore_filename
_is_dict_feed_eligible = is_dict_feed_eligible
_logger = logging.getLogger("focus_feed")

# Mapping from uppercase level strings to ``logging`` constants.
_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Valid log-level choices for the CLI option.
_LOG_LEVEL_CHOICES: tuple[str, ...] = tuple(_LOG_LEVEL_MAP.keys())


def _configure_logging_level(log_level: str) -> None:
    """Reconfigure the application loggers to the given level.

    Maps a string such as ``"DEBUG"`` or ``"INFO"`` to the corresponding
    :mod:`logging` constant and applies it to both the main application
    logger (``focus_feed``) and the dedicated error-log logger
    (``focus_feed.error_log``).

    This is called at application startup so that the level chosen via the
    ``--log-level`` CLI option or ``LOG_LEVEL`` environment variable takes
    effect before any other logging occurs.

    Args:
        log_level: Uppercase log level string.  Must be one of
            :data:`_LOG_LEVEL_CHOICES`.  An unrecognised value falls back
            to ``INFO``.
    """
    level = _LOG_LEVEL_MAP.get(log_level.upper(), logging.INFO)

    # Reconfigure the main application logger (stderr StreamHandler).
    _logger.setLevel(level)

    # Error log logger always stays at ERROR level to ensure errors are
    # always written to error.log regardless of the user's log level.
    error_log_logger = logging.getLogger("focus_feed.error_log")
    error_log_logger.setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_config(config_file: str | None) -> None:
    """Load configuration and reload the config singleton.

    If *config_file* is provided its values are loaded into the environment
    (without overriding values already set in the shell).

    Args:
        config_file: Path to a ``.env``-style configuration file, or ``None``.

    Raises:
        click.ClickException: If the configuration cannot be loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore[assignment]

    if config_file:
        if load_dotenv is not None:
            load_dotenv(config_file, override=False)
        else:
            _logger.warning(
                "python-dotenv not available; cannot load config file %s",
                config_file,
            )
    elif load_dotenv is not None:
        # Load .env from the current directory so that env vars are
        # available before _check_required_env_vars() runs.
        load_dotenv(override=False)

    try:
        reload_config()
    except Exception as exc:
        log_error(exc, stage="config_load")
        raise click.ClickException(f"Configuration error: {exc}") from exc

    _logger.debug("Configuration loaded from %s", config_file or "environment")


def _resolve_paths(paths: tuple[str, ...]) -> list[Path]:
    """Resolve relative paths and validate they exist.

    Args:
        paths: Tuple of path strings (may be relative or absolute).

    Returns:
        A list of resolved :class:`~pathlib.Path` objects.

    Raises:
        click.UsageError: If no paths are provided.
        click.ClickException: If any path does not exist.
    """
    if not paths:
        raise click.UsageError("At least one file or directory path is required.")
    resolved: list[Path] = []
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            raise click.ClickException(f"Path does not exist: {p}")
        resolved.append(path)
    return resolved


def _resolve_data_dir_from_paths(paths: list[Path]) -> Path:
    """Determine the data directory from a list of resolved paths.

    Computes the common parent directory of all paths.  If a single path
    is a file, its parent directory is used.

    Args:
        paths: List of resolved :class:`~pathlib.Path` objects (must not be empty).

    Returns:
        An absolute :class:`~pathlib.Path` to the data directory.
    """
    if len(paths) == 1:
        if paths[0].is_dir():
            return paths[0]
        return paths[0].parent
    # Multiple paths: use the common parent directory.
    common = Path(os.path.commonpath([str(p) for p in paths]))
    if common.is_file():
        common = common.parent
    return common


def _expand_paths_to_files(paths: list[Path]) -> list[ScannedFile]:
    """Expand file and directory paths into a list of ScannedFile objects.

    For each path that's a file: check if it's a supported source file
    and create a :class:`ScannedFile`.

    For each path that's a directory: scan it recursively for source files
    and create :class:`ScannedFile` objects for each one found.

    All ``filepath`` values in the returned :class:`ScannedFile` objects are
    relative to the current working directory (not a data directory).

    Args:
        paths: List of resolved :class:`~pathlib.Path` objects (files or dirs).

    Returns:
        A sorted list of :class:`ScannedFile` instances.
    """
    cwd = Path.cwd()
    files: list[ScannedFile] = []

    for path in paths:
        if path.is_file():
            if (
                not _should_ignore_filename(path.name)
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                relative = os.path.relpath(path, cwd)
                files.append(
                    ScannedFile(
                        filepath=relative,
                        absolute_path=str(path),
                        filename=path.name,
                        filetype=path.suffix.lower().lstrip("."),
                    )
                )
        elif path.is_dir():
            for root, _dirs, filenames in os.walk(path):
                root_path = Path(root)
                for filename in filenames:
                    if _should_ignore_filename(filename):
                        continue
                    source_path = root_path / filename
                    if source_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        relative = os.path.relpath(source_path, cwd)
                        files.append(
                            ScannedFile(
                                filepath=relative,
                                absolute_path=str(source_path),
                                filename=filename,
                                filetype=source_path.suffix.lower().lstrip("."),
                            )
                        )

    files.sort(key=lambda f: f.filepath)
    return files


def _find_json_files(data_dir: Path) -> list[Path]:
    """Find all JSON metadata files in *data_dir*.

    Returns a sorted list of paths to ``.json`` files, excluding generated
    feed files (``podcast.xml``, ``articles.xml``, etc.).
    """
    json_files: list[Path] = []
    for path in data_dir.rglob("*.json"):
        if _should_ignore_filename(path.name):
            continue
        json_files.append(path)
    return sorted(json_files)


def _check_special_characters(filepath: Path) -> list[str]:
    """Check if *filepath*'s filename contains special characters.

    Returns a list of human-readable warning strings for any issues found.
    """
    filename = filepath.name
    warnings: list[str] = []
    if filename != filename.strip():
        warnings.append(f"Filename '{filename}' has leading/trailing whitespace: {filepath}")
    matches = _SPECIAL_CHAR_PATTERN.findall(filename)
    if matches:
        unique_chars = sorted(set(matches))
        warnings.append(
            f"Filename '{filename}' contains special characters {unique_chars}: {filepath}"
        )
    return warnings


def _generate_feeds(data_dir_path: Path) -> dict[str, int]:
    """Generate both feeds and return item counts.

    Creates a :class:`~focus_feed.feeds.FeedGenerator`, calls
    ``generate_all_feeds()``, then reads back the JSON metadata files to
    produce a summary of how many items ended up in each feed.

    Returns:
        A dict with ``"podcast"`` and ``"articles"`` keys.
    """
    from focus_feed.feeds import FeedGenerator

    generator = FeedGenerator(data_dir=str(data_dir_path))
    generator.generate_all_feeds()

    # Count items for the summary.
    podcast_count = 0
    article_count = 0
    for json_path in _find_json_files(data_dir_path):
        try:
            metadata = Metadata.from_json_file(json_path)
            if metadata.feed_eligible:
                if metadata.filetype in MEDIA_FILETYPES:
                    podcast_count += 1
                elif metadata.filetype in DOCUMENT_FILETYPES:
                    article_count += 1
        except Exception:
            pass

    return {"podcast": podcast_count, "articles": article_count}


def _print_feed_summary(counts: dict[str, int]) -> None:
    """Print a feed summary to stdout."""
    click.echo(
        f"Feeds generated: {counts['podcast']} podcast item(s), "
        f"{counts['articles']} article item(s)."
    )


# ---------------------------------------------------------------------------
# Granular pipeline helpers
# ---------------------------------------------------------------------------


def _is_path_within(source_file: Path, paths: list[Path]) -> bool:
    """Check if *source_file* matches or is within one of the *paths*.

    A file is considered to be within *paths* if its resolved path equals
    one of the specified paths, or if it resides inside one of the specified
    directories.
    """
    source_resolved = source_file.resolve()
    for p in paths:
        p_resolved = p.resolve()
        if source_resolved == p_resolved:
            return True
        if str(source_resolved).startswith(str(p_resolved) + os.sep):
            return True
    return False


def _find_source_files(data_dir: Path, filetypes: tuple[str, ...]) -> list[Path]:
    """Find all source files of the given types in *data_dir*.

    Recursively walks *data_dir* and returns a sorted list of paths to files
    whose extension (case-insensitive, without the leading dot) matches one of
    *filetypes*.  Generated feed files (``podcast.xml``, etc.) and ``.json``
    metadata files are excluded.
    """
    results: list[Path] = []
    for root, _dirs, files in os.walk(data_dir):
        root_path = Path(root)
        for filename in files:
            if _should_ignore_filename(filename):
                continue
            filepath = root_path / filename
            filetype = filepath.suffix.lower().lstrip(".")
            if filetype in filetypes:
                results.append(filepath)
    return sorted(results)


def _get_json_metadata(json_path: Path) -> dict[str, Any] | None:
    """Read and parse a JSON metadata file.

    Returns ``None`` if the file does not exist or cannot be parsed.
    """
    if not json_path.exists():
        return None
    data = read_metadata_json(json_path)
    if data is None:
        log_error(
            ValueError("Invalid metadata JSON"),
            filepath=str(json_path),
            stage="metadata_parse",
        )
    return data


def _write_json_metadata(data: dict[str, Any], source_path: Path) -> None:
    """Write a metadata dictionary atomically next to *source_path*.

    The JSON file is named ``<source_filename>.json`` and placed in the same
    directory as the source file.  Writes go to a temporary file first, then
    are renamed to the final path to avoid corruption from partial writes.
    """
    write_metadata_json(data, source_path)


def _find_files_needing_transcription(
    data_dir: Path, force: bool = False, paths: list[Path] | None = None
) -> list[Path]:
    """Find media files that need transcription.

    A file needs transcription when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the ``transcribe`` step map's
      ``status`` is not ``"complete"``.

    When *force* is ``True``, all media files are returned regardless of
    their current transcription status.

    When *paths* is provided, only files that match one of the specified
    paths (or are within one of the specified directories) are returned.
    """
    media_files = _find_source_files(data_dir, tuple(MEDIA_FILETYPES))
    needs: list[Path] = []
    for media_file in media_files:
        if paths is not None and not _is_path_within(media_file, paths):
            continue
        json_path = media_file.with_name(media_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if force or metadata is None or not _is_step_complete(metadata, "transcribe"):
            needs.append(media_file)
    _logger.info("Found %d files needing transcription", len(needs))
    return needs


def _find_files_needing_summary(
    data_dir: Path, force: bool = False, paths: list[Path] | None = None
) -> list[Path]:
    """Find media, PDF, and HTML files that have content but need a summary.

    A file needs summarization when:
    - The JSON metadata file exists, **and**
    - It has a non-empty ``content`` field (transcript for media, extracted
      text for PDF/HTML), **and**
    - The ``summarize`` step map's ``status`` is not ``"complete"``.

    When *force* is ``True``, all files with content are returned regardless
    of their current summarization status.

    When *paths* is provided, only files that match one of the specified
    paths (or are within one of the specified directories) are returned.
    """
    source_files = _find_source_files(data_dir, (*MEDIA_FILETYPES, *DOCUMENT_FILETYPES))
    needs: list[Path] = []
    for source_file in source_files:
        if paths is not None and not _is_path_within(source_file, paths):
            continue
        json_path = source_file.with_name(source_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if metadata is None:
            continue
        content = metadata.get("content")
        if not content:
            continue
        if force or not _is_step_complete(metadata, "summarize"):
            needs.append(source_file)
    _logger.info("Found %d files needing summary", len(needs))
    return needs


def _find_files_needing_metadata_extraction(
    data_dir: Path, force: bool = False, paths: list[Path] | None = None
) -> list[Path]:
    """Find PDF/HTML files that need metadata extraction.

    A file needs metadata extraction when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the ``extract`` step map's
      ``status`` is not ``"complete"``, or it is missing a ``title`` field.

    When *force* is ``True``, all PDF/HTML files are returned regardless
    of their current extraction status.

    When *paths* is provided, only files that match one of the specified
    paths (or are within one of the specified directories) are returned.
    """
    doc_files = _find_source_files(data_dir, tuple(DOCUMENT_FILETYPES))
    needs: list[Path] = []
    for doc_file in doc_files:
        if paths is not None and not _is_path_within(doc_file, paths):
            continue
        json_path = doc_file.with_name(doc_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if (
            force
            or metadata is None
            or not _is_step_complete(metadata, "extract")
            or not metadata.get("title")
        ):
            needs.append(doc_file)
    _logger.info("Found %d files needing extraction", len(needs))
    return needs


def _find_files_needing_json_generation(data_dir: Path, force: bool = False) -> list[Path]:
    """Find all source files that need JSON metadata generation.

    A file needs JSON generation when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the file is not feed-eligible
      (i.e. not all applicable step maps have ``status == "complete"``).

    When *force* is ``True``, all source files are returned regardless
    of their current step statuses.
    """
    source_files = _find_source_files(data_dir, tuple(MEDIA_FILETYPES | DOCUMENT_FILETYPES))
    needs: list[Path] = []
    for source_file in source_files:
        json_path = source_file.with_name(source_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if (
            force
            or metadata is None
            or not _is_dict_feed_eligible(metadata, source_file.suffix.lower().lstrip("."))
        ):
            needs.append(source_file)
    return needs


def _find_files_needing_title(
    data_dir: Path, force: bool = False, paths: list[Path] | None = None
) -> list[Path]:
    """Find source files that need title generation.

    A file needs a title when:
    - A JSON metadata file exists for it, **and**
    - The source file exists, **and**
    - The source file has a supported extension, **and**
    - The title does not have a ``generated`` field, **or** *force* is ``True``.

    When *force* is ``True``, all source files with JSON metadata are returned
    regardless of their current title.

    When *paths* is provided, only files that match one of the specified
    paths (or are within one of the specified directories) are returned.

    Args:
        data_dir: Absolute path to the data directory.
        force: When ``True``, include files that already have a generated title.
        paths: Optional list of resolved paths to filter results by.

    Returns:
        A sorted list of absolute paths to source files needing title
        generation.
    """
    needs: list[Path] = []
    for json_file in data_dir.rglob("*.json"):
        if _should_ignore_filename(json_file.name):
            continue
        try:
            metadata_dict = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            continue
        source_file = json_file.with_name(json_file.stem)
        if not source_file.exists():
            continue
        if source_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if paths is not None and not _is_path_within(source_file, paths):
            continue
        title = metadata_dict.get("title", "")
        # Normalize string title to dict for backward compatibility.
        if isinstance(title, str):
            title = {"filename": title, "extracted": None, "generated": None}
        # A file needs a title if 'generated' is not set or empty.
        has_generated = isinstance(title, dict) and bool((title.get("generated") or "").strip())
        if force or not has_generated:
            needs.append(source_file)
    return needs


def _get_content_for_title(file_path: Path, metadata_dict: dict[str, Any]) -> str:
    """Get content for title generation based on file type.

    Uses non-empty content from the metadata dict when available. For PDF/HTML
    files, falls back to reading the first 8000 characters of the source file.

    Args:
        file_path: Path to the source file.
        metadata_dict: The parsed JSON metadata dictionary.

    Returns:
        Content text suitable for title generation, or an empty string if
        the file type is unsupported or content cannot be retrieved.
    """
    content = metadata_dict.get("content")
    if isinstance(content, str) and content:
        return content
    ext = file_path.suffix.lower()
    if ext in (".pdf", ".html"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return content[:8000]
        except OSError:
            return ""
    return ""


def _is_step_complete(metadata: dict[str, Any], step: str) -> bool:
    """Check if a specific step is complete in a raw metadata dict.

    Checks the step map (``transcribe``, ``summarize``, or ``extract``)
    for the given step name.

    Args:
        metadata: The parsed JSON metadata dictionary.
        step: The step name ("transcribe", "summarize", or "extract").

    Returns:
        ``True`` if the step map exists and its ``status`` is ``"complete"``,
        ``False`` otherwise.
    """
    return is_step_complete(metadata, step)


# ---------------------------------------------------------------------------
# Schema constants for cleanup command
# ---------------------------------------------------------------------------

# Allowed top-level keys in metadata JSON files, derived from the Metadata model.
_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(Metadata.model_fields.keys())

# Allowed keys inside the ``title`` dict (when it is a dict, not a string).
_ALLOWED_TITLE_KEYS: frozenset[str] = frozenset({"filename", "extracted", "generated"})

# Allowed keys inside step-map dicts (transcribe, summarize, extract,
# generate_title).
_ALLOWED_STEP_MAP_KEYS: frozenset[str] = frozenset(StepMap.model_fields.keys())

# Step-map keys at the top level of metadata that should be validated as dicts.
_STEP_MAP_KEYS: frozenset[str] = frozenset({"transcribe", "summarize", "extract", "generate_title"})


def _find_unknown_keys(data: dict[str, Any]) -> list[str]:
    """Detect keys in a metadata dict that are not part of the schema.

    Checks:

    - **Top-level keys** against :data:`_ALLOWED_TOP_LEVEL_KEYS`.
    - **Nested ``title`` dict keys** against :data:`_ALLOWED_TITLE_KEYS`
      (only when ``title`` is a dict; a plain-string title is accepted).
    - **Nested step-map dict keys** (``transcribe``, ``summarize``,
      ``extract``, ``generate_title``) against
      :data:`_ALLOWED_STEP_MAP_KEYS` (only when the value is a dict).

    Args:
        data: The parsed JSON metadata dictionary.

    Returns:
        A list of human-readable strings describing each unknown key,
        using dotted path notation for nested keys (e.g.
        ``"transcribe.extra_field"``).
    """
    unknown: list[str] = []

    # --- Top-level keys --------------------------------------------------
    for key in data:
        if key not in _ALLOWED_TOP_LEVEL_KEYS:
            unknown.append(key)

    # --- Nested title dict keys ----------------------------------------
    title = data.get("title")
    if isinstance(title, dict):
        for key in title:
            if key not in _ALLOWED_TITLE_KEYS:
                unknown.append(f"title.{key}")

    # --- Nested step-map keys --------------------------------------------
    for step_key in _STEP_MAP_KEYS:
        step_map = data.get(step_key)
        if isinstance(step_map, dict):
            for key in step_map:
                if key not in _ALLOWED_STEP_MAP_KEYS:
                    unknown.append(f"{step_key}.{key}")

    return unknown


def _cleanup_json_file(json_path: Path) -> tuple[int, list[str]]:
    """Remove unknown keys from a single JSON metadata file.

    Reads the file, detects unknown keys via :func:`_find_unknown_keys`,
    removes them, and rewrites the file only if changes were made.

    Args:
        json_path: Path to the JSON metadata file.

    Returns:
        A tuple of ``(keys_removed_count, unknown_key_paths)``.
        If the file cannot be parsed as JSON, returns ``(0, [])`` and
        logs the error.
    """
    try:
        raw = json_path.read_text(encoding="utf-8")
        data: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        log_error(exc, filepath=str(json_path), stage="metadata_parse")
        return 0, []

    if not isinstance(data, dict):
        _logger.warning("Skipping non-dict JSON file: %s", json_path)
        return 0, []

    unknown_keys = _find_unknown_keys(data)
    # Legacy metadata may still contain fields intentionally removed from the
    # current schema; strip them even when callers do not inspect the report.
    legacy_keys_present = "filepath" in data
    if not unknown_keys and not legacy_keys_present:
        return 0, []

    # Remove top-level unknown keys.
    for key in list(data.keys()):
        if key not in _ALLOWED_TOP_LEVEL_KEYS:
            del data[key]

    # Remove nested title dict unknown keys.
    title = data.get("title")
    if isinstance(title, dict):
        for key in list(title.keys()):
            if key not in _ALLOWED_TITLE_KEYS:
                del title[key]

    # Remove nested step-map unknown keys.
    for step_key in _STEP_MAP_KEYS:
        step_map = data.get(step_key)
        if isinstance(step_map, dict):
            for key in list(step_map.keys()):
                if key not in _ALLOWED_STEP_MAP_KEYS:
                    del step_map[key]

    source_path = json_path.with_name(json_path.name.removesuffix(".json"))
    data["mtime"] = (
        source_path.stat().st_mtime
        if source_path.exists()
        else data.get("mtime", json_path.stat().st_mtime)
    )

    # Rewrite the file with cleaned data.
    try:
        write_metadata_json(data, source_path)
    except OSError as exc:
        log_error(exc, filepath=str(json_path), stage="metadata_write")
        raise

    return len(unknown_keys), unknown_keys


def _find_feed_eligible_sources(
    data_dir_path: Path,
    paths: list[Path] | None = None,
) -> list[Path]:
    """Find source files whose JSON sidecar exists, parses, and has a summary.

    Prints a note and skips items that have no JSON sidecar, an
    unparseable sidecar, or a sidecar without a non-empty summary.

    Args:
        data_dir_path: Absolute path to the data directory.
        paths: Optional list of resolved paths to limit the scan to.

    Returns:
        A list of absolute source file paths eligible for feed generation.
    """
    eligible: list[Path] = []
    for source_file in _find_source_files(
        data_dir_path, tuple(MEDIA_FILETYPES | DOCUMENT_FILETYPES)
    ):
        if paths is not None and not _is_path_within(source_file, paths):
            continue
        relative = os.path.relpath(source_file, Path.cwd())
        json_path = source_file.with_name(source_file.name + ".json")
        if not json_path.exists():
            click.echo(f"Skipping {relative}: no JSON metadata sidecar found.")
            continue
        metadata = _get_json_metadata(json_path)
        if metadata is None or not isinstance(metadata, dict):
            click.echo(f"Skipping {relative}: JSON metadata could not be parsed.")
            continue
        summary = metadata.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            click.echo(f"Skipping {relative}: no summary in JSON metadata.")
            continue
        eligible.append(source_file)
    return eligible


def _finalize_incomplete_json(
    data_dir_path: Path,
    paths: list[Path] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Finalize incomplete JSON metadata files in the data directory.

    Finds media and document source files that have incomplete JSON metadata
    (i.e., not all applicable step maps have ``status == "complete"``) and
    assembles the final metadata by:

    - Filling in all required fields, using existing values from intermediate
      JSON, file system statistics, or LLM-generated titles and summaries as
      fallbacks.
    - Setting all applicable step maps to ``"complete"`` status.
    - Preserving existing ``content`` (transcript for media, extracted text for
      documents) fields.
    - Generating LLM summaries for PDF/HTML files when not already present.

    This makes files feed-eligible so they appear in generated feeds.

    When *paths* is provided, only files within those paths are finalized.
    When *force* is ``True``, JSON is recreated even if the file is already
    feed-eligible.

    Args:
        data_dir_path: Absolute path to the data directory.
        paths: Optional list of resolved paths to limit finalization to.
        force: When ``True``, finalize even feed-eligible files.

    Returns:
        A tuple of ``(success_count, error_count)``.
    """
    from focus_feed.config import get_config
    from focus_feed.services.summarizer import Summarizer

    if paths is not None:
        # Filter to only source files within the given paths.
        all_files = _find_files_needing_json_generation(data_dir_path, force=force)
        files = []
        for f in all_files:
            if _is_path_within(f, paths):
                files.append(f)
    else:
        files = _find_files_needing_json_generation(data_dir_path, force=force)

    if not files:
        all_source_files = _find_source_files(
            data_dir_path, tuple(MEDIA_FILETYPES | DOCUMENT_FILETYPES)
        )
        if not all_source_files:
            _logger.warning(
                "No supported source files found in %s for JSON generation. "
                "Expected .mp4, .pdf, .html, .docx, .doc, .pptx, .ppt, "
                ".xlsx, or .xls files.",
                data_dir_path,
            )
        else:
            _logger.debug(
                "All %d source files in %s are already feed-eligible; no JSON generation needed.",
                len(all_source_files),
                data_dir_path,
            )
        return 0, 0

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for source_file in files:
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            # Skip files with errors — they should not be finalized.
            if metadata.get("error"):
                continue

            # Preserve producer step failures; finalization must not turn them
            # into apparently successful metadata.
            if any(
                isinstance(step_map, dict)
                and step_map.get("error") is not None
                and step_map.get("status") != "complete"
                for step_map in (
                    metadata.get("transcribe"),
                    metadata.get("summarize"),
                    metadata.get("extract"),
                    metadata.get("generate_title"),
                )
            ):
                continue

            filetype = source_file.suffix.lower().lstrip(".")
            stat_result = source_file.stat()
            file_size = stat_result.st_size
            file_mtime = stat_result.st_mtime
            config = get_config()

            # Ensure required fields are present.
            title = metadata.get("title")
            # Normalize string title to dict for backward compatibility.
            if isinstance(title, str):
                title = {"filename": title, "extracted": None, "generated": None}
            if not isinstance(title, dict):
                title = {}
            # Ensure all three required keys are present.
            title.setdefault("filename", None)
            title.setdefault("extracted", None)
            title.setdefault("generated", None)
            filename_title = title.get("filename") or _clean_title(source_file.stem)
            extracted = title.get("extracted")

            # Generate title via LLM if not already present.
            title_generated_tokens: tuple[int, int] | None = None
            generate_title_content: str | None = None
            if not title.get("generated") or not title["generated"].strip():
                generated = filename_title
                try:
                    if filetype in MEDIA_FILETYPES:
                        content = metadata.get("content") or ""
                        if content.strip():
                            generate_title_content = content
                            generated, prompt_tokens, completion_tokens = summarizer.generate_title(
                                content,
                                filename=filename_title,
                            )
                            title_generated_tokens = (prompt_tokens, completion_tokens)
                            generated = _clean_title(generated) or generated
                    else:
                        content = _get_content_for_title(source_file, metadata)
                        if content.strip():
                            generate_title_content = content
                            generated, prompt_tokens, completion_tokens = summarizer.generate_title(
                                content,
                                filename=filename_title,
                                extracted=extracted,
                            )
                            title_generated_tokens = (prompt_tokens, completion_tokens)
                            generated = _clean_title(generated) or generated
                except Exception as gen_exc:
                    _logger.warning(
                        "Failed to generate title for %s: %s; falling back to filename",
                        source_file,
                        gen_exc,
                    )
                    generated = filename_title
                title["generated"] = generated

            # Ensure filename key is set.
            title["filename"] = filename_title

            description = metadata.get("description")
            summary = metadata.get("summary")

            # Generate summary via LLM for PDF/HTML if not already present.
            if filetype in DOCUMENT_FILETYPES and not summary:
                try:
                    content = _get_content_for_title(source_file, metadata)
                    if content and content.strip():
                        summary, _, _ = summarizer.summarize(
                            content,
                            filepath=str(source_file),
                            filetype=filetype,
                        )
                except Exception as sum_exc:
                    _logger.warning(
                        "Failed to generate summary for %s: %s",
                        source_file,
                        sum_exc,
                    )
                    summary = ""

            # Build the final metadata dict.
            final_metadata: dict[str, Any] = {
                key: value for key, value in metadata.items() if key in _ALLOWED_TOP_LEVEL_KEYS
            }
            final_metadata.update(
                {
                    "title": title,
                    "description": description,
                    "filename": source_file.name,
                    "filetype": filetype,
                    "date": metadata.get("date") or file_mtime,
                    "mtime": stat_result.st_mtime,
                    "size": metadata.get("size") or file_size,
                }
            )

            # Set step maps to complete and include type-specific fields.
            now_ts = now_timestamp()
            if filetype in MEDIA_FILETYPES:
                # For MP4 files, description is left blank and summary holds
                # the LLM-generated key takeaways.
                final_metadata["description"] = ""
                final_metadata["summary"] = summary or ""
                final_metadata["transcribe"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": file_size,
                    "output_size": len((metadata.get("content") or "").encode("utf-8")),
                    "words": len((metadata.get("content") or "").split()),
                    "model": config.transcription_remote_model or "remote",
                }
                final_metadata["summarize"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": len((metadata.get("content") or "").encode("utf-8")),
                    "output_size": len((summary or "").encode("utf-8")),
                    "words": len((summary or "").split()),
                    "model": config.llm_model,
                }
                final_metadata["content"] = metadata.get("content") or ""
                # Add generate_title step map when the LLM was called.
                if title_generated_tokens is not None and generate_title_content is not None:
                    prompt_tokens, completion_tokens = title_generated_tokens
                    final_metadata["generate_title"] = {
                        "status": "complete",
                        "start_time": now_ts,
                        "end_time": now_ts,
                        "input_size": len(generate_title_content.encode("utf-8")),
                        "output_size": len((title.get("generated") or "").encode("utf-8")),
                        "words": len((title.get("generated") or "").split()),
                        "model": config.llm_model,
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                    }
            elif filetype in DOCUMENT_FILETYPES:
                final_metadata["extract"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": file_size,
                    "output_size": len((metadata.get("content") or "").encode("utf-8")),
                    "words": len((metadata.get("content") or "").split()),
                }
                final_metadata["summarize"] = {
                    "status": "complete",
                    "start_time": now_ts,
                    "end_time": now_ts,
                    "input_size": len((metadata.get("content") or "").encode("utf-8")),
                    "output_size": len((summary or "").encode("utf-8")),
                    "words": len((summary or "").split()),
                    "model": config.llm_model,
                }
                final_metadata["summary"] = summary or ""
                final_metadata["content"] = metadata.get("content")
                # Add generate_title step map when the LLM was called.
                if title_generated_tokens is not None and generate_title_content is not None:
                    prompt_tokens, completion_tokens = title_generated_tokens
                    final_metadata["generate_title"] = {
                        "status": "complete",
                        "start_time": now_ts,
                        "end_time": now_ts,
                        "input_size": len(generate_title_content.encode("utf-8")),
                        "output_size": len((title.get("generated") or "").encode("utf-8")),
                        "words": len((title.get("generated") or "").split()),
                        "model": config.llm_model,
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                    }

            _write_json_metadata(final_metadata, source_file)

            success_count += 1
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(source_file), stage="json_finalization")

    return success_count, error_count
