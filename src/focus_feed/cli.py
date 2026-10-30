"""CLI for Focus Feed.

Provides a Click-based command-line interface with granular subcommands for
each step of the processing pipeline:

- ``process``          — Batch process all files in the given paths.
- ``watch``            — Watch directories for changes (incremental mode).
- ``feeds``            — Generate RSS feeds from existing JSON metadata.
- ``validate``         — Validate all JSON metadata files.
- ``transcribe``       — Transcribe MP4 files to markdown text.
- ``summarize``        — Generate summaries from existing content.
- ``extract``          — Extract metadata from PDF/HTML files.
- ``title``            — Generate titles for files that lack proper titles.
- ``create-feed``      — Create a template ``feed.json`` file in a target directory.
- ``cleanup``          — Remove unknown keys from JSON metadata files.

Files and directories are passed as positional arguments, e.g.::

    uv run focus-feed process ./data
    uv run focus-feed transcribe videos/lecture.mp4 --force
    uv run focus-feed watch ./data
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import click
from watchdog.events import FileSystemEventHandler

from focus_feed.config import reload_config
from focus_feed.errors import log_error
from focus_feed.feeds import _is_orphaned_json
from focus_feed.metadata import Metadata, StepMap
from focus_feed.producer import _check_reflection_md
from focus_feed.scanner import (
    _IGNORE_FILENAMES,
    _SPECIAL_CHAR_PATTERN,
    SUPPORTED_EXTENSIONS,
    ScannedFile,
    Scanner,
    _is_dict_feed_eligible,
)
from focus_feed.services.extractors import _clean_title

_logger = logging.getLogger("focus_feed")


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

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
    (``.mp4``, ``.pdf``, ``.html``) and create a :class:`ScannedFile`.

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
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
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
                    if filename.lower() in _IGNORE_FILENAMES:
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
        if path.name.lower() in _IGNORE_FILENAMES:
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
                if metadata.filetype == "mp4":
                    podcast_count += 1
                elif metadata.filetype in ("pdf", "html"):
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
            if filename.lower() in _IGNORE_FILENAMES:
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
    try:
        data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        log_error(exc, filepath=str(json_path), stage="metadata_parse")
        return None


def _write_json_metadata(data: dict[str, Any], source_path: Path) -> None:
    """Write a metadata dictionary atomically next to *source_path*.

    The JSON file is named ``<source_filename>.json`` and placed in the same
    directory as the source file.  Writes go to a temporary file first, then
    are renamed to the final path to avoid corruption from partial writes.
    """
    json_path = source_path.with_name(source_path.name + ".json")
    data.pop("filepath", None)
    data.pop("extracted_title", None)
    data["mtime"] = source_path.stat().st_mtime
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(json_path)
    except OSError as exc:
        log_error(exc, filepath=str(json_path), stage="metadata_write")
        raise


def _find_files_needing_transcription(
    data_dir: Path, force: bool = False, paths: list[Path] | None = None
) -> list[Path]:
    """Find MP4 files that need transcription.

    A file needs transcription when:
    - No JSON metadata file exists, **or**
    - The JSON metadata file exists but the ``transcribe`` step map's
      ``status`` is not ``"complete"``.

    When *force* is ``True``, all MP4 files are returned regardless of
    their current transcription status.

    When *paths* is provided, only files that match one of the specified
    paths (or are within one of the specified directories) are returned.
    """
    mp4_files = _find_source_files(data_dir, ("mp4",))
    needs: list[Path] = []
    for mp4_file in mp4_files:
        if paths is not None and not _is_path_within(mp4_file, paths):
            continue
        json_path = mp4_file.with_name(mp4_file.name + ".json")
        metadata = _get_json_metadata(json_path)
        if force or metadata is None or not _is_step_complete(metadata, "transcribe"):
            needs.append(mp4_file)
    _logger.info("Found %d files needing transcription", len(needs))
    return needs


def _find_files_needing_summary(
    data_dir: Path, force: bool = False, paths: list[Path] | None = None
) -> list[Path]:
    """Find MP4, PDF, and HTML files that have content but need a summary.

    A file needs summarization when:
    - The JSON metadata file exists, **and**
    - It has a non-empty ``content`` field (transcript for MP4, extracted
      text for PDF/HTML), **and**
    - The ``summarize`` step map's ``status`` is not ``"complete"``.

    When *force* is ``True``, all files with content are returned regardless
    of their current summarization status.

    When *paths* is provided, only files that match one of the specified
    paths (or are within one of the specified directories) are returned.
    """
    source_files = _find_source_files(data_dir, ("mp4", "pdf", "html"))
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
    doc_files = _find_source_files(data_dir, ("pdf", "html"))
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
    source_files = _find_source_files(data_dir, ("mp4", "pdf", "html"))
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
    - The source file has a supported extension (``.mp4``, ``.pdf``, ``.html``), **and**
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
        if json_file.name.lower() in _IGNORE_FILENAMES:
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

    For MP4 files, uses the content (transcript) from the metadata dict.
    For PDF/HTML files, reads the first 8000 characters of the file content.

    Args:
        file_path: Path to the source file.
        metadata_dict: The parsed JSON metadata dictionary.

    Returns:
        Content text suitable for title generation, or an empty string if
        the file type is unsupported or content cannot be retrieved.
    """
    ext = file_path.suffix.lower()
    if ext == ".mp4":
        return str(metadata_dict.get("content", ""))
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
    step_map = metadata.get(step)
    return isinstance(step_map, dict) and step_map.get("status") == "complete"


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
    legacy_keys_present = "filepath" in data or "extracted_title" in data
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
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        log_error(exc, filepath=str(json_path), stage="metadata_write")
        raise

    return len(unknown_keys), unknown_keys


# ---------------------------------------------------------------------------
# Shared option decorators
# ---------------------------------------------------------------------------


def _config_file_option(func: Any) -> Any:
    """Add the ``--config-file`` optional option to a Click command."""
    return click.option(
        "--config-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="Path to a .env-style configuration file.",
    )(func)


def _force_option(func: Any) -> Any:
    """Add the ``--force`` flag to a Click command.

    When specified, the step is run even if it appears to already be
    complete (based on the ``status`` dict).
    """
    return click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Run the step even if it appears to already be complete.",
    )(func)


def _dry_run_option(func: Any) -> Any:
    """Add the ``--dry-run`` flag to a Click command."""
    return click.option(
        "--dry-run",
        is_flag=True,
        default=False,
        help="Show what would be processed without making any changes.",
    )(func)


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="focus-feed")
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(_LOG_LEVEL_CHOICES, case_sensitive=False),
    default=None,
    envvar="LOG_LEVEL",
    help="Set the logging level (default: INFO). "
    "Can also be set via the LOG_LEVEL environment variable.",
)
def app(log_level: str | None) -> None:
    """Focus Feed — Privacy-first RSS/Atom feed generator.

    Processes PDF, HTML, and MP4 files from the given paths and generates
    RSS feeds for personal consumption through mobile applications.
    """
    # Configure logging level early, before any subcommand logging occurs.
    # Precedence: CLI option > LOG_LEVEL env var > default INFO.
    effective_level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    _configure_logging_level(effective_level)


# ---------------------------------------------------------------------------
# Process command  (T5.2)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def process(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Process all files in the given paths (batch mode).

    Scans the given paths for content files (MP4, PDF, HTML) that need
    processing, processes each file through the Producer pipeline,
    and generates both RSS feeds afterwards.

    When ``--force`` is specified, all files are processed even if
    already feed-eligible.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting process command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    # --- Scan -----------------------------------------------------------
    if force:
        # Force processing: expand all source files regardless of status.
        files = _expand_paths_to_files(resolved_paths)
    else:
        scanner = Scanner(data_dir=data_dir_path)
        scanned = scanner.scan_directory()
        # Adjust filepath to be relative to CWD instead of data_dir.
        cwd = Path.cwd()
        files = [
            ScannedFile(
                filepath=os.path.relpath(Path(f.absolute_path), cwd),
                absolute_path=f.absolute_path,
                filename=f.filename,
                filetype=f.filetype,
                current_stage=f.current_stage,
                json_exists=f.json_exists,
            )
            for f in scanned
        ]

    if not files:
        click.echo("No files to process.")
        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)
        return

    click.echo(f"Found {len(files)} file(s) to process.")
    _logger.info("Found %d file(s) to process", len(files))

    # --- Process --------------------------------------------------------
    from focus_feed.producer import Producer

    producer = Producer(force=force)

    success_count = 0
    error_count = 0

    for i, file in enumerate(files, 1):
        click.echo(f"[{i}/{len(files)}] Processing {file.filepath}...")
        try:
            result = producer.process_file(file)
            if result is not None and result.feed_eligible:
                success_count += 1
            else:
                error_count += 1
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=file.filepath, stage="processing")

    click.echo(f"\nProcessing complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Process complete: %d succeeded, %d failed", success_count, error_count)

    # --- Generate feeds -------------------------------------------------
    counts = _generate_feeds(data_dir_path)
    _print_feed_summary(counts)

    if error_count > 0:
        _logger.warning("Exiting with code 1 due to %d error(s)", error_count)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Watch command  (T5.3)
# ---------------------------------------------------------------------------


class _WatchHandler(FileSystemEventHandler):
    """File-system event handler for watch mode.

    On file creation or modification:

    * **Source files** (``.mp4``, ``.pdf``, ``.html``) — process the file
      and regenerate feeds.
    * **JSON files** — check whether a corresponding source file exists;
      if not, log a warning about an orphaned JSON file.
    * **Other files** — ignored.

    Events are debounced per file path: rapid successive events for the
    same file within the :attr:`_debounce_interval` window are collapsed
    into a single processing.  Different files are tracked independently
    and are not blocked by each other's debounce timers.
    """

    #: Minimum time (seconds) between processing the same file path.
    _debounce_interval: float = 1.0

    # Maximum number of error log entries per minute before rate limiting
    # suppresses further messages (see T6.4).
    _max_errors_per_minute = 10

    # Window in seconds for error rate limiting.
    _error_window_seconds = 60

    # Periodic rescan interval in seconds (T6.3).
    # Fallback for file systems (NFS, network drives, etc.) that may not
    # reliably trigger watchdog events.  Can be overridden via the
    # ``RESCAN_INTERVAL`` environment variable in the ``watch`` command.
    # A value of ``0`` disables the periodic rescan.
    _rescan_interval: int = 300

    def __init__(self, data_dir: Path, producer: Any) -> None:
        self.data_dir = data_dir
        self.producer = producer
        _logger.info("Watch handler initialized for %s", str(self.data_dir))
        # Last event timestamp per file path, for debouncing (T6.2).
        self._last_event: dict[str, float] = {}
        # Lock protecting _last_event, since watchdog dispatches events
        # from a separate thread.
        self._debounce_lock = threading.Lock()
        # Timestamps of recent errors, used for rate limiting (T6.4).
        self._error_timestamps: list[float] = []
        # Timestamp of the last periodic rescan, or None if no rescan
        # has occurred yet (T6.3).
        self.last_rescan_time: float | None = None

    # watchdog callbacks --------------------------------------------------

    def on_created(self, event: Any) -> None:
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_modified(self, event: Any) -> None:
        if not event.is_directory:
            path = event.src_path
            _logger.info("Detected changes in %s", str(path))
            self._handle_event(path)

    # internal helpers ----------------------------------------------------

    def _log_processing_error(
        self,
        exc: Exception,
        filepath: str,
        stage: str | None = None,
    ) -> None:
        """Log a processing error with rate limiting (T6.4).

        Tracks error timestamps to prevent log flooding. After
        ``_max_errors_per_minute`` errors within a 60-second window, further
        detailed error messages are suppressed and a summary message is logged
        instead.

        Args:
            exc: The exception that occurred.
            filepath: Path of the file being processed.
            stage: Optional processing stage label for additional context.
        """
        now = time.monotonic()
        # Trim timestamps older than the error window.
        self._error_timestamps = [
            ts for ts in self._error_timestamps if now - ts < self._error_window_seconds
        ]

        if len(self._error_timestamps) >= self._max_errors_per_minute:
            # Rate-limited: log a summary instead of the detailed error.
            _logger.warning(
                "Error rate limit reached (%d errors in %ds); "
                "suppressing detailed error for file=%s",
                self._max_errors_per_minute,
                self._error_window_seconds,
                filepath,
            )
            return

        # Record this error timestamp and log the detailed error.
        self._error_timestamps.append(now)
        log_error(exc, filepath=filepath, stage=stage)

    def _handle_event(self, filepath: str) -> None:
        """Dispatch a single file-system event.

        Handles exceptions internally so that a single file processing error
        does not crash the watcher (T6.4). The watcher continues monitoring
        after errors.

        Implements per-filepath debouncing (T6.2): if an event for the same
        path was received within :attr:`_debounce_interval` seconds, the
        event is skipped and the debounce window is extended.
        """
        now = time.monotonic()
        with self._debounce_lock:
            last = self._last_event.get(filepath)
            if last is not None and (now - last) < self._debounce_interval:
                # Within the debounce window — extend the window and skip.
                self._last_event[filepath] = now
                return
            self._last_event[filepath] = now

        source_path = Path(filepath)

        # --- JSON events: check for orphaned JSON -----------------------
        if source_path.suffix.lower() == ".json":
            self._handle_json_event(source_path)
            return

        # --- Source files: process and regenerate feeds -----------------
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # Check if the file has an error state in its JSON metadata and the
        # source file is not newer than the JSON.  In watch mode, a file
        # modification event may fire even when the source content hasn't
        # actually changed (e.g. metadata touch).  If the JSON already has
        # an error and the source isn't newer, there's no point retrying
        # without --force.
        json_path = source_path.with_name(source_path.name + ".json")
        if json_path.exists():
            existing_metadata = _get_json_metadata(json_path)
            if existing_metadata and existing_metadata.get("error"):
                source_mtime = source_path.stat().st_mtime
                json_mtime = json_path.stat().st_mtime
                if source_mtime <= json_mtime:
                    _logger.warning(
                        "File %s has error state but source not modified; use --force to retry",
                        filepath,
                    )
                    return

        try:
            relative_path = os.path.relpath(source_path, Path.cwd())
            scanned = ScannedFile(
                filepath=relative_path,
                absolute_path=str(source_path),
                filename=source_path.name,
                filetype=source_path.suffix.lower().lstrip("."),
            )

            click.echo(f"Processing {scanned.filepath}...")
            self.producer.process_file(scanned)

            from focus_feed.feeds import FeedGenerator

            generator = FeedGenerator(data_dir=str(self.data_dir))
            generator.generate_all_feeds()
            click.echo(f"Feeds regenerated after processing {scanned.filepath}.")
        except Exception as exc:
            # Log the error without crashing the watcher (T6.4).
            # The watcher continues monitoring after errors.
            self._log_processing_error(
                exc,
                filepath=str(source_path),
                stage="processing",
            )

    def _handle_json_event(self, json_path: Path) -> None:
        """Check a JSON file for orphaned status."""
        # feed.json is a configuration file, not orphaned metadata.
        if json_path.name == "feed.json":
            return

        source_filename = json_path.name[: -len(".json")]
        source_path = json_path.parent / source_filename

        if not source_path.exists():
            _logger.warning("Orphaned JSON file detected (no source file): %s", json_path)
        elif source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            _logger.warning(
                "Orphaned JSON file detected (unsupported source extension): %s",
                json_path,
            )

    # ------------------------------------------------------------------
    # Periodic rescan (T6.3)
    # ------------------------------------------------------------------

    def _periodic_rescan(self) -> None:
        """Periodically scan the data directory for missed files.

        This is a fallback for file systems (NFS, network drives, etc.)
        that may not reliably trigger watchdog events.  Uses the
        :class:`~focus_feed.scanner.Scanner` to find files that
        need processing based on JSON metadata freshness (missing or
        outdated JSON, incomplete step statuses).  The scanner already
        handles duplicate detection via file modification times, so files
        that were already processed by the event handler will not be
        returned.

        After processing any newly-discovered files, feeds are regenerated.
        """
        self.last_rescan_time = time.time()
        _logger.info("Periodic rescan started for %s", self.data_dir)

        scanner = Scanner(data_dir=self.data_dir)
        files = scanner.scan_directory()

        if not files:
            _logger.info("Periodic rescan: no new files to process.")
            return

        _logger.info("Periodic rescan: found %d file(s) to process.", len(files))

        for file in files:
            click.echo(f"Rescan: Processing {file.filepath}...")
            try:
                result = self.producer.process_file(file)
                if result is not None and result.feed_eligible:
                    click.echo(f"Rescan: Processed {file.filepath} successfully.")
                else:
                    click.echo(
                        f"Rescan: Processing {file.filepath} resulted in ineligible metadata."
                    )
            except Exception as exc:
                self._log_processing_error(
                    exc,
                    filepath=file.filepath,
                    stage="rescan",
                )

        counts = _generate_feeds(self.data_dir)
        _print_feed_summary(counts)
        _logger.info("Periodic rescan complete. Feeds regenerated.")


@app.command()
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def watch(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Watch directories for changes (incremental mode).

    Starts a file-system watcher that processes new and modified files
    in real-time, regenerating feeds after each file is processed.
    Press ``Ctrl+C`` to stop.

    When ``--force`` is specified, all files are processed on startup
    even if already feed-eligible.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting watch command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    if not data_dir_path.exists():
        _logger.critical("Data directory does not exist: %s", data_dir_path)
        click.echo(
            f"Error: Data directory does not exist: {data_dir_path}",
            err=True,
        )
        sys.exit(1)

    # --- Initial scan --------------------------------------------------
    if force:
        # Force processing: expand all source files regardless of status.
        files = _expand_paths_to_files(resolved_paths)
    else:
        scanner = Scanner(data_dir=data_dir_path)
        scanned = scanner.scan_directory()
        # Adjust filepath to be relative to CWD instead of data_dir.
        cwd = Path.cwd()
        files = [
            ScannedFile(
                filepath=os.path.relpath(Path(f.absolute_path), cwd),
                absolute_path=f.absolute_path,
                filename=f.filename,
                filetype=f.filetype,
                current_stage=f.current_stage,
                json_exists=f.json_exists,
            )
            for f in scanned
        ]

    from focus_feed.producer import Producer

    if files:
        click.echo(f"Initial scan: found {len(files)} file(s) to process.")
        producer = Producer(force=force)

        for file in files:
            click.echo(f"Processing {file.filepath}...")
            try:
                producer.process_file(file)
            except Exception as exc:
                log_error(exc, filepath=file.filepath, stage="processing")

        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)
    else:
        click.echo("Initial scan: no files to process.")
        counts = _generate_feeds(data_dir_path)
        _print_feed_summary(counts)

    # --- Start watching ------------------------------------------------
    producer = Producer()
    event_handler = _WatchHandler(data_dir=data_dir_path, producer=producer)

    from watchdog.observers import Observer

    observer = Observer()
    observer.schedule(event_handler, str(data_dir_path), recursive=True)
    observer.start()
    _logger.info("Watching %s for changes", str(data_dir_path))

    # --- Start periodic rescan thread (T6.3) ----------------------------
    # Fallback for file systems (NFS, network drives, etc.) that may not
    # reliably trigger watchdog events.
    rescan_interval = int(os.environ.get("RESCAN_INTERVAL", event_handler._rescan_interval))
    stop_event = threading.Event()

    def _rescan_loop() -> None:
        """Background thread loop for periodic rescan."""
        while not stop_event.wait(timeout=rescan_interval):
            try:
                event_handler._periodic_rescan()
            except Exception as exc:
                log_error(exc, stage="periodic_rescan")

    if rescan_interval > 0:
        rescan_thread = threading.Thread(target=_rescan_loop, daemon=True, name="periodic-rescan")
        rescan_thread.start()
        click.echo(f"Periodic rescan enabled (every {rescan_interval}s).")
    else:
        rescan_thread = None
        click.echo("Periodic rescan disabled.")

    click.echo(f"\nWatching {data_dir_path} for changes... (Ctrl+C to stop)")

    # --- Graceful shutdown --------------------------------------------
    shutdown_event = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_event
        shutdown_event = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not shutdown_event:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        # Gracefully shut down the rescan thread (T6.3).
        stop_event.set()
        if rescan_thread is not None:
            rescan_thread.join(timeout=5)
            if rescan_thread.is_alive():
                _logger.warning("Rescan thread did not shut down within timeout.")
        click.echo("\nWatcher stopped.")
        _logger.info("Watcher stopped")


# ---------------------------------------------------------------------------
# Feeds command  (T5.4)
# ---------------------------------------------------------------------------


@app.command("feeds")
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def feeds(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Generate RSS feeds from existing JSON metadata.

    Before generating feeds, finalizes any incomplete JSON metadata files
    (i.e., files with step statuses that are not yet ``"complete"``) by
    filling in missing required fields, setting applicable step statuses
    to ``"complete"``, and adding timestamps. This makes files
    feed-eligible so they appear in the generated feeds.

    When ``--force`` is specified, all files are processed even if already
    feed-eligible.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting feeds command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    # Finalize incomplete JSON metadata before generating feeds.
    _finalize_incomplete_json(data_dir_path, paths=resolved_paths, force=force)

    # Warn if no feed.json is found anywhere in the data directory.
    feed_json_files = list(data_dir_path.glob("**/feed.json"))
    if not feed_json_files:
        _logger.warning("No feed.json found at %s; feeds not generated", str(data_dir_path))

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
                if metadata.filetype == "mp4":
                    podcast_count += 1
                elif metadata.filetype in ("pdf", "html"):
                    article_count += 1
        except Exception:
            pass

    click.echo("Feeds generated successfully.")
    click.echo(f"  Podcast feed: {podcast_count} item(s)")
    click.echo(f"  Articles feed: {article_count} item(s)")
    _logger.info("Feeds generated: %d podcast, %d articles", podcast_count, article_count)


# ---------------------------------------------------------------------------
# Validate command  (T5.5)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@click.argument("paths", nargs=-1)
def validate(
    config_file: str | None,
    force: bool,
    paths: tuple[str, ...],
) -> None:
    """Validate all JSON metadata files in the given paths.

    Before validation, finalizes any incomplete JSON metadata files
    (i.e., files with step statuses that are not yet ``"complete"``) by
    filling in missing required fields, setting applicable step statuses
    to ``"complete"``, and adding timestamps. This makes files
    feed-eligible.

    When ``--force`` is specified, all files are processed even if already
    feed-eligible.

    Then scans for all JSON metadata files, validates each against the
    schema, checks for orphaned JSON files, and checks for special
    characters in filenames.  Exits with code 0 when all files are valid,
    1 when errors are found.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting validate command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    # Finalize incomplete JSON metadata before validation.
    _finalize_incomplete_json(data_dir_path, paths=resolved_paths, force=force)

    errors: list[str] = []
    warnings: list[str] = []

    # --- Find all JSON files -------------------------------------------
    json_files = _find_json_files(data_dir_path)

    # --- Validate each JSON file against the schema --------------------
    for json_path in json_files:
        if not json_path.exists():
            continue
        try:
            Metadata.from_json_file(json_path)
        except Exception as exc:
            log_error(exc, filepath=str(json_path), stage="validation")
            errors.append(f"{json_path}: Schema validation failed - {exc}")

    # --- Check for orphaned JSON files ---------------------------------
    for json_path in json_files:
        if not json_path.exists():
            continue
        if _is_orphaned_json(json_path):
            warnings.append(f"Orphaned JSON file (no matching source): {json_path}")

    # --- Check for special characters in filenames ---------------------
    for root, _dirs, files in os.walk(data_dir_path):
        root_path = Path(root)
        for filename in files:
            if filename.lower() in _IGNORE_FILENAMES:
                continue
            source_path = root_path / filename
            if source_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                warnings.extend(_check_special_characters(source_path))

    # --- Output report -------------------------------------------------
    click.echo("Validation Report")
    click.echo("=" * 50)
    click.echo(f"Checked {len(json_files)} JSON metadata file(s).")

    if errors:
        click.echo(f"\nErrors ({len(errors)}):")
        for error in errors:
            click.echo(f"  ERROR: {error}")

    if warnings:
        click.echo(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            click.echo(f"  WARNING: {warning}")

    if not errors and not warnings:
        click.echo("\nAll files are valid. No errors or warnings found.")

    _logger.info("Validation complete: %d errors, %d warnings", len(errors), len(warnings))

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: transcribe  (T5.7)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def transcribe(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Transcribe MP4 files to markdown text.

    Scans the given paths for MP4 files that do not yet have a complete
    transcript, transcribes each one using the remote transcription API,
    and writes the transcript to the adjacent JSON metadata file.

    Files that already have ``transcribe.status="complete"`` are skipped
    unless ``--force`` is specified.  Use ``--dry-run`` to preview which
    files would be processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting transcribe command")

    from focus_feed.config import get_config
    from focus_feed.services.transcriber import Transcriber

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_transcription(data_dir_path, force=force, paths=resolved_paths)

    if not files:
        click.echo("No MP4 files need transcription.")
        return

    click.echo(f"Found {len(files)} MP4 file(s) to transcribe.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be transcribed", len(files))
        click.echo("[Dry run] Files that would be transcribed:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    transcriber = Transcriber()

    config = get_config()

    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Transcribing {relative_path}...")
        try:
            # Set transcribe step to in_progress and write before transcription begins.
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            now = datetime.now(UTC)
            # Initialize or update transcribe step map
            transcribe_step = metadata.get("transcribe") or {}
            transcribe_step["status"] = "in_progress"
            transcribe_step["start_time"] = int(now.timestamp())
            transcribe_step["input_size"] = source_file.stat().st_size
            metadata["transcribe"] = transcribe_step
            metadata.update(
                {
                    "filename": source_file.name,
                    "filetype": "mp4",
                    "mtime": source_file.stat().st_mtime,
                }
            )
            _write_json_metadata(metadata, source_file)

            transcript = transcriber.transcribe(str(source_file))

            # Update transcribe step to complete after successful transcription.
            transcribe_step["status"] = "complete"
            transcribe_step["end_time"] = int(now.timestamp())
            transcribe_step["output_size"] = len(transcript.encode("utf-8")) if transcript else 0
            transcribe_step["words"] = len(transcript.split()) if transcript else 0
            transcribe_step["model"] = config.transcription_remote_model or "remote"
            metadata["transcribe"] = transcribe_step
            metadata["content"] = transcript
            _write_json_metadata(metadata, source_file)

            success_count += 1
            word_count = len(transcript.split())
            click.echo(f"  Transcribed {word_count} words.")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="transcription")

    click.echo(f"\nTranscription complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Transcribe complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: summarize  (T5.8)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def summarize(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Generate summaries from existing content.

    Scans the given paths for MP4, PDF, and HTML files that have content
    (transcript for MP4, extracted text for PDF/HTML) but no summary,
    generates a key-takeaways summary via the LLM, and writes the summary
    to the adjacent JSON metadata file.

    Files with ``summarize.status="complete"`` are skipped unless
    ``--force`` is specified.  Use ``--dry-run`` to preview which files
    would be processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting summarize command")

    from focus_feed.config import get_config
    from focus_feed.services.summarizer import Summarizer

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_summary(data_dir_path, force=force, paths=resolved_paths)

    if not files:
        click.echo("No files need summarization.")
        return

    click.echo(f"Found {len(files)} file(s) to summarize.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be summarized", len(files))
        click.echo("[Dry run] Files that would be summarized:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Summarizing {relative_path}...")
        reflection_questions = None
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path)
            if metadata is None:
                click.echo(f"  WARNING: No JSON metadata found for {relative_path}; skipping.")
                error_count += 1
                continue

            content = metadata.get("content") or ""
            if not content.strip():
                click.echo(f"  WARNING: No content found for {relative_path}; skipping.")
                error_count += 1
                continue

            # Set summarize step to in_progress and write before summarization begins.
            now = datetime.now(UTC)
            config = get_config()
            summarize_step = metadata.get("summarize") or {}
            summarize_step["status"] = "in_progress"
            summarize_step["start_time"] = int(now.timestamp())
            summarize_step["input_size"] = len(content.encode("utf-8")) if content else 0
            metadata["summarize"] = summarize_step
            _write_json_metadata(metadata, source_file)

            # Check for reflection.md in the source file's directory.
            reflection_questions = _check_reflection_md(str(source_file))
            summary, prompt_tokens, completion_tokens = summarizer.summarize(
                content,
                reflection_questions=reflection_questions,
                filepath=str(relative_path),
                filetype=metadata.get("filetype"),
            )

            # Update summarize step to complete after successful summarization.
            summarize_step["status"] = "complete"
            summarize_step["end_time"] = int(now.timestamp())
            summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
            summarize_step["words"] = len(summary.split()) if summary else 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = prompt_tokens
            summarize_step["output_tokens"] = completion_tokens
            metadata["summarize"] = summarize_step
            metadata["summary"] = summary
            # For MP4 files, leave description blank and use summary instead.
            # For PDF/HTML files, keep the existing description from extraction.
            if metadata.get("filetype") == "mp4":
                metadata["description"] = ""
            _write_json_metadata(metadata, source_file)

            success_count += 1
            click.echo(f"  Tokens in: {prompt_tokens}, Tokens out: {completion_tokens}")
            word_count = len(summary.split())
            click.echo(f"  Summarized {word_count} words.")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="summarization")

    click.echo(f"\nSummarization complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Summarize complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: extract  (T5.9)
# ---------------------------------------------------------------------------


@app.command("extract")
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def extract(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Extract from PDF/HTML files.

    Scans the given paths for PDF and HTML files that do not yet have
    a complete metadata extraction, extracts metadata (title, description,
    date, size) using pypdf or BeautifulSoup, and writes the extracted data
    to the adjacent JSON metadata file.

    Files with ``extract.status="complete"`` are skipped unless ``--force``
    is specified.  Use ``--dry-run`` to preview which files would be
    processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting extract command")

    from focus_feed.services.extractors import HTMLExtractor, PDFExtractor

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_metadata_extraction(
        data_dir_path, force=force, paths=resolved_paths
    )

    if not files:
        click.echo("No PDF/HTML files need metadata extraction.")
        return

    click.echo(f"Found {len(files)} PDF/HTML file(s) to extract metadata from.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be processed", len(files))
        click.echo("[Dry run] Files that would be processed:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    pdf_extractor = PDFExtractor()
    html_extractor = HTMLExtractor()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Extracting metadata from {relative_path}...")
        try:
            filetype = source_file.suffix.lower().lstrip(".")
            if filetype == "pdf":
                extracted = pdf_extractor.extract(str(source_file))
            elif filetype == "html":
                extracted = html_extractor.extract(str(source_file))
            else:
                click.echo(f"  WARNING: Unsupported file type '{filetype}'; skipping.")
                error_count += 1
                continue

            # Set extract step to in_progress and write before extraction completes.
            json_path = source_file.with_name(source_file.name + ".json")
            metadata = _get_json_metadata(json_path) or {}
            now = datetime.now(UTC)
            title = extracted.get("title") or source_file.stem
            if not title.strip():
                title = source_file.stem
            # Preserve existing title dict if present; otherwise use the
            # extracted title as a plain string (will be finalized later).
            if "title" not in metadata or not isinstance(metadata.get("title"), dict):
                metadata["title"] = title
            # Initialize or update extract step map
            extract_step = metadata.get("extract") or {}
            extract_step["status"] = "in_progress"
            extract_step["start_time"] = int(now.timestamp())
            extract_step["input_size"] = source_file.stat().st_size
            metadata["extract"] = extract_step
            metadata.update(
                {
                    "filename": source_file.name,
                    "filetype": filetype,
                    "mtime": source_file.stat().st_mtime,
                    "description": extracted.get("description") or "",
                    "date": extracted.get("date", 0),
                    "publication_date": extracted.get("publication_date"),
                    "size": extracted.get("size", 0),
                }
            )
            _write_json_metadata(metadata, source_file)

            # Update extract step to complete after successful extraction.
            extract_step["status"] = "complete"
            extract_step["end_time"] = int(now.timestamp())
            content = extracted.get("content")
            extract_step["output_size"] = len(content.encode("utf-8")) if content else 0
            extract_step["words"] = len(content.split()) if content else 0
            metadata["extract"] = extract_step
            metadata["content"] = content
            _write_json_metadata(metadata, source_file)

            success_count += 1
            click.echo(f"  Title: {title}")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="metadata_extraction")

    click.echo(f"\nMetadata extraction complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Extract complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Granular subcommand: title  (T5.10)
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@_force_option
@_dry_run_option
@click.argument("paths", nargs=-1)
def title(
    config_file: str | None,
    force: bool,
    dry_run: bool,
    paths: tuple[str, ...],
) -> None:
    """Generate titles for files that lack proper titles.

    Scans for JSON metadata files where the title is missing, empty, or is
    just the filename stem (with underscores replaced by spaces), and
    generates a descriptive title using the LLM.

    Files that already have a proper title are skipped unless ``--force``
    is specified.  Use ``--dry-run`` to preview which files would be
    processed without making any changes.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting title command")

    from focus_feed.config import get_config
    from focus_feed.services.summarizer import Summarizer

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    files = _find_files_needing_title(data_dir_path, force=force, paths=resolved_paths)

    if not files:
        click.echo("No files need title generation.")
        return

    click.echo(f"Found {len(files)} file(s) to generate titles for.")

    if dry_run:
        _logger.warning("Dry-run mode active; no files will be modified")
        _logger.info("Dry run: %d files would be processed", len(files))
        click.echo("[Dry run] Files that would be processed:")
        for f in files:
            relative = os.path.relpath(f, Path.cwd())
            click.echo(f"  - {relative}")
        return

    summarizer = Summarizer()
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(files, 1):
        relative_path = os.path.relpath(source_file, Path.cwd())
        click.echo(f"[{i}/{len(files)}] Generating title for {relative_path}...")
        try:
            json_path = source_file.with_name(source_file.name + ".json")
            if not json_path.exists():
                click.echo(f"  WARNING: No JSON metadata found for {source_file.name}; skipping.")
                error_count += 1
                continue

            metadata_dict = json.loads(json_path.read_text(encoding="utf-8"))

            # Check if title needs generation (redundant with _find_files_needing_title
            # but needed for the force case).
            title = metadata_dict.get("title", "")
            # Normalize string title to dict for backward compatibility.
            if isinstance(title, str):
                title = {"filename": title, "extracted": None, "generated": None}
            has_generated = isinstance(title, dict) and bool((title.get("generated") or "").strip())
            if has_generated and not force:
                click.echo("  Already has a proper title; skipping.")
                continue

            # Generate title from content.
            content = _get_content_for_title(source_file, metadata_dict)
            if not content.strip():
                click.echo("  WARNING: No content available for title generation; skipping.")
                error_count += 1
                continue

            # Get filename and extracted title from the nested title metadata.
            filename_title = _clean_title(source_file.stem)
            extracted_title = title.get("extracted") if isinstance(title, dict) else None

            # Initialize the generate_title step map before the LLM call.
            now = datetime.now(UTC)
            config = get_config()
            generate_title_step = metadata_dict.get("generate_title") or {}
            generate_title_step["status"] = "in_progress"
            generate_title_step["start_time"] = int(now.timestamp() * 1000) / 1000.0
            generate_title_step["input_size"] = len(content.encode("utf-8")) if content else 0
            generate_title_step["output_size"] = 0
            generate_title_step["words"] = 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = 0
            generate_title_step["output_tokens"] = 0
            metadata_dict["generate_title"] = generate_title_step
            _write_json_metadata(metadata_dict, source_file)

            new_title, prompt_tokens, completion_tokens = summarizer.generate_title(
                content,
                filename=filename_title,
                extracted_title=extracted_title,
            )
            new_title = _clean_title(new_title) or new_title

            # Update the generate_title step map with completion info.
            generate_title_end = datetime.now(UTC)
            generate_title_step["status"] = "complete"
            generate_title_step["end_time"] = int(generate_title_end.timestamp() * 1000) / 1000.0
            generate_title_step["output_size"] = len(new_title.encode("utf-8")) if new_title else 0
            generate_title_step["words"] = len(new_title.split()) if new_title else 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = prompt_tokens
            generate_title_step["output_tokens"] = completion_tokens
            metadata_dict["generate_title"] = generate_title_step

            # Update metadata title dict with generated title.
            if isinstance(title, dict):
                title["generated"] = new_title
            else:
                title = {
                    "filename": filename_title,
                    "extracted": extracted_title,
                    "generated": new_title,
                }
            metadata_dict["title"] = title

            _write_json_metadata(metadata_dict, source_file)

            success_count += 1
            click.echo(f"  Title: {new_title}")
            click.echo(f"  Tokens in: {prompt_tokens}, Tokens out: {completion_tokens}")
        except Exception as exc:
            error_count += 1
            log_error(exc, filepath=str(relative_path), stage="title_generation")

    click.echo(f"\nTitle generation complete: {success_count} succeeded, {error_count} failed.")
    _logger.info("Title complete: %d succeeded, %d failed", success_count, error_count)

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# JSON finalization utility
# ---------------------------------------------------------------------------


def _finalize_incomplete_json(
    data_dir_path: Path,
    paths: list[Path] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Finalize incomplete JSON metadata files in the data directory.

    Finds source files (MP4, PDF, HTML) that have incomplete JSON metadata
    (i.e., not all applicable step maps have ``status == "complete"``) and
    assembles the final metadata by:

    - Filling in all required fields, using existing values from intermediate
      JSON, file system statistics, or LLM-generated titles and summaries as
      fallbacks.
    - Setting all applicable step maps to ``"complete"`` status.
    - Preserving existing ``content`` (transcript for MP4, extracted text for
      PDF/HTML) fields.
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
            f_resolved = f.resolve()
            for p in paths:
                if str(f_resolved).startswith(str(p)) or f_resolved == p:
                    files.append(f)
                    break
    else:
        files = _find_files_needing_json_generation(data_dir_path, force=force)

    if not files:
        all_source_files = _find_source_files(data_dir_path, ("mp4", "pdf", "html"))
        if not all_source_files:
            _logger.warning(
                "No supported source files found in %s for JSON generation. "
                "Expected .mp4, .pdf, or .html files.",
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
            metadata.pop("filepath", None)
            metadata.pop("extracted_title", None)

            # Skip files with errors — they should not be finalized.
            if metadata.get("error"):
                continue

            filetype = source_file.suffix.lower().lstrip(".")
            stat_result = source_file.stat()
            file_size = stat_result.st_size
            file_mtime = stat_result.st_mtime
            now = datetime.now(UTC)
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
            extracted_title = title.get("extracted")

            # Generate title via LLM if not already present.
            title_generated_tokens: tuple[int, int] | None = None
            generate_title_content: str | None = None
            if not title.get("generated") or not title["generated"].strip():
                generated = filename_title
                try:
                    if filetype == "mp4":
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
                                extracted_title=extracted_title,
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
            if filetype in ("pdf", "html") and not summary:
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
                "title": title,
                "description": description,
                "filename": source_file.name,
                "filetype": filetype,
                "date": metadata.get("date") or file_mtime,
                "mtime": stat_result.st_mtime,
                "size": metadata.get("size") or file_size,
            }

            # Set step maps to complete and include type-specific fields.
            now_ts = int(now.timestamp())
            if filetype == "mp4":
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
            elif filetype in ("pdf", "html"):
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


# ---------------------------------------------------------------------------
# Create-feed command
# ---------------------------------------------------------------------------


# Template for the feed.json file created by the ``create-feed`` subcommand.
# The ``link`` field is intentionally omitted: feed links are derived from
# ``BASE_URL`` + relative path at generation time (see feeds.py).
_FEED_JSON_TEMPLATE: dict[str, Any] = {
    "title": "Feed Title",
    "description": "Feed description",
    "language": "en-us",
    "author": "Feed Author",
    "image": {
        "url": "https://www.example.com/logo.jpg",
        "title": "Logo",
        "link": "https://www.example.com/",
    },
}


@app.command("create-feed")
@_config_file_option
@click.argument(
    "directory",
    type=click.Path(file_okay=False, dir_okay=True),
)
def create_feed(config_file: str | None, directory: str) -> None:
    """Create a template ``feed.json`` file in a target directory.

    Creates a ``feed.json`` file with default template fields (title,
    description, language, author, image) in the specified directory.
    The directory is created if it does not exist. If a ``feed.json``
    already exists, it is overwritten.
    """
    _logger.info("Starting create-feed command")
    target_dir = Path(directory)
    feed_json_path = target_dir / "feed.json"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        feed_json_path.write_text(json.dumps(_FEED_JSON_TEMPLATE, indent=2), encoding="utf-8")
        _logger.info("Created feed.json at %s", str(feed_json_path))
    except (OSError, PermissionError) as exc:
        log_error(
            exc,
            filepath=str(feed_json_path),
            stage="feed_json_creation",
        )
        raise click.ClickException(f"Failed to create feed.json: {exc}") from exc

    click.echo(f"Created feed.json at {feed_json_path}")


# ---------------------------------------------------------------------------
# Cleanup command
# ---------------------------------------------------------------------------


@app.command()
@_config_file_option
@click.argument("paths", nargs=-1)
def cleanup(config_file: str | None, paths: tuple[str, ...]) -> None:
    """Remove unknown keys from JSON metadata files.

    Scans the given paths for ``.json`` metadata files (excluding
    ``feed.json`` and other ignored filenames) and removes any keys
    that are not part of the Metadata schema:

    - Unknown top-level keys (not in ``Metadata.model_fields``).
    - Unknown keys inside the ``title`` dict (``filename``, ``extracted``,
      ``generated`` are allowed; a plain-string title is left as-is).
    - Unknown keys inside step-map dicts (``transcribe``, ``summarize``,
      ``extract``, ``generate_title``) — only ``StepMap.model_fields``
      keys are kept.

    Files are rewritten in-place only when unknown keys were found.
    A summary of the number of files cleaned and keys removed is printed.

    At least one file or directory path is required.
    """
    _load_config(config_file)
    _logger.info("Starting cleanup command")

    resolved_paths = _resolve_paths(paths)
    data_dir_path = _resolve_data_dir_from_paths(resolved_paths)

    json_files = _find_json_files(data_dir_path)

    if not json_files:
        click.echo("No JSON metadata files found.")
        return

    files_checked = 0
    files_modified = 0
    total_keys_removed = 0

    for json_path in json_files:
        files_checked += 1
        try:
            keys_removed, unknown_key_paths = _cleanup_json_file(json_path)
        except OSError:
            # Error already logged by _cleanup_json_file.
            continue

        if keys_removed > 0:
            files_modified += 1
            total_keys_removed += keys_removed
            relative = os.path.relpath(json_path, Path.cwd())
            click.echo(f"  {relative}: removed {keys_removed} unknown key(s):")
            for key_path in unknown_key_paths:
                click.echo(f"    - {key_path}")
            _logger.info(
                "Cleaned %s: removed %d unknown key(s) %s",
                json_path,
                keys_removed,
                unknown_key_paths,
            )

    click.echo(
        f"\nCleanup complete: checked {files_checked} file(s), "
        f"modified {files_modified} file(s), removed {total_keys_removed} key(s)."
    )
    _logger.info(
        "Cleanup complete: %d files checked, %d modified, %d keys removed",
        files_checked,
        files_modified,
        total_keys_removed,
    )


# ---------------------------------------------------------------------------
# Entry point  (T5.6)
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``focus-feed`` CLI."""
    app()


if __name__ == "__main__":
    main()
