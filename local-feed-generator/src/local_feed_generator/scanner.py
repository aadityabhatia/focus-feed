"""File scanner for local-feed-generator.

Discovers content files (MP4, PDF, HTML) in the data directory and determines
which ones need processing based on the presence, status, and freshness of
their adjacent JSON metadata files.
Scanning logic (design doc section 5.2, tasks.md T2.3):
- Recursively walk ``DATA_DIR`` for files with extensions ``.mp4``, ``.pdf``,
  ``.html`` (case-insensitive).
- For each source file, check whether a corresponding ``.json`` metadata file
  exists in the same directory (named ``<source_filename>.json``).
- A source file needs processing when:
    * No matching ``.json`` file exists, **or**
    * The ``.json`` file exists but is older than the source file
      (source was modified after metadata was written), **or**
    * The ``.json`` file has step maps that indicate incomplete
      processing (e.g. ``transcribe.status`` is not ``"complete"`` for
      MP4 files, or ``extract.status`` is not ``"complete"`` for
      PDF/HTML).
- A source file is **skipped** (not returned) when:
    * The ``.json`` file exists, is newer than or equal to the source file,
      **and** all applicable step maps are ``"complete"`` (i.e. the file
      is feed-eligible).
- Files with an ``error`` field are reprocessed only when the source file is
  newer than the JSON (i.e., the user has modified the source to fix the
  issue). If the source has not changed, the error state is considered still
  relevant and the file is skipped. To force a retry without modifying the
  source, delete the ``.json`` file manually (see design.md section 13.3).
- Filenames containing special characters produce a WARNING but are not modified.
- Orphaned JSON files (a ``.json`` with no corresponding source file) are
  detected and logged as warnings. They are **not** included in the result set.

The scanner only reads the filesystem; it never writes or deletes files.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from local_feed_generator.config import get_config

_logger = logging.getLogger("local_feed_generator")


# Source file extensions supported by the scanner (case-insensitive).
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp4", ".pdf", ".html")

# Characters that are problematic in filenames for XML/URL contexts.
# These trigger a warning but do not block processing.
_SPECIAL_CHAR_PATTERN = re.compile(r'[<>&"\'\x00-\x1f]')

# Filenames that should be ignored entirely (not treated as metadata or source).
# Comparison is case-insensitive (see ``scan_directory``).
_IGNORE_FILENAMES = frozenset(
    {"podcast.xml", "articles.xml", "podcast.atom", "articles.atom", "index.html", "feed.json"}
)

# JSON status values that indicate a step is complete.
_COMPLETED_STEP_STATUS: str = "complete"

# JSON status values that indicate a step is in progress.
_IN_PROGRESS_STEP_STATUS: str = "in_progress"

# JSON field that indicates a previous processing attempt failed.
_ERROR_FIELD: str = "error"

# File types that use transcribe and summarize step maps.
_MP4_FILETYPE: str = "mp4"

# File types that use extract step map.
_DOC_FILETYPES: frozenset[str] = frozenset({"pdf", "html"})


@dataclass(frozen=True)
class ScannedFile:
    """A content file discovered by the scanner.

    Attributes:
        filepath: Relative path from ``DATA_DIR`` root (e.g. ``"videos/lecture.mp4"``).
        absolute_path: Absolute path to the source file on disk.
        filename: The base filename including extension (e.g. ``"lecture.mp4"``).
        filetype: Lowercase extension without the leading dot (e.g. ``"mp4"``).
        current_stage: The processing stage from the JSON metadata's
            ``stage`` field (or ``current_stage`` as a legacy fallback),
            if present. Allows the producer to resume from the last
            completed stage. ``None`` when no JSON metadata exists or
            the field is absent.
        json_exists: Whether a ``.json`` metadata file exists for this source
            file. ``True`` if the JSON file is present (regardless of its
            content or validity), ``False`` otherwise.
    """

    filepath: str
    absolute_path: str
    filename: str
    filetype: str
    current_stage: str | None = None
    json_exists: bool = False


def _is_supported_source_file(path: Path) -> bool:
    """Return True if *path* has a supported source extension (case-insensitive)."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _filetype_from_path(path: Path) -> str:
    """Return the lowercase filetype (extension without dot) for *path*."""
    return path.suffix.lower().lstrip(".")


def _validate_filename(filename: str, absolute_path: Path) -> None:
    """Log a warning if *filename* contains special characters.

    Per design doc section 7.2 and constraint C-006, filenames are never
    modified — this is a read-only warning to alert the user.
    """
    if not filename:
        _logger.warning("Empty filename encountered at %s", absolute_path)
        return

    # Leading/trailing whitespace.
    if filename != filename.strip():
        _logger.warning(
            "Filename '%s' has leading/trailing whitespace at %s",
            filename,
            absolute_path,
        )

    # Special characters: XML-significant chars, quotes, control chars.
    matches = _SPECIAL_CHAR_PATTERN.findall(filename)
    if matches:
        unique_chars = sorted(set(matches))
        _logger.warning(
            "Filename '%s' contains special characters %s at %s",
            filename,
            unique_chars,
            absolute_path,
        )


class Scanner:
    """Discovers content files in the data directory that need processing.

    The scanner walks ``DATA_DIR`` recursively, identifies source files
    (``.mp4``, ``.pdf``, ``.html``), and determines which ones lack up-to-date
    JSON metadata. It also detects orphaned JSON files and warns about
    filenames with special characters.

    Attributes:
        data_dir: Absolute path to the data directory.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the scanner with a data directory.

        Args:
            data_dir: Path to the data directory. When ``None``, the value is
                loaded from the application configuration (``DATA_DIR``).
        """
        if data_dir is None:
            config = get_config()
            data_dir = config.data_dir
        self.data_dir: Path = Path(data_dir).resolve()

    def scan_directory(self) -> list[ScannedFile]:
        """Recursively scan ``DATA_DIR`` and return files needing processing.

        A source file needs processing when:
        - Its adjacent ``.json`` metadata file is missing, **or**
        - The ``.json`` file exists but is older than the source file
          (source was modified after metadata was written), **or**
        - The ``.json`` file has step maps indicating incomplete
          processing (e.g. ``transcribe.status`` is not ``"complete"`` for
          MP4 files, or ``extract.status`` is not ``"complete"`` for
          PDF/HTML files).

        Files with all applicable step maps set to ``"complete"`` and a
        fresh JSON file are skipped. Files with an ``error`` field are
        reprocessed only when the source file is newer than the JSON (user
        may have fixed the source).

        When a file needs processing, its ``current_stage`` field (if present
        in the JSON metadata) is included in the returned :class:`ScannedFile`
        so the producer can resume from the last completed stage.

        During the scan the method also:
        - Logs warnings for filenames containing special characters.
        - Detects and logs warnings for orphaned JSON files (JSON without a
          corresponding source file).

        Returns:
            A list of :class:`ScannedFile` instances, one per source file that
            needs processing. The list is sorted by relative path for
            deterministic ordering.

        Raises:
            Nothing — filesystem errors on individual files or directories are
            caught, logged, and skipped so a single inaccessible path does not
            abort the entire scan.
        """
        if not self.data_dir.exists():
            _logger.warning("Data directory does not exist: %s", self.data_dir)
            return []

        if not self.data_dir.is_dir():
            _logger.warning("Data path is not a directory: %s", self.data_dir)
            return []

        files_to_process: list[ScannedFile] = []

        for root, _dirs, files in os.walk(self.data_dir):
            root_path = Path(root)
            for filename in files:
                # Skip generated feed files and known non-content artifacts.
                # Comparison is case-insensitive to handle INDEX.HTML etc.
                if filename.lower() in _IGNORE_FILENAMES:
                    continue

                source_path = root_path / filename

                # --- Orphaned JSON detection ---
                # A .json file is orphaned when stripping the ".json" suffix
                # leaves no source file with a supported extension.
                if filename.lower().endswith(".json"):
                    self._check_orphaned_json(source_path, root_path)
                    continue

                # --- Source file handling ---
                if not _is_supported_source_file(source_path):
                    continue

                _validate_filename(filename, source_path)

                needs_processing, current_stage, json_exists = self._evaluate_json_status(
                    source_path
                )

                if needs_processing:
                    relative_path = source_path.relative_to(self.data_dir)
                    files_to_process.append(
                        ScannedFile(
                            filepath=str(relative_path),
                            absolute_path=str(source_path),
                            filename=filename,
                            filetype=_filetype_from_path(source_path),
                            current_stage=current_stage,
                            json_exists=json_exists,
                        )
                    )

        files_to_process.sort(key=lambda f: f.filepath)
        return files_to_process

    def _read_json_metadata(self, json_path: Path) -> dict[str, Any] | None:
        """Read and parse a JSON metadata file.

        Args:
            json_path: Path to the ``.json`` metadata file.

        Returns:
            The parsed JSON as a dictionary, or ``None`` if the file cannot
            be read or parsed. A warning is logged on failure.
        """
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data: dict[str, Any] | None = json.load(f)
                return data
        except json.JSONDecodeError as exc:
            _logger.warning("Could not parse JSON metadata %s: %s", json_path, exc)
            return None
        except OSError as exc:
            _logger.warning("Could not read JSON metadata %s: %s", json_path, exc)
            return None

    def _is_source_newer(self, source_path: Path, json_path: Path) -> bool:
        """Return True if *source_path* was modified after *json_path*.

        Uses file modification times. If either file cannot be stat-ed, the
        source is assumed to be newer (i.e., processing is needed) to avoid
        silently skipping files due to stat errors.

        Args:
            source_path: Path to the source file.
            json_path: Path to the JSON metadata file.

        Returns:
            ``True`` if the source file's mtime is strictly greater than the
            JSON file's mtime, or if the mtime comparison could not be
            performed.
        """
        try:
            source_mtime = source_path.stat().st_mtime
            json_mtime = json_path.stat().st_mtime
        except OSError as exc:
            _logger.warning("Could not stat file %s: %s", source_path, exc)
            return True

        return source_mtime > json_mtime

    @staticmethod
    def _is_feed_eligible_dict(metadata: dict[str, Any], filetype: str) -> bool:
        """Check if a raw metadata dict indicates the file is feed-eligible.

        A file is feed-eligible when all applicable steps are
        ``"complete"`` and there is no error field.

        Checks the separate step maps (``transcribe``, ``summarize``,
        ``extract``) for step completion status.

        - **MP4**: ``transcribe`` step map exists and
          ``transcribe.status == "complete"`` and ``summarize`` step map
          exists and ``summarize.status == "complete"``.
        - **PDF/HTML**: ``extract`` step map exists and
          ``extract.status == "complete"``.

        Legacy fields (``transcribe_status``, ``summarize_status``,
        ``extract_status``) are also supported for backward compatibility.

        Args:
            metadata: The parsed JSON metadata dictionary.
            filetype: The source file type (``"mp4"``, ``"pdf"``, ``"html"``).

        Returns:
            ``True`` if the file is feed-eligible, ``False`` otherwise.
        """
        if metadata.get(_ERROR_FIELD):
            return False
        if filetype == _MP4_FILETYPE:
            # New schema: step maps
            transcribe = metadata.get("transcribe")
            summarize = metadata.get("summarize")
            transcribe_complete = (
                isinstance(transcribe, dict) and transcribe.get("status") == _COMPLETED_STEP_STATUS
            )
            summarize_complete = (
                isinstance(summarize, dict) and summarize.get("status") == _COMPLETED_STEP_STATUS
            )
            # Legacy schema support
            if not transcribe_complete:
                transcribe_complete = metadata.get("transcribe_status") == _COMPLETED_STEP_STATUS
            if not summarize_complete:
                summarize_complete = metadata.get("summarize_status") == _COMPLETED_STEP_STATUS
            return transcribe_complete and summarize_complete
        if filetype in _DOC_FILETYPES:
            # New schema
            extract = metadata.get("extract")
            extract_complete = (
                isinstance(extract, dict) and extract.get("status") == _COMPLETED_STEP_STATUS
            )
            # Legacy schema support
            if not extract_complete:
                extract_complete = metadata.get("extract_status") == _COMPLETED_STEP_STATUS
            return extract_complete
        return False

    def _evaluate_json_status(self, source_path: Path) -> tuple[bool, str | None, bool]:
        """Determine if *source_path* needs processing based on its JSON metadata.

        This method reads the adjacent ``.json`` metadata file (if it exists)
        and evaluates the step map status fields along with file freshness to
        decide whether the source file should be returned for processing.

        Decision logic:
        - **No JSON file**: Needs processing.
        - **JSON with all applicable steps complete and no error**: Skip,
          unless the source file is newer than the JSON (outdated metadata).
        - **JSON with any applicable step not complete**: Needs processing
          (resume from the last completed stage, if available).
        - **JSON with an error field**: Reprocessed only if the source
          file is newer than the JSON (user may have fixed the source). If
          the source is not newer, the error state is still relevant and
          the file is skipped.
        - **JSON with unknown or missing step statuses**: Falls back to the
          freshness check — if the source is newer, needs processing;
          otherwise, skipped conservatively.

        Args:
            source_path: Path to the source file (e.g. ``"lecture.mp4"``).

        Returns:
            A tuple of ``(needs_processing, current_stage, json_exists)``:
            - ``needs_processing``: Whether the file should be returned for
              processing.
            - ``current_stage``: The ``stage`` value from the JSON
              metadata, if present and a string. ``None`` otherwise.
            - ``json_exists``: Whether the ``.json`` file exists.
        """
        json_path = source_path.with_name(source_path.name + ".json")

        if not json_path.exists():
            return True, None, False

        metadata = self._read_json_metadata(json_path)

        # Extract current_stage from metadata if available.
        current_stage: str | None = None
        if metadata is not None and isinstance(metadata, dict):
            stage = metadata.get("stage") or metadata.get("current_stage")
            if isinstance(stage, str):
                current_stage = stage

        # Check freshness: if source is newer, reprocess regardless of status.
        source_newer = self._is_source_newer(source_path, json_path)

        if source_newer:
            return True, current_stage, True

        # Source is not newer than JSON.
        # If JSON can't be parsed, skip conservatively (don't reprocess
        # unreadable metadata unless source is newer).
        if metadata is None:
            return False, None, True

        # Check for error state — skip unless source was modified after
        # the error (already handled by the freshness check above).
        if metadata.get(_ERROR_FIELD):
            return False, current_stage, True

        # Check step statuses to determine if the file is complete.
        filetype = source_path.suffix.lower().lstrip(".")
        if self._is_feed_eligible_dict(metadata, filetype):
            return False, current_stage, True

        # Any step not complete → needs processing.
        return True, current_stage, True

    def _check_orphaned_json(self, json_path: Path, root_path: Path) -> None:
        """Check if *json_path* is orphaned (no corresponding source file).

        A JSON file is orphaned when removing the ``.json`` suffix does not
        yield an existing file with a supported source extension. Such files
        are logged as warnings and never included in the processing list.
        """
        source_filename = json_path.name[: -len(".json")]
        source_path = root_path / source_filename

        # The stripped name must have a supported extension to be a real source.
        if not _is_supported_source_file(source_path):
            _logger.warning(
                "Orphaned JSON file (no matching source): %s",
                json_path,
            )
            return

        if not source_path.exists():
            _logger.warning(
                "Orphaned JSON file (source missing): %s",
                json_path,
            )
