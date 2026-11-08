"""Orchestration and base producer pipeline."""

# Methods resolve attributes provided by the concrete Producer mixin class.
# mypy sees this base class in isolation, before the final MRO is composed.
# mypy: disable-error-code="attr-defined,no-any-return"

from collections.abc import Iterator
from contextlib import contextmanager, suppress
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

from focus_feed.errors import create_error_metadata, log_error
from focus_feed.filetypes import (
    DOCLING_FILETYPES,
    DOCUMENT_FILETYPES,
    MEDIA_FILETYPES,
    SUPPORTED_FILETYPES,
    DoclingFiletype,
)
from focus_feed.metadata import Metadata, write_metadata_json
from focus_feed.scanner import ScannedFile, Scanner
from focus_feed.services.extractors import (
    DoclingExtractor,
    HTMLExtractor,
    PDFExtractor,
)
from focus_feed.services.summarizer import Summarizer
from focus_feed.services.transcriber import Transcriber
from focus_feed.utils import clean_title, is_audio_filetype, is_video_filetype, now_timestamp

_clean_title = clean_title
_logger = logging.getLogger("focus_feed")


def get_config() -> Any:
    """Resolve configuration through the legacy producer module namespace."""
    import focus_feed.producer

    return focus_feed.producer.get_config()


def get_audio_metadata(*args: Any, **kwargs: Any) -> Any:
    """Resolve media metadata extraction through the legacy module namespace."""
    import focus_feed.producer

    return focus_feed.producer.get_audio_metadata(*args, **kwargs)


@contextmanager
def _tag_step(step: str) -> Iterator[None]:
    """Tag exceptions with the pipeline step that raised them."""
    try:
        yield
    except Exception as exc:
        cast(Any, exc).focus_feed_step = step
        raise


def _now_timestamp() -> float:
    """Return the current UNIX epoch timestamp with millisecond precision."""
    return now_timestamp()


def _check_reflection_md(absolute_path: str) -> str | None:
    """Check if a ``reflection.md`` file exists next to the source file.

    Args:
        absolute_path: Absolute path to the source file.

    Returns:
        The contents of ``reflection.md`` if it exists, otherwise ``None``.
    """
    source_dir = Path(absolute_path).parent
    reflection_path = source_dir / "reflection.md"
    if reflection_path.exists():
        try:
            return reflection_path.read_text(encoding="utf-8")
        except OSError as e:
            _logger.warning("Failed to read reflection.md at %s: %s", reflection_path, e)
    return None


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename for use as a title.

    Removes the file extension and replaces all underscores with spaces.

    Example:
        ``my_lecture.mp4`` -> ``my lecture``

    Args:
        filename: The original filename with extension.

    Returns:
        The sanitized filename (no extension, no underscores).
    """
    return _clean_title(Path(filename).stem) or ""


def _compute_file_hash(filepath: str) -> str:
    """Compute the SHA256 hash of a file.

    Reads the file in binary mode and returns the full hex digest (64
    characters).  Used to populate the ``hash`` field in JSON metadata
    for guid generation in feeds.

    Args:
        filepath: Absolute path to the source file.

    Returns:
        The SHA256 hex digest string (64 characters), or an empty string
        if the file cannot be read.
    """
    try:
        _logger.debug("Computing hash for %s", filepath)
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError as e:
        _logger.warning("Failed to compute hash for %s: %s", filepath, e)
        return ""


class ProducerBase:
    """Orchestrates media and document processing and metadata generation.

    The Producer accepts an optional :class:`~focus_feed.scanner.Scanner`
    and optional service instances. When services are not provided, singleton
    instances are created lazily so that the configured remote transcription
    client is initialized only when needed and reused.

    When a scanner is provided, :meth:`process_all` can be used to discover and
    process files in batch. Without a scanner, individual files can still be
    processed via :meth:`process_file` using a :class:`ScannedFile` instance.

    Attributes:
        scanner: The optional :class:`Scanner` used for batch discovery.
            May be ``None`` when only :meth:`process_file` is used.
        transcriber: The :class:`Transcriber` for media audio transcription.
        summarizer: The :class:`Summarizer` for LLM-based summarization
            and title generation.
        pdf_extractor: The :class:`PDFExtractor` for PDF metadata.
        html_extractor: The :class:`HTMLExtractor` for HTML metadata.
        docling_extractor: The :class:`DoclingExtractor` for optional document
            extraction before PDF fallback.
    """

    def __init__(
        self,
        scanner: Scanner | None = None,
        transcriber: Transcriber | None = None,
        summarizer: Summarizer | None = None,
        pdf_extractor: PDFExtractor | None = None,
        html_extractor: HTMLExtractor | None = None,
        force: bool = False,
        docling_extractor: DoclingExtractor | None = None,
    ) -> None:
        """Initialize the Producer with dependencies.

        Args:
            scanner: Optional pre-configured :class:`Scanner` instance used
                for batch file discovery via :meth:`process_all`. When
                ``None``, :meth:`process_all` will raise an error.
            transcriber: Optional :class:`Transcriber` instance. When
                ``None``, a new instance is created (lazy client initialization).
            summarizer: Optional :class:`Summarizer` instance. When
                ``None``, a new instance is created.
            pdf_extractor: Optional :class:`PDFExtractor` instance. When
                ``None``, a new instance is created.
            html_extractor: Optional :class:`HTMLExtractor` instance. When
                ``None``, a new instance is created.
            docling_extractor: Optional :class:`DoclingExtractor` instance. When
                ``None``, a new instance is created.
            force: When ``True``, all processing steps are re-run even if
                existing step maps show ``"complete"`` status. This is used
                by the ``process --force`` CLI command to force full
                re-processing of files.
        """
        self.scanner: Scanner | None = scanner
        self.transcriber: Transcriber = transcriber or Transcriber()
        self.summarizer: Summarizer = summarizer or Summarizer()
        self.pdf_extractor: PDFExtractor = pdf_extractor or PDFExtractor()
        self.html_extractor: HTMLExtractor = html_extractor or HTMLExtractor()
        self.docling_extractor: DoclingExtractor = docling_extractor or DoclingExtractor()
        self.force: bool = force
        _logger.info("Producer initialized (force=%s)", self.force)

    def process_file(self, scanned_file: ScannedFile, force: bool | None = None) -> Metadata | None:
        """Process a single file and write its JSON metadata.

        Determines the file type from the extension and dispatches to the
        appropriate processing path:

        - **Media**: transcribe -> summarize -> generate title -> generate
          metadata with content (transcript) and description (summary).
        - **PDF**: extract metadata -> summarize -> generate title -> generate
          metadata with content (extracted text) and summary.
        - **HTML**: extract metadata -> summarize -> generate title -> generate
          metadata with content (extracted text) and summary.

        For all file types, title generation uses the LLM with the
        filename, extracted title (if available), and content as context.

        On any exception, an error-state metadata file is written and the
        error is logged to stderr.

        Args:
            scanned_file: A :class:`~focus_feed.scanner.ScannedFile`
                instance describing the file to process.
            force: When ``True``, all processing steps are re-run even if
                existing step maps show ``"complete"`` status. When ``None``
                (default), uses the producer's ``self.force`` value.

        Returns:
            The :class:`Metadata` instance that was written, or ``None``
            if the file type is not supported.
        """
        if force is None:
            force = self.force

        if force:
            _logger.debug("Force mode active; processing all files regardless of status")

        filetype: str = scanned_file.filetype
        filepath: str = scanned_file.filepath
        filename: str = scanned_file.filename
        absolute_path: str = scanned_file.absolute_path
        try:
            source_mtime = Path(absolute_path).stat().st_mtime
        except OSError as exc:
            _logger.warning("Could not stat source file %s: %s", absolute_path, exc)
            source_mtime = None

        if filetype not in SUPPORTED_FILETYPES:
            _logger.warning(
                "Unsupported file type '%s' for file %s; skipping",
                filetype,
                filepath,
            )
            return None

        _logger.info("Processing %s file: %s", filetype, filepath)

        # Compute the JSON metadata path and write an initial in-progress
        # state so that the file is marked as being processed even if an
        # error occurs early.
        json_path = Path(absolute_path).with_name(Path(absolute_path).name + ".json")

        # If JSON metadata already exists, preserve existing step maps so that
        # already-complete steps are not overwritten with "in_progress".
        # When force=True, skip preservation to re-run all steps.
        existing_transcribe: dict[str, Any] | None = None
        existing_summarize: dict[str, Any] | None = None
        existing_extract: dict[str, Any] | None = None
        existing_generate_title: dict[str, Any] | None = None
        existing_content: str | None = None
        existing_hash: str | None = None
        existing_error: str | None = None
        existing_mtime: float | None = None
        existing_data: dict[str, Any] = {}
        if json_path.exists() and not force:
            try:
                existing_data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(existing_data, dict):
                    existing_transcribe = existing_data.get("transcribe")
                    existing_summarize = existing_data.get("summarize")
                    existing_extract = existing_data.get("extract")
                    existing_generate_title = existing_data.get("generate_title")
                    existing_content = existing_data.get("content")
                    existing_hash = existing_data.get("hash")
                    existing_error = existing_data.get("error")
                    stored_mtime = existing_data.get("mtime")
                    if isinstance(stored_mtime, (int, float)) and not isinstance(
                        stored_mtime, bool
                    ):
                        existing_mtime = float(stored_mtime)
            except json.JSONDecodeError, OSError:
                pass

        mtime_matches = (
            source_mtime is not None
            and existing_mtime is not None
            and source_mtime == existing_mtime
        )
        if not mtime_matches:
            # A changed source invalidates every result derived from it.
            existing_transcribe = None
            existing_summarize = None
            existing_extract = None
            existing_generate_title = None
            existing_content = None
            existing_hash = None
            existing_error = None

        # If the file is already feed-eligible and not forcing, skip processing.
        if not force:
            is_feed_eligible = False
            if filetype in MEDIA_FILETYPES:
                is_feed_eligible = (
                    existing_transcribe is not None
                    and existing_transcribe.get("status") == "complete"
                    and existing_summarize is not None
                    and existing_summarize.get("status") == "complete"
                )
            elif filetype in DOCUMENT_FILETYPES:
                is_feed_eligible = (
                    existing_extract is not None
                    and existing_extract.get("status") == "complete"
                    and existing_summarize is not None
                    and existing_summarize.get("status") == "complete"
                )
            if is_feed_eligible and mtime_matches:
                _logger.debug("Skipping %s: already feed-eligible", filepath)
                return None

            # If the file has an existing error state and we're not forcing,
            # skip it — the user needs --force to retry.
            # However, legacy top-level errors with zero step output can be
            # migrated to the step-map error format by processing the file.
            if existing_error and (existing_mtime is None or mtime_matches):
                is_migratable_legacy = False
                if (
                    filetype in MEDIA_FILETYPES
                    or is_audio_filetype(filetype)
                    or is_video_filetype(filetype)
                ):
                    transcribe = existing_data.get("transcribe")
                    if isinstance(transcribe, dict) and transcribe.get("output_size", 0) == 0:
                        is_migratable_legacy = True
                elif filetype in DOCUMENT_FILETYPES:
                    extract = existing_data.get("extract")
                    if isinstance(extract, dict) and extract.get("output_size", 0) == 0:
                        is_migratable_legacy = True

                if not is_migratable_legacy:
                    _logger.warning(
                        "File %s has existing error; skipping (use --force to retry)",
                        filepath,
                    )
                    return None

        partial_metadata: dict[str, Any] = {
            "filename": filename,
            "filetype": filetype,
            "mtime": source_mtime,
        }
        if existing_hash:
            partial_metadata["hash"] = existing_hash
        # Set the initial step status based on file type, preserving
        # existing step maps so completed steps are not overwritten.
        now_ts = _now_timestamp()
        if filetype in MEDIA_FILETYPES:
            partial_metadata["transcribe"] = {
                "status": "in_progress",
                "start_time": now_ts,
                "input_size": 0,
                "output_size": 0,
                "words": 0,
                "model": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            partial_metadata["summarize"] = {
                "status": "in_progress",
                "start_time": now_ts,
                "input_size": 0,
                "output_size": 0,
                "words": 0,
                "model": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        elif filetype in DOCUMENT_FILETYPES:
            partial_metadata["extract"] = {
                "status": "in_progress",
                "start_time": now_ts,
                "input_size": 0,
                "output_size": 0,
                "words": 0,
            }
            partial_metadata["summarize"] = {
                "status": "in_progress",
                "start_time": now_ts,
                "input_size": 0,
                "output_size": 0,
                "words": 0,
                "model": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        # Initialize the generate_title step map for all file types.  This is
        # an LLM-based step (like summarize) so it includes model and token
        # fields.  The step is populated with completion metrics when the LLM
        # is actually called; if title generation is skipped (e.g. no content
        # available for PDF/HTML), the key is removed before final metadata
        # construction so it does not appear in the output.
        partial_metadata["generate_title"] = {
            "status": "in_progress",
            "start_time": now_ts,
            "input_size": 0,
            "output_size": 0,
            "words": 0,
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Preserve existing step maps that are complete (only when not forcing).
        if not force:
            if filetype in MEDIA_FILETYPES:
                if existing_transcribe and existing_transcribe.get("status") == "complete":
                    partial_metadata["transcribe"] = existing_transcribe
                if existing_summarize and existing_summarize.get("status") == "complete":
                    partial_metadata["summarize"] = existing_summarize
            elif filetype in DOCUMENT_FILETYPES:
                if existing_extract and existing_extract.get("status") == "complete":
                    partial_metadata["extract"] = existing_extract
                if existing_summarize and existing_summarize.get("status") == "complete":
                    partial_metadata["summarize"] = existing_summarize
            # Preserve existing generate_title step map if already complete.
            if existing_generate_title and existing_generate_title.get("status") == "complete":
                partial_metadata["generate_title"] = existing_generate_title

            _logger.debug(
                "Resuming %s: transcribe=%s, summarize=%s, extract=%s, generate_title=%s",
                filepath,
                existing_transcribe.get("status") if existing_transcribe else None,
                existing_summarize.get("status") if existing_summarize else None,
                existing_extract.get("status") if existing_extract else None,
                existing_generate_title.get("status") if existing_generate_title else None,
            )

        # Preserve existing content (media transcript, only when not forcing).
        if existing_content and not force:
            partial_metadata["content"] = existing_content

        self._write_metadata_partial(partial_metadata, absolute_path)

        try:
            if filetype in MEDIA_FILETYPES:
                metadata = self._process_media(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                    filetype=filetype,
                    force=force,
                )
            elif filetype in ("pdf", *DOCLING_FILETYPES):
                metadata = self._process_pdf(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                    force=force,
                    filetype=cast(Literal["pdf"] | DoclingFiletype, filetype),
                )
            elif filetype == "html":
                metadata = self._process_html(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                    force=force,
                )
            else:
                # Should not reach here due to the check above, but kept
                # for exhaustiveness.
                return None

            _logger.info("Successfully processed %s", filepath)
            _logger.info("Metadata written for %s", filepath)
            return metadata

        except Exception as exc:
            self._handle_error(exc, filepath, filename, filetype, absolute_path)
            return None

    def process_all(self) -> list[Metadata | None]:
        """Process all files returned by the scanner.

        Uses :meth:`Scanner.scan_directory` to discover files that need
        processing, then calls :meth:`process_file` for each one.

        Returns:
            A list of :class:`Metadata` instances (or ``None`` values for
            unsupported file types) in the same order as the scanner's
            output.

        Raises:
            RuntimeError: If no scanner was provided to the Producer.
        """
        if self.scanner is None:
            raise RuntimeError(
                "Cannot call process_all() without a scanner. "
                "Provide a Scanner instance to the Producer, or use "
                "process_file() directly."
            )
        files = self.scanner.scan_directory()
        results: list[Metadata | None] = []

        for scanned_file in files:
            metadata = self.process_file(scanned_file)
            results.append(metadata)

        return results

    def build_title_dict(
        self,
        filename_title: str,
        extracted: str | None = None,
        generated_title: str | None = None,
    ) -> dict[str, str | None]:
        """Build a title dict with the three required keys.

        The dict always contains exactly three keys:

        - ``filename``: sanitized filename (no extension, underscores→spaces)
        - ``extracted``: title extracted from source metadata, or ``None``
        - ``generated``: LLM-generated title, or ``None``

        Args:
            filename_title: The already-sanitized filename (no extension,
                underscores replaced with spaces) to use as the ``filename``
                key.
            extracted: Title extracted from source metadata and mapped to
                ``title.extracted`` in the metadata schema, or
                ``None`` if not available.
            generated_title: LLM-generated title, or ``None`` if not
                generated.

        Returns:
            A dict with keys ``filename``, ``extracted``, and ``generated``.
        """
        return {
            "filename": filename_title,
            "extracted": extracted,
            "generated": generated_title,
        }

    def generate_title(
        self,
        content: str,
        filename: str | None = None,
        extracted: str | None = None,
        filepath: str | None = None,
    ) -> tuple[str, int, int]:
        """Generate a title from content using the LLM.

        This is a thin wrapper around
        :meth:`Summarizer.generate_title` that serves as a fallback when
        source metadata does not provide a title.

        Args:
            content: The content text to generate a title from
                (e.g. a transcript or file content).
            filename: Optional filename (without extension, underscores
                replaced with spaces) to provide as context to the LLM.
            extracted: Optional title extracted from source metadata (PDF
                metadata title, HTML ``<title>`` tag), mapped to
                ``title.extracted`` in the metadata schema.
            filepath: Optional path of the file being processed, included in
                error logging for debugging context.

        Returns:
            A tuple of ``(title, prompt_tokens, completion_tokens)``.

        Raises:
            SummarizationError: If the LLM request fails.
        """
        return self.summarizer.generate_title(content, filename, extracted, filepath=filepath)

    def _write_metadata_partial(self, metadata_dict: dict[str, Any], source_path: str) -> None:
        """Write a raw metadata dictionary to a JSON file atomically.

        Args:
            metadata_dict: The raw dictionary to write as JSON.
            source_path: Absolute path to the source file. The JSON
                metadata file will be written next to it.
        """
        write_metadata_json(metadata_dict, source_path)

    def _handle_error(
        self,
        exc: Exception,
        filepath: str,
        filename: str,
        filetype: str | None = None,
        absolute_path: str | None = None,
    ) -> None:
        """Handle a processing error by writing error metadata and logging.

        Creates an error-state metadata dictionary using
        :func:`create_error_metadata`, preserves any step maps that were
        already set from in-progress JSON metadata, writes it atomically as
        a JSON file adjacent to the source file, and logs the error to
        stderr via :func:`log_error`.

        Args:
            exc: The exception that occurred during processing.
            filepath: Relative path of the source file from the data
                directory root.
            filename: The source filename (e.g. ``"lecture.mp4"``).
            filetype: The file type, if known. Document formats include
                PDF, HTML, and Docling formats (``"docx"``, ``"doc"``,
                ``"pptx"``, ``"ppt"``, ``"xlsx"``, ``"xls"``).
            absolute_path: Absolute path to the source file. Used to
                determine where to write the adjacent JSON metadata file.
                Must be provided.
        """
        mtime: float | None = None
        if absolute_path:
            with suppress(OSError):
                mtime = Path(absolute_path).stat().st_mtime

        tagged_step = getattr(exc, "focus_feed_step", None)
        if tagged_step in {"transcribe", "summarize", "extract", "generate_title"}:
            step_error = (tagged_step, str(exc))
        elif filetype in MEDIA_FILETYPES:
            step_error = ("transcribe", str(exc))
        elif filetype in DOCUMENT_FILETYPES:
            step_error = ("extract", str(exc))
        else:
            step_error = None

        error_metadata = create_error_metadata(
            exc=exc,
            filename=filename,
            filepath=filepath,
            filetype=filetype,
            mtime=mtime,
            step_error=step_error,
        )

        # Determine the absolute path for writing the error JSON.
        if absolute_path is None:
            raise ValueError(
                "absolute_path is required for _handle_error; "
                "cannot fall back to data_dir since Producer no longer "
                "manages a data directory."
            )
        source_path = Path(absolute_path)

        json_path = source_path.with_name(source_path.name + ".json")

        # Preserve step maps from any existing in-progress metadata.
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    # Preserve any step maps that were already set.
                    for step_key in ("transcribe", "summarize", "extract", "generate_title"):
                        if step_key in existing:
                            step_map = existing[step_key]
                            if step_error and step_key == step_error[0]:
                                step_map = step_map.copy() if isinstance(step_map, dict) else {}
                                step_map["error"] = step_error[1]
                                step_map["output_size"] = 0
                                if step_map.get("status") != "complete":
                                    step_map["status"] = None
                            error_metadata[step_key] = step_map
                    # Preserve existing content (transcript for MP4).
                    if "content" in existing:
                        error_metadata["content"] = existing["content"]
                    # Preserve hash if already computed.
                    if "hash" in existing:
                        error_metadata["hash"] = existing["hash"]
            except json.JSONDecodeError, OSError:
                pass

        # Write the error metadata file atomically.
        self._write_metadata_partial(error_metadata, str(source_path))

        log_error(exc, filepath=filepath, logger=_logger, stage="processing")
