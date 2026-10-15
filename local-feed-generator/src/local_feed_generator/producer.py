"""Producer pipeline for local-feed-generator.

Orchestrates file processing based on file type. For MP4 files, it
transcribes audio to markdown, then generates key-takeaways summary via
LLM. For PDF/HTML files, it extracts metadata from the source file and
uses LLM to generate a title if source metadata is unavailable.

The Producer writes a JSON metadata file adjacent to each source file,
following the schema defined in design doc section 3.3. On errors, it
writes an error-state metadata file and logs the error to stderr.

Processing progress is tracked through a unified ``status`` dictionary:
- ``status["transcribe"]``: set to ``"in_progress"`` / ``"complete"`` for MP4
- ``status["summarize"]``: set to ``"in_progress"`` / ``"complete"`` for MP4
- ``status["extract"]``: set to ``"in_progress"`` / ``"complete"`` for PDF/HTML

Date fields (``date``, ``processing_date``, ``started_at``, ``completed_at``)
are stored as UNIX epoch integers (seconds since 1970-01-01 UTC).

Design references:
- design.md section 2.2 (Producer Pipeline responsibility)
- design.md section 5.2 (Producer module responsibilities)
- design.md section 6.1 (Batch processing flow)
- design.md section 7.1 (Error handling flow)
- tasks.md T3.1 (Producer implementation)
"""

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

from local_feed_generator.config import get_config
from local_feed_generator.errors import (
    create_error_metadata,
    log_error,
)
from local_feed_generator.metadata import Metadata
from local_feed_generator.scanner import ScannedFile, Scanner
from local_feed_generator.services.extractors import HTMLExtractor, PDFExtractor, _clean_title
from local_feed_generator.services.summarizer import Summarizer
from local_feed_generator.services.transcriber import CUDA_ERROR_MESSAGE, Transcriber

_logger = logging.getLogger("local_feed_generator")

# File types supported by the producer, mapped to their processing methods.
SUPPORTED_FILETYPES: frozenset[str] = frozenset({"mp4", "pdf", "html"})


class Producer:
    """Orchestrates file processing and metadata generation.

    The Producer accepts a :class:`~local_feed_generator.scanner.Scanner`
    (or data directory path) and optional service instances. When services
    are not provided, singleton instances are created lazily so that the
    expensive model loading (e.g. the Whisper transcription model) happens
    only once.

    Attributes:
        scanner: The :class:`Scanner` used for batch discovery.
        transcriber: The :class:`Transcriber` for MP4 audio transcription.
        summarizer: The :class:`Summarizer` for LLM-based summarization
            and title generation.
        pdf_extractor: The :class:`PDFExtractor` for PDF metadata.
        html_extractor: The :class:`HTMLExtractor` for HTML metadata.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        scanner: Scanner | None = None,
        transcriber: Transcriber | None = None,
        summarizer: Summarizer | None = None,
        pdf_extractor: PDFExtractor | None = None,
        html_extractor: HTMLExtractor | None = None,
    ) -> None:
        """Initialize the Producer with dependencies.

        Args:
            data_dir: Path to the data directory. When ``scanner`` is
                ``None``, a :class:`Scanner` is created with this directory.
                When both are ``None``, the data directory is loaded from
                configuration.
            scanner: Optional pre-configured :class:`Scanner` instance.
                Takes precedence over ``data_dir``.
            transcriber: Optional :class:`Transcriber` instance. When
                ``None``, a new instance is created (lazy model loading).
            summarizer: Optional :class:`Summarizer` instance. When
                ``None``, a new instance is created.
            pdf_extractor: Optional :class:`PDFExtractor` instance. When
                ``None``, a new instance is created.
            html_extractor: Optional :class:`HTMLExtractor` instance. When
                ``None``, a new instance is created.
        """
        self.scanner: Scanner = scanner or Scanner(data_dir=data_dir)
        self.transcriber: Transcriber = transcriber or Transcriber()
        self.summarizer: Summarizer = summarizer or Summarizer()
        self.pdf_extractor: PDFExtractor = pdf_extractor or PDFExtractor()
        self.html_extractor: HTMLExtractor = html_extractor or HTMLExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, scanned_file: ScannedFile) -> Metadata | None:
        """Process a single file and write its JSON metadata.

        Determines the file type from the extension and dispatches to the
        appropriate processing path:

        - **MP4**: transcribe -> summarize -> generate metadata with
          transcript and summary.
        - **PDF**: extract metadata -> generate metadata without
          transcript/summary.
        - **HTML**: extract metadata -> generate metadata without
          transcript/summary.

        On any exception, an error-state metadata file is written and the
        error is logged to stderr.

        Args:
            scanned_file: A :class:`~local_feed_generator.scanner.ScannedFile`
                instance describing the file to process.

        Returns:
            The :class:`Metadata` instance that was written, or ``None``
            if the file type is not supported.
        """
        filetype: str = scanned_file.filetype
        filepath: str = scanned_file.filepath
        filename: str = scanned_file.filename
        absolute_path: str = scanned_file.absolute_path

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
        now = datetime.now(UTC)

        # If JSON metadata already exists, preserve its status so that
        # already-complete steps are not overwritten with "in_progress".
        existing_status: dict[str, Any] | None = None
        existing_transcript: str | None = None
        if json_path.exists():
            try:
                existing_data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(existing_data, dict):
                    existing_status = existing_data.get("status")
                    existing_transcript = existing_data.get("transcript")
            except (json.JSONDecodeError, OSError):
                pass

        partial_metadata: dict[str, Any] = {
            "started_at": int(now.timestamp()),
            "filename": filename,
            "filepath": filepath,
            "filetype": filetype,
        }
        # Set the initial step status based on file type, preserving
        # existing status values so completed steps are not overwritten.
        if filetype == "mp4":
            base_status = {"transcribe": "in_progress", "summarize": "in_progress"}
        elif filetype in ("pdf", "html"):
            base_status = {"extract": "in_progress"}
        else:
            base_status = {}

        if existing_status and isinstance(existing_status, dict):
            # Merge: keep existing completed statuses, set others to in_progress
            merged_status = dict(base_status)
            for key, value in existing_status.items():
                if value == "complete":
                    merged_status[key] = "complete"
            partial_metadata["status"] = merged_status
        else:
            partial_metadata["status"] = base_status

        # Preserve existing transcript so _process_mp4 can skip re-transcription.
        if existing_transcript:
            partial_metadata["transcript"] = existing_transcript

        self._write_metadata_partial(partial_metadata, absolute_path)

        try:
            if filetype == "mp4":
                metadata = self._process_mp4(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                )
            elif filetype == "pdf":
                metadata = self._process_pdf(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                )
            elif filetype == "html":
                metadata = self._process_html(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                )
            else:
                # Should not reach here due to the check above, but kept
                # for exhaustiveness.
                return None

            _logger.info("Successfully processed %s", filepath)
            return metadata

        except Exception as exc:
            _logger.error("Error processing %s: %s", filepath, exc)
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
        """
        files = self.scanner.scan_directory()
        results: list[Metadata | None] = []

        for scanned_file in files:
            metadata = self.process_file(scanned_file)
            results.append(metadata)

        return results

    # ------------------------------------------------------------------
    # File-type-specific processing
    # ------------------------------------------------------------------

    def _process_mp4(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
    ) -> Metadata:
        """Process an MP4 file: transcribe -> summarize -> build metadata.

        Writes intermediate JSON states at each processing stage so that
        progress can be tracked. Uses :meth:`_write_metadata_partial` to
        bypass :class:`Metadata` validation for in-progress states.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"lecture.mp4"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).

        Returns:
            A :class:`Metadata` instance with transcript and summary.

        Raises:
            Exception: Any exception from transcription, summarization,
                or metadata construction is propagated to the caller
                (which writes error metadata).
        """
        # Read the initial metadata dict written by process_file.
        json_path_obj = Path(json_path)
        if json_path_obj.exists():
            metadata_dict: dict[str, Any] = json.loads(json_path_obj.read_text(encoding="utf-8"))
        else:
            metadata_dict = {
                "status": {"transcribe": "in_progress"},
                "started_at": int(datetime.now(UTC).timestamp()),
                "filename": filename,
                "filepath": filepath,
                "filetype": "mp4",
            }

        # Pre-emptive CUDA availability check: surface a clear, actionable
        # error before attempting to transcribe.  The Transcriber._load_model
        # method also performs this check, but failing early here avoids
        # writing an "in_progress" status that would never complete.
        config = get_config()
        if config.transcription_device == "cuda":
            from local_feed_generator.services.transcriber import check_cuda_available

            if not check_cuda_available():
                raise RuntimeError(CUDA_ERROR_MESSAGE)

        # Before transcription: set transcribe status to in_progress.
        metadata_dict["status"] = metadata_dict.get("status") or {}

        # Skip transcription if already complete (safety check at the producer
        # level; the CLI transcribe command also filters via
        # _find_files_needing_transcription, and the transcriber itself checks
        # the adjacent JSON as a last-resort guard).
        if metadata_dict["status"].get("transcribe") == "complete":
            transcript = metadata_dict.get("transcript", "")
        else:
            metadata_dict["status"]["transcribe"] = "in_progress"
            self._write_metadata_partial(metadata_dict, absolute_path)

            # Step 1: Transcribe audio to markdown text.
            transcript = self.transcriber.transcribe(absolute_path)

        # After transcription, before summarization: update statuses.
        metadata_dict["status"]["transcribe"] = "complete"
        metadata_dict["status"]["summarize"] = "in_progress"
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 2: Summarize the transcript via LLM.
        summary, _, _ = self.summarizer.summarize(transcript)

        # After summarization, before title generation: update status.
        metadata_dict["status"]["summarize"] = "complete"
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 3: Generate a title.  For MP4 files the transcript is the
        # best available content for title generation.
        title, _, _ = self.generate_title(transcript)
        title = _clean_title(title)

        # Step 4: Build metadata.
        stat_result = Path(absolute_path).stat()
        file_size = stat_result.st_size
        file_mtime = int(stat_result.st_mtime)
        now = datetime.now(UTC)

        # Determine which transcription model was used.
        if config.transcription_endpoint:
            transcription_model_used = config.transcription_remote_model or "remote"
        else:
            transcription_model_used = config.transcription_model

        # On completion: update to final state with all fields.
        metadata_dict.update(
            {
                "status": {"transcribe": "complete", "summarize": "complete"},
                "completed_at": int(now.timestamp()),
                "title": title,
                "description": summary,
                "date": file_mtime,
                "size": file_size,
                "transcript": transcript,
                "summary": summary,
                "processing_date": int(now.timestamp()),
                "transcription_model": transcription_model_used,
            }
        )
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Return a Metadata object for in-memory use.
        metadata = Metadata(
            status={"transcribe": "complete", "summarize": "complete"},
            title=title,
            description=summary,
            filename=filename,
            filepath=filepath,
            filetype="mp4",
            date=file_mtime,
            size=file_size,
            transcript=transcript,
            summary=summary,
            processing_date=int(now.timestamp()),
            transcription_model=transcription_model_used,
            started_at=metadata_dict.get("started_at"),
            completed_at=int(now.timestamp()),
        )
        return metadata

    def _process_pdf(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
    ) -> Metadata:
        """Process a PDF file: extract metadata -> build metadata.

        Writes intermediate JSON states at each processing stage. Uses
        :meth:`_write_metadata_partial` to bypass :class:`Metadata`
        validation for in-progress states.

        Uses :class:`PDFExtractor` to obtain title, description, date, and
        size. If the extractor returns no title (empty or None), the LLM
        is used as a fallback via :meth:`generate_title`.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"doc.pdf"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).

        Returns:
            A :class:`Metadata` instance without transcript/summary.

        Raises:
            Exception: Any exception from extraction, title generation,
                or metadata construction is propagated to the caller.
        """
        # Read the initial metadata dict written by process_file.
        json_path_obj = Path(json_path)
        if json_path_obj.exists():
            metadata_dict: dict[str, Any] = json.loads(json_path_obj.read_text(encoding="utf-8"))
        else:
            metadata_dict = {
                "status": {"extract": "in_progress"},
                "started_at": int(datetime.now(UTC).timestamp()),
                "filename": filename,
                "filepath": filepath,
                "filetype": "pdf",
            }

        # Before extraction: set extract status to in_progress.
        metadata_dict["status"] = metadata_dict.get("status") or {}
        metadata_dict["status"]["extract"] = "in_progress"
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 1: Extract metadata from the PDF.
        extracted = self.pdf_extractor.extract(absolute_path)

        title = extracted.get("title") or ""
        if not title.strip():
            # After extraction, before title generation: update status.
            self._write_metadata_partial(metadata_dict, absolute_path)

            # LLM fallback for title generation.
            title, _, _ = self.generate_title_from_file(absolute_path)
            title = _clean_title(title)

        description = extracted.get("description") or ""
        file_date = extracted.get("date", 0)
        file_size = extracted.get("size", 0)

        # After extraction (including title fallback): mark extract as complete.
        metadata_dict["status"]["extract"] = "complete"
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 2: Build final metadata.
        now = datetime.now(UTC)

        # On completion: update to final state.
        metadata_dict.update(
            {
                "status": {"extract": "complete"},
                "completed_at": int(now.timestamp()),
                "title": title,
                "description": description,
                "date": file_date,
                "size": file_size,
                "processing_date": int(now.timestamp()),
            }
        )
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Return a Metadata object for in-memory use.
        metadata = Metadata(
            status={"extract": "complete"},
            title=title,
            description=description,
            filename=filename,
            filepath=filepath,
            filetype="pdf",
            date=file_date,
            size=file_size,
            processing_date=int(now.timestamp()),
            started_at=metadata_dict.get("started_at"),
            completed_at=int(now.timestamp()),
        )
        return metadata

    def _process_html(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
    ) -> Metadata:
        """Process an HTML file: extract metadata -> build metadata.

        Writes intermediate JSON states at each processing stage. Uses
        :meth:`_write_metadata_partial` to bypass :class:`Metadata`
        validation for in-progress states.

        Uses :class:`HTMLExtractor` to obtain title, description, date, and
        size. If the extractor returns no title (empty or None), the LLM
        is used as a fallback via :meth:`generate_title`.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"page.html"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).

        Returns:
            A :class:`Metadata` instance without transcript/summary.

        Raises:
            Exception: Any exception from extraction, title generation,
                or metadata construction is propagated to the caller.
        """
        # Read the initial metadata dict written by process_file.
        json_path_obj = Path(json_path)
        if json_path_obj.exists():
            metadata_dict: dict[str, Any] = json.loads(json_path_obj.read_text(encoding="utf-8"))
        else:
            metadata_dict = {
                "status": {"extract": "in_progress"},
                "started_at": int(datetime.now(UTC).timestamp()),
                "filename": filename,
                "filepath": filepath,
                "filetype": "html",
            }

        # Before extraction: set extract status to in_progress.
        metadata_dict["status"] = metadata_dict.get("status") or {}
        metadata_dict["status"]["extract"] = "in_progress"
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 1: Extract metadata from the HTML.
        extracted = self.html_extractor.extract(absolute_path)

        title = extracted.get("title") or ""
        if not title.strip():
            # After extraction, before title generation: update status.
            self._write_metadata_partial(metadata_dict, absolute_path)

            # LLM fallback for title generation.
            title, _, _ = self.generate_title_from_file(absolute_path)
            title = _clean_title(title)

        description = extracted.get("description") or ""
        file_date = extracted.get("date", 0)
        file_size = extracted.get("size", 0)

        # After extraction (including title fallback): mark extract as complete.
        metadata_dict["status"]["extract"] = "complete"
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 2: Build final metadata.
        now = datetime.now(UTC)

        # On completion: update to final state.
        metadata_dict.update(
            {
                "status": {"extract": "complete"},
                "completed_at": int(now.timestamp()),
                "title": title,
                "description": description,
                "date": file_date,
                "size": file_size,
                "processing_date": int(now.timestamp()),
            }
        )
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Return a Metadata object for in-memory use.
        metadata = Metadata(
            status={"extract": "complete"},
            title=title,
            description=description,
            filename=filename,
            filepath=filepath,
            filetype="html",
            date=file_date,
            size=file_size,
            processing_date=int(now.timestamp()),
            started_at=metadata_dict.get("started_at"),
            completed_at=int(now.timestamp()),
        )
        return metadata

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def generate_title(self, content: str) -> tuple[str, int, int]:
        """Generate a title from content using the LLM.

        This is a thin wrapper around
        :meth:`Summarizer.generate_title` that serves as a fallback when
        source metadata does not provide a title.

        Args:
            content: The content text to generate a title from
                (e.g. a transcript or file content).

        Returns:
            A tuple of ``(title, prompt_tokens, completion_tokens)``.

        Raises:
            SummarizationError: If the LLM request fails.
        """
        return self.summarizer.generate_title(content)

    def generate_title_from_file(self, filepath: str | Path) -> tuple[str, int, int]:
        """Generate a title from a file's content using the LLM.

        Reads the first portion of the file and passes it to
        :meth:`generate_title` as a fallback when the extractor did not
        find a usable title.

        Args:
            filepath: Path to the file whose content should be used for
                title generation.

        Returns:
            A tuple of ``(title, prompt_tokens, completion_tokens)``.

        Raises:
            SummarizationError: If the LLM request fails.
            OSError: If the file cannot be read.
        """
        path = Path(filepath)
        # Read a reasonable chunk of the file for title generation.
        content = path.read_text(encoding="utf-8", errors="replace")[:8000]
        return self.generate_title(content)

    def _write_metadata(self, metadata: Metadata, source_path: str) -> None:
        """Write a metadata JSON file adjacent to the source file.

        The JSON file is named ``<source_filename>.json`` and placed in
        the same directory as the source file.

        Args:
            metadata: The :class:`Metadata` instance to write.
            source_path: Absolute path to the source file. The JSON
                metadata file will be written next to it.
        """
        json_path = Path(source_path).with_name(Path(source_path).name + ".json")
        metadata.to_json_file(json_path)

    def _write_metadata_partial(self, metadata_dict: dict[str, Any], source_path: str) -> None:
        """Write a raw metadata dictionary to a JSON file atomically.

        Writes to a temporary file first, then renames to the final path
        to avoid corruption from partial writes. This bypasses
        :class:`Metadata` model validation, allowing intermediate states
        with fields like ``status``, ``started_at``, ``completed_at``,
        and ``transcription_model``.

        Args:
            metadata_dict: The raw dictionary to write as JSON.
            source_path: Absolute path to the source file. The JSON
                metadata file will be written next to it.
        """
        json_path = Path(source_path).with_name(Path(source_path).name + ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file first, then rename for atomicity.
        tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(metadata_dict, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(json_path)

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
        :func:`create_error_metadata`, preserves the ``stage`` and
        ``started_at`` from any existing in-progress JSON metadata, writes
        it atomically as a JSON file adjacent to the source file, and logs
        the error to stderr via :func:`log_error`.

        Args:
            exc: The exception that occurred during processing.
            filepath: Relative path of the source file from the data
                directory root.
            filename: The source filename (e.g. ``"lecture.mp4"``).
            filetype: The file type (``"mp4"``, ``"pdf"``, ``"html"``),
                if known.
            absolute_path: Absolute path to the source file. Used to
                determine where to write the adjacent JSON metadata file.
                When ``None``, the path is resolved from ``filepath``
                using the configured ``DATA_DIR``.
        """
        error_metadata = create_error_metadata(
            exc=exc,
            filename=filename,
            filepath=filepath,
            filetype=filetype,
        )

        # Determine the absolute path for writing the error JSON.
        if absolute_path is not None:
            source_path = Path(absolute_path)
        else:
            config = get_config()
            data_dir = Path(config.data_dir).resolve()
            source_path = data_dir / filepath

        json_path = source_path.with_name(source_path.name + ".json")

        # Preserve status and started_at from any existing in-progress metadata.
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
                if "started_at" in existing:
                    error_metadata["started_at"] = existing["started_at"]
                # Preserve any status map that was already set.
                if "status" in existing:
                    error_metadata["status"] = existing["status"]
            except json.JSONDecodeError, OSError:
                pass

        # Write the error metadata file atomically.
        self._write_metadata_partial(error_metadata, str(source_path))

        log_error(exc, filepath=filepath, logger=_logger)
