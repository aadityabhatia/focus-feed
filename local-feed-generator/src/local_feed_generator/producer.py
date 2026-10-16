"""Producer pipeline for local-feed-generator.

Orchestrates file processing based on file type. For MP4 files, it
transcribes audio to markdown, then generates key-takeaways summary via
LLM. For PDF/HTML files, it extracts metadata from the source file and
uses the filename (without extension, underscores→spaces) as the
``title.filename``. For all file types, it generates a descriptive title
via LLM using the filename, extracted title (if available), and content
as context, stored as ``title.generated``.

The Producer writes a JSON metadata file adjacent to each source file,
following the schema defined in design doc section 3.3. On errors, it
writes an error-state metadata file and logs the error to stderr.

Processing progress is tracked through separate step maps:
- ``transcribe``: StepMap for transcription (MP4 only).
- ``summarize``: StepMap for summarization (MP4 only).
- ``extract``: StepMap for metadata extraction (PDF/HTML only).

Date fields (``date``) are stored as UNIX epoch floats with millisecond
precision.  Per-step timing is tracked within each step map's
``start_time`` and ``end_time`` fields, also as epoch floats with
millisecond precision.

The title field is a dict with exactly three keys:
- ``filename``: sanitized filename (no extension, underscores→spaces)
- ``extracted``: title from source metadata (PDF metadata, HTML <title>,
  or MP4 metadata where available)
- ``generated``: LLM-generated title (all file types when content is
  available)

Title precedence for the final title: generated > extracted > filename.
The module name prefix is applied to the final title after precedence
resolution.

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
from local_feed_generator.metadata import Metadata, StepMap
from local_feed_generator.scanner import ScannedFile, Scanner
from local_feed_generator.services.extractors import (
    HTMLExtractor,
    PDFExtractor,
    _clean_title,
    get_audio_metadata,
)
from local_feed_generator.services.summarizer import Summarizer
from local_feed_generator.services.transcriber import CUDA_ERROR_MESSAGE, Transcriber

_logger = logging.getLogger("local_feed_generator")

# File types supported by the producer, mapped to their processing methods.
SUPPORTED_FILETYPES: frozenset[str] = frozenset({"mp4", "pdf", "html"})


def _now_timestamp() -> float:
    """Return the current UNIX epoch timestamp with millisecond precision."""
    return int(datetime.now(UTC).timestamp() * 1000) / 1000.0


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
        force: bool = False,
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
            force: When ``True``, all processing steps are re-run even if
                existing step maps show ``"complete"`` status. This is used
                by the ``process --force`` CLI command to force full
                re-processing of files.
        """
        self.scanner: Scanner = scanner or Scanner(data_dir=data_dir)
        self.transcriber: Transcriber = transcriber or Transcriber()
        self.summarizer: Summarizer = summarizer or Summarizer()
        self.pdf_extractor: PDFExtractor = pdf_extractor or PDFExtractor()
        self.html_extractor: HTMLExtractor = html_extractor or HTMLExtractor()
        self.force: bool = force

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, scanned_file: ScannedFile, force: bool | None = None) -> Metadata | None:
        """Process a single file and write its JSON metadata.

        Determines the file type from the extension and dispatches to the
        appropriate processing path:

        - **MP4**: transcribe -> summarize -> generate title -> generate
          metadata with content (transcript) and description (summary).
        - **PDF**: extract metadata -> generate title -> generate metadata
          with content (extracted text).
        - **HTML**: extract metadata -> generate title -> generate metadata
          with content (extracted text).

        For all file types, title generation uses the LLM with the
        filename, extracted title (if available), and content as context.

        On any exception, an error-state metadata file is written and the
        error is logged to stderr.

        Args:
            scanned_file: A :class:`~local_feed_generator.scanner.ScannedFile`
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

        # If JSON metadata already exists, preserve existing step maps so that
        # already-complete steps are not overwritten with "in_progress".
        # When force=True, skip preservation to re-run all steps.
        existing_transcribe: dict[str, Any] | None = None
        existing_summarize: dict[str, Any] | None = None
        existing_extract: dict[str, Any] | None = None
        existing_content: str | None = None
        if json_path.exists() and not force:
            try:
                existing_data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(existing_data, dict):
                    existing_transcribe = existing_data.get("transcribe")
                    existing_summarize = existing_data.get("summarize")
                    existing_extract = existing_data.get("extract")
                    existing_content = existing_data.get("content")
            except json.JSONDecodeError, OSError:
                pass

        partial_metadata: dict[str, Any] = {
            "filename": filename,
            "filepath": filepath,
            "filetype": filetype,
        }
        # Set the initial step status based on file type, preserving
        # existing step maps so completed steps are not overwritten.
        now_ts = _now_timestamp()
        if filetype == "mp4":
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
        elif filetype in ("pdf", "html"):
            partial_metadata["extract"] = {
                "status": "in_progress",
                "start_time": now_ts,
                "input_size": 0,
                "output_size": 0,
                "words": 0,
            }

        # Preserve existing step maps that are complete (only when not forcing).
        if not force:
            if filetype == "mp4":
                if existing_transcribe and existing_transcribe.get("status") == "complete":
                    partial_metadata["transcribe"] = existing_transcribe
                if existing_summarize and existing_summarize.get("status") == "complete":
                    partial_metadata["summarize"] = existing_summarize
            elif filetype in ("pdf", "html"):
                if existing_extract and existing_extract.get("status") == "complete":
                    partial_metadata["extract"] = existing_extract

        # Preserve existing content (transcript for MP4, only when not forcing).
        if existing_content and not force:
            partial_metadata["content"] = existing_content

        self._write_metadata_partial(partial_metadata, absolute_path)

        try:
            if filetype == "mp4":
                metadata = self._process_mp4(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                    force=force,
                )
            elif filetype == "pdf":
                metadata = self._process_pdf(
                    filepath=filepath,
                    filename=filename,
                    absolute_path=absolute_path,
                    json_path=str(json_path),
                    force=force,
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
    # Title dict building
    # ------------------------------------------------------------------

    def build_title_dict(
        self,
        filename_title: str,
        extracted_title: str | None = None,
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
            extracted_title: Title extracted from source metadata, or
                ``None`` if not available.
            generated_title: LLM-generated title, or ``None`` if not
                generated.

        Returns:
            A dict with keys ``filename``, ``extracted``, and ``generated``.
        """
        return {
            "filename": filename_title,
            "extracted": extracted_title,
            "generated": generated_title,
        }

    # ------------------------------------------------------------------
    # File-type-specific processing
    # ------------------------------------------------------------------

    def _process_mp4(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
        force: bool = False,
    ) -> Metadata:
        """Process an MP4 file: transcribe -> summarize -> generate title -> build metadata.

        Writes intermediate JSON states at each processing stage so that
        progress can be tracked. Uses :meth:`_write_metadata_partial` to
        bypass :class:`Metadata` validation for in-progress states.

        The transcript is used as content for LLM title generation, with
        the filename (without extension, underscores→spaces) passed as
        context. The ``title.generated`` field holds the LLM-generated
        title (module prefix is applied in ``Metadata.final_title``).
        The ``title.filename`` field holds the filename-derived title.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"lecture.mp4"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).
            force: When ``True``, re-run transcription and summarization
                even if existing step maps show ``"complete"`` status.

        Returns:
            A :class:`Metadata` instance with content (transcript) and
            summary (key takeaways from LLM). The description field is left
            empty for MP4 files.

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
            now_ts = _now_timestamp()
            metadata_dict = {
                "transcribe": {
                    "status": "in_progress",
                    "start_time": now_ts,
                    "input_size": 0,
                    "output_size": 0,
                    "words": 0,
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "summarize": {
                    "status": "in_progress",
                    "start_time": now_ts,
                    "input_size": 0,
                    "output_size": 0,
                    "words": 0,
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
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
        transcribe_step = metadata_dict.get("transcribe") or {}

        # Determine input size for the transcribe step.
        stat_result = Path(absolute_path).stat()
        input_size = stat_result.st_size

        # Determine the transcription model to use.
        transcribe_model = (
            config.transcription_remote_model or "remote"
            if config.transcription_endpoint
            else config.transcription_model
        )

        # Skip transcription if already complete (safety check at the producer
        # level; the CLI transcribe command also filters via
        # _find_files_needing_transcription, and the transcriber itself checks
        # the adjacent JSON as a last-resort guard).
        # When force=True, always re-transcribe to ensure fresh content.
        if transcribe_step.get("status") == "complete" and not force:
            transcript = metadata_dict.get("content", "")
        else:
            transcribe_step["status"] = "in_progress"
            transcribe_step["start_time"] = _now_timestamp()
            transcribe_step["input_size"] = input_size
            transcribe_step["output_size"] = 0
            transcribe_step["words"] = 0
            transcribe_step["model"] = transcribe_model
            transcribe_step["input_tokens"] = 0
            transcribe_step["output_tokens"] = 0
            metadata_dict["transcribe"] = transcribe_step
            self._write_metadata_partial(metadata_dict, absolute_path)

            # Step 1: Transcribe audio to markdown text.
            transcript = self.transcriber.transcribe(absolute_path)
            transcribe_end = datetime.now(UTC)

            # Update transcribe step with completion info.
            transcribe_step["status"] = "complete"
            transcribe_step["end_time"] = int(transcribe_end.timestamp() * 1000) / 1000.0
            transcribe_step["output_size"] = len(transcript.encode("utf-8")) if transcript else 0
            transcribe_step["words"] = len(transcript.split()) if transcript else 0
            metadata_dict["transcribe"] = transcribe_step

        # After transcription, before summarization: update statuses.
        metadata_dict["content"] = transcript
        summarize_step = metadata_dict.get("summarize") or {}
        summarize_step["status"] = "in_progress"
        summarize_step["start_time"] = _now_timestamp()
        summarize_step["input_size"] = len(transcript.encode("utf-8")) if transcript else 0
        summarize_step["output_size"] = 0
        summarize_step["words"] = 0
        summarize_step["model"] = config.llm_model
        summarize_step["input_tokens"] = 0
        summarize_step["output_tokens"] = 0
        metadata_dict["summarize"] = summarize_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 2: Summarize the transcript via LLM.
        # Check for reflection.md in the source file's directory.
        reflection_questions = _check_reflection_md(absolute_path)
        summary, prompt_tokens, completion_tokens = self.summarizer.summarize(
            transcript, reflection_questions=reflection_questions, filepath=filepath
        )
        summarize_end = datetime.now(UTC)

        # Update summarize step with completion info.
        summarize_step["status"] = "complete"
        summarize_step["end_time"] = int(summarize_end.timestamp() * 1000) / 1000.0
        summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
        summarize_step["words"] = len(summary.split()) if summary else 0
        summarize_step["model"] = config.llm_model
        summarize_step["input_tokens"] = prompt_tokens
        summarize_step["output_tokens"] = completion_tokens
        metadata_dict["summarize"] = summarize_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 3: Extract audio metadata via ffprobe for duration and creation_time.
        # This is done before title generation so the extracted title (if any)
        # can be passed to the LLM for context.
        file_size = stat_result.st_size
        file_mtime = int(stat_result.st_mtime * 1000) / 1000.0

        audio_meta = get_audio_metadata(absolute_path)

        # Use creation_time from media metadata as the date when available;
        # fall back to file modification time if ffprobe fails or no
        # creation_time tag is present.
        file_date = file_mtime
        creation_time_str: str | None = None
        duration: float | None = None
        extracted_title: str | None = None
        if audio_meta is not None:
            creation_time_str = audio_meta.get("creation_time")
            duration = audio_meta.get("duration")
            extracted_title = audio_meta.get("title")
            if creation_time_str:
                try:
                    parsed_date = datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=UTC)
                    file_date = int(parsed_date.timestamp() * 1000) / 1000.0
                except ValueError, TypeError:
                    _logger.warning(
                        "Failed to parse creation_time '%s' for %s; "
                        "falling back to file modification time",
                        creation_time_str,
                        absolute_path,
                    )

        # Step 4: Generate a title.  For MP4 files the transcript is the
        # best available content for title generation.  Pass the filename
        # (without extension, underscores→spaces) and any extracted title
        # from audio metadata as context to the LLM.
        filename_title = _sanitize_filename(filename)
        generated_title, _, _ = self.summarizer.generate_title(
            transcript,
            filename=filename_title,
            extracted_title=extracted_title,
            filepath=filepath,
        )
        generated_title = _clean_title(generated_title) or generated_title

        # Module prefix is applied in Metadata.final_title after precedence
        # resolution, so it is not applied here.

        # Build title dict for MP4 with all three keys.
        title_dict = self.build_title_dict(
            filename_title=filename_title,
            extracted_title=extracted_title,
            generated_title=generated_title,
        )

        # On completion: update to final state with all fields.
        # For MP4 files, description is left blank and the LLM-generated
        # summary is stored in the summary field instead.
        metadata_dict.update(
            {
                "title": title_dict,
                "description": "",  # MP4: leave description blank, use summary instead
                "summary": summary,
                "date": file_date,
                "size": file_size,
                "duration": duration,
                "creation_time": creation_time_str,
                "content": transcript,
            }
        )
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Return a Metadata object for in-memory use.
        metadata = Metadata(
            title=title_dict,
            extracted_title=extracted_title,
            description="",  # MP4: leave description blank, use summary instead
            summary=summary,
            filename=filename,
            filepath=filepath,
            filetype="mp4",
            date=file_date,
            size=file_size,
            duration=duration,
            creation_time=creation_time_str,
            content=transcript,
            transcribe=StepMap(**transcribe_step),
            summarize=StepMap(**summarize_step),
        )
        return metadata

    def _process_pdf(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
        force: bool = False,
    ) -> Metadata:
        """Process a PDF file: extract metadata -> generate title -> build metadata.

        Writes intermediate JSON states at each processing stage. Uses
        :meth:`_write_metadata_partial` to bypass :class:`Metadata`
        validation for in-progress states.

        Uses :class:`PDFExtractor` to obtain title, description, date, and
        size. The title is built using the filename-derived title and
        extracted title (from PDF metadata) as context for LLM title
        generation. The ``title.generated`` field holds the LLM-generated
        title (module prefix is applied in ``Metadata.final_title``).
        The ``title.extracted`` field stores the title from PDF metadata
        but is NOT used as the final title.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"doc.pdf"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).
            force: When ``True``, re-run extraction even if existing step
                maps show ``"complete"`` status.

        Returns:
            A :class:`Metadata` instance without transcript/summary.

        Raises:
            Exception: Any exception from extraction or metadata
                construction is propagated to the caller.
        """
        # Read the initial metadata dict written by process_file.
        json_path_obj = Path(json_path)
        if json_path_obj.exists():
            metadata_dict: dict[str, Any] = json.loads(json_path_obj.read_text(encoding="utf-8"))
        else:
            now_ts = _now_timestamp()
            metadata_dict = {
                "extract": {
                    "status": "in_progress",
                    "start_time": now_ts,
                    "input_size": 0,
                    "output_size": 0,
                    "words": 0,
                },
                "filename": filename,
                "filepath": filepath,
                "filetype": "pdf",
            }

        # Determine input size for the extract step.
        stat_result = Path(absolute_path).stat()
        input_size = stat_result.st_size

        # Before extraction: set extract status to in_progress.
        extract_step = metadata_dict.get("extract") or {}
        extract_step["status"] = "in_progress"
        extract_step["start_time"] = _now_timestamp()
        extract_step["input_size"] = input_size
        extract_step["output_size"] = 0
        extract_step["words"] = 0
        metadata_dict["extract"] = extract_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 1: Extract metadata from the PDF.
        extracted = self.pdf_extractor.extract(absolute_path)

        # The filename-derived title is used as the actual title.
        # The extracted title from PDF metadata is stored in the 'extracted'
        # key of the title dict for tracking and precedence.
        # Fall back to filename stem if extractor returns empty title.
        filename_title = extracted.get("title") or _sanitize_filename(filename) or filename
        extracted_title = extracted.get("extracted_title")
        description = extracted.get("description") or ""
        file_date = extracted.get("date", 0)
        file_publication_date = extracted.get("publication_date")
        file_size = extracted.get("size", 0)
        file_content = extracted.get("content")

        # After extraction: mark extract as complete.
        extract_step["status"] = "complete"
        extract_step["end_time"] = _now_timestamp()
        extract_step["output_size"] = len(file_content.encode("utf-8")) if file_content else 0
        extract_step["words"] = len(file_content.split()) if file_content else 0
        metadata_dict["extract"] = extract_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 2: Generate title via LLM using filename, extracted title,
        # and the extracted content (summarizer truncates to ~4000 chars).
        if file_content and file_content.strip():
            generated_title, _, _ = self.summarizer.generate_title(
                file_content,
                filename=filename_title,
                extracted_title=extracted_title,
                filepath=filepath,
            )
            generated_title = _clean_title(generated_title) or generated_title
        else:
            # No content available; fall back to filename-derived title.
            generated_title = filename_title

        # Module prefix is applied in Metadata.final_title after precedence
        # resolution, so it is not applied here.

        # Build title dict for PDF with all three keys.
        title_dict = self.build_title_dict(
            filename_title=filename_title,
            extracted_title=extracted_title,
            generated_title=generated_title,
        )

        # On completion: update to final state.
        metadata_dict.update(
            {
                "title": title_dict,
                "description": description,
                "date": file_date,
                "size": file_size,
                "content": file_content,
            }
        )
        if file_publication_date is not None:
            metadata_dict["publication_date"] = file_publication_date
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Return a Metadata object for in-memory use.
        metadata = Metadata(
            title=title_dict,
            extracted_title=extracted_title,
            description=description,
            filename=filename,
            filepath=filepath,
            filetype="pdf",
            date=file_date,
            publication_date=file_publication_date,
            size=file_size,
            content=file_content,
            extract=StepMap(**extract_step),
        )
        return metadata

    def _process_html(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
        force: bool = False,
    ) -> Metadata:
        """Process an HTML file: extract metadata -> generate title -> build metadata.

        Writes intermediate JSON states at each processing stage. Uses
        :meth:`_write_metadata_partial` to bypass :class:`Metadata`
        validation for in-progress states.

        Uses :class:`HTMLExtractor` to obtain title, description, date, and
        size. The title is built using the filename-derived title and
        extracted title (from the HTML ``<title>`` tag) as context for
        LLM title generation. The ``title.generated`` field holds the
        LLM-generated title (module prefix is applied in
        ``Metadata.final_title``). The ``title.extracted`` field stores
        the title from the HTML ``<title>`` tag but is NOT used as the
        final title.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"page.html"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).
            force: When ``True``, re-run extraction even if existing step
                maps show ``"complete"`` status.

        Returns:
            A :class:`Metadata` instance without transcript/summary.

        Raises:
            Exception: Any exception from extraction or metadata
                construction is propagated to the caller.
        """
        # Read the initial metadata dict written by process_file.
        json_path_obj = Path(json_path)
        if json_path_obj.exists():
            metadata_dict: dict[str, Any] = json.loads(json_path_obj.read_text(encoding="utf-8"))
        else:
            now_ts = _now_timestamp()
            metadata_dict = {
                "extract": {
                    "status": "in_progress",
                    "start_time": now_ts,
                    "input_size": 0,
                    "output_size": 0,
                    "words": 0,
                },
                "filename": filename,
                "filepath": filepath,
                "filetype": "html",
            }

        # Determine input size for the extract step.
        stat_result = Path(absolute_path).stat()
        input_size = stat_result.st_size

        # Before extraction: set extract status to in_progress.
        extract_step = metadata_dict.get("extract") or {}
        extract_step["status"] = "in_progress"
        extract_step["start_time"] = _now_timestamp()
        extract_step["input_size"] = input_size
        extract_step["output_size"] = 0
        extract_step["words"] = 0
        metadata_dict["extract"] = extract_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 1: Extract metadata from the HTML.
        extracted = self.html_extractor.extract(absolute_path)

        # The filename-derived title is used as the actual title.
        # The extracted title from <title> tag is stored in the 'extracted'
        # key of the title dict for tracking and precedence.
        # Fall back to filename stem if extractor returns empty title.
        filename_title = extracted.get("title") or _sanitize_filename(filename) or filename
        extracted_title = extracted.get("extracted_title")
        description = extracted.get("description") or ""
        file_date = extracted.get("date", 0)
        file_publication_date = extracted.get("publication_date")
        file_size = extracted.get("size", 0)
        file_content = extracted.get("content")

        # After extraction: mark extract as complete.
        extract_step["status"] = "complete"
        extract_step["end_time"] = _now_timestamp()
        extract_step["output_size"] = len(file_content.encode("utf-8")) if file_content else 0
        extract_step["words"] = len(file_content.split()) if file_content else 0
        metadata_dict["extract"] = extract_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Step 2: Generate title via LLM using filename, extracted title,
        # and the extracted content (summarizer truncates to ~4000 chars).
        if file_content and file_content.strip():
            generated_title, _, _ = self.summarizer.generate_title(
                file_content,
                filename=filename_title,
                extracted_title=extracted_title,
                filepath=filepath,
            )
            generated_title = _clean_title(generated_title) or generated_title
        else:
            # No content available; fall back to filename-derived title.
            generated_title = filename_title

        # Module prefix is applied in Metadata.final_title after precedence
        # resolution, so it is not applied here.

        # Build title dict for HTML with all three keys.
        title_dict = self.build_title_dict(
            filename_title=filename_title,
            extracted_title=extracted_title,
            generated_title=generated_title,
        )

        # On completion: update to final state.
        metadata_dict.update(
            {
                "title": title_dict,
                "description": description,
                "date": file_date,
                "size": file_size,
                "content": file_content,
            }
        )
        if file_publication_date is not None:
            metadata_dict["publication_date"] = file_publication_date
        self._write_metadata_partial(metadata_dict, absolute_path)

        # Return a Metadata object for in-memory use.
        metadata = Metadata(
            title=title_dict,
            extracted_title=extracted_title,
            description=description,
            filename=filename,
            filepath=filepath,
            filetype="html",
            date=file_date,
            publication_date=file_publication_date,
            size=file_size,
            content=file_content,
            extract=StepMap(**extract_step),
        )
        return metadata

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def generate_title(
        self,
        content: str,
        filename: str | None = None,
        extracted_title: str | None = None,
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
            extracted_title: Optional title extracted from source metadata
                (PDF metadata title, HTML ``<title>`` tag).
            filepath: Optional path of the file being processed, included in
                error logging for debugging context.

        Returns:
            A tuple of ``(title, prompt_tokens, completion_tokens)``.

        Raises:
            SummarizationError: If the LLM request fails.
        """
        return self.summarizer.generate_title(content, filename, extracted_title, filepath=filepath)

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
        filename = _sanitize_filename(path.name)
        return self.generate_title(content, filename=filename, filepath=str(filepath))

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
        with step maps and other fields.

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
        :func:`create_error_metadata`, preserves any step maps that were
        already set from in-progress JSON metadata, writes it atomically as
        a JSON file adjacent to the source file, and logs the error to
        stderr via :func:`log_error`.

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

        # Preserve step maps from any existing in-progress metadata.
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    # Preserve any step maps that were already set.
                    for step_key in ("transcribe", "summarize", "extract"):
                        if step_key in existing:
                            error_metadata[step_key] = existing[step_key]
                    # Preserve existing content (transcript for MP4).
                    if "content" in existing:
                        error_metadata["content"] = existing["content"]
            except json.JSONDecodeError, OSError:
                pass

        # Write the error metadata file atomically.
        self._write_metadata_partial(error_metadata, str(source_path))

        log_error(exc, filepath=filepath, logger=_logger)
