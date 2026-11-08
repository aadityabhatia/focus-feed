"""Document (PDF/HTML) processing strategies."""

# These methods use attributes supplied by ProducerBase in the final MRO.
# mypy sees the mixin in isolation, before that MRO is composed.
# mypy: disable-error-code="attr-defined"

import json
import logging
from pathlib import Path
from typing import Any, Literal

from focus_feed.filetypes import DOCLING_FILETYPES, DoclingFiletype
from focus_feed.metadata import Metadata, StepMap
from focus_feed.services.extractors import DoclingExtractor

from ._base import (
    _check_reflection_md,
    _clean_title,
    _compute_file_hash,
    _now_timestamp,
    _sanitize_filename,
    _tag_step,
    get_config,
    now_timestamp,
)

_logger = logging.getLogger("focus_feed")


class DocumentProcessingMixin:
    def _process_pdf(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
        force: bool = False,
        filetype: Literal["pdf"] | DoclingFiletype = "pdf",
    ) -> Metadata:
        """Process a PDF or Docling document: extract -> summarize -> title.

        Writes intermediate JSON states at each processing stage. Uses
        :meth:`_write_metadata_partial` to bypass :class:`Metadata`
        validation for in-progress states.

        Uses :class:`PDFExtractor` to obtain title, description, date, and
        size. The extracted content is then summarized via the LLM to
        produce key takeaways stored in the ``summary`` field. The title is
        built using the filename-derived title and extracted title (from PDF
        metadata) as context for LLM title generation. The ``title.generated``
        field holds the LLM-generated title (module prefix is applied in
        ``Metadata.final_title``). The ``title.extracted`` field stores the
        title from PDF metadata but is NOT used as the final title.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"doc.pdf"`` or ``"slides.pptx"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).
            force: When ``True``, re-run extraction and summarization even
                if existing step maps show ``"complete"`` status.
            filetype: The source document extension without its leading dot.

        Returns:
            A :class:`Metadata` instance with extract and summarize steps
            complete.

        Raises:
            Exception: Any exception from extraction, summarization, or
                metadata construction is propagated to the caller.
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
                "generate_title": {
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
                "filetype": filetype,
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

        _logger.info("Starting extraction for %s (step 1/3)", filepath)
        # Step 1: Use Docling for its supported document formats. PDF uses the
        # local PDFExtractor only when Docling is not configured.
        config = get_config()
        extraction: dict[str, Any] | None = None
        extension = Path(absolute_path).suffix.lower()
        with _tag_step("extract"):
            if filetype in DOCLING_FILETYPES:
                if not config.docling_api:
                    raise RuntimeError("DOCLING_API is not configured")
                extraction = self.docling_extractor.extract(absolute_path)
                if extraction.get("content") is None:
                    raise RuntimeError(f"Docling extraction produced no content for {filepath}")
            elif config.docling_api and extension in DoclingExtractor.SUPPORTED_EXTENSIONS:
                try:
                    docling_extracted = self.docling_extractor.extract(absolute_path)
                except (ConnectionError, RuntimeError, ValueError) as exc:
                    _logger.error("Docling extraction failed for %s: %s", filepath, exc)
                    raise RuntimeError(f"Docling extraction failed for {filepath}: {exc}") from exc

                if not docling_extracted.get("content"):
                    _logger.error("Docling extraction produced no content for %s", filepath)
                    raise RuntimeError(f"Docling extraction produced no content for {filepath}")
                extraction = docling_extracted

            if extraction is None:
                extraction = self.pdf_extractor.extract(absolute_path)
        metadata_dict["hash"] = _compute_file_hash(absolute_path)

        # The filename-derived title is used as the actual title.
        # The extracted title from PDF metadata is stored in the 'extracted'
        # key of the title dict for tracking and precedence.
        # Fall back to filename stem if extractor returns empty title.
        filename_title = extraction.get("title") or _sanitize_filename(filename) or filename
        extracted = extraction.get("extracted")
        description = extraction.get("description") or ""
        file_date = extraction.get("date", 0)
        file_publication_date = extraction.get("publication_date")
        file_size = extraction.get("size", 0)
        file_content = extraction.get("content")

        # After extraction: mark extract as complete.
        extract_step["status"] = "complete"
        extract_step["end_time"] = _now_timestamp()
        extract_step["output_size"] = len(file_content.encode("utf-8")) if file_content else 0
        extract_step["words"] = len(file_content.split()) if file_content else 0
        metadata_dict["extract"] = extract_step
        self._write_metadata_partial(metadata_dict, absolute_path)
        _logger.info(
            "Extraction completed for %s: title='%s', %d content chars",
            filepath,
            filename_title[:50],
            len(file_content) if file_content else 0,
        )

        # Step 2: Summarize the extracted content via LLM.
        # Check for reflection.md in the source file's directory.
        reflection_questions = _check_reflection_md(absolute_path)
        summarize_step = metadata_dict.get("summarize") or {}
        if summarize_step.get("status") == "complete" and not force:
            # Summarization already complete; reuse existing summary.
            summary = metadata_dict.get("summary", "")
        else:
            summarize_step["status"] = "in_progress"
            summarize_step["start_time"] = _now_timestamp()
            summarize_step["input_size"] = len(file_content.encode("utf-8")) if file_content else 0
            summarize_step["output_size"] = 0
            summarize_step["words"] = 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = 0
            summarize_step["output_tokens"] = 0
            metadata_dict["summarize"] = summarize_step
            self._write_metadata_partial(metadata_dict, absolute_path)

            _logger.info("Starting summarization for %s (step 2/3)", filepath)
            if file_content and file_content.strip():
                with _tag_step("summarize"):
                    summary, prompt_tokens, completion_tokens = self.summarizer.summarize(
                        file_content,
                        reflection_questions=reflection_questions,
                        filepath=filepath,
                        filetype=filetype,
                    )
            else:
                # No content to summarize; use empty summary.
                summary = ""
                prompt_tokens = 0
                completion_tokens = 0

            summarize_step["status"] = "complete"
            summarize_step["end_time"] = _now_timestamp()
            summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
            summarize_step["words"] = len(summary.split()) if summary else 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = prompt_tokens
            summarize_step["output_tokens"] = completion_tokens
            metadata_dict["summarize"] = summarize_step
            metadata_dict["summary"] = summary
            self._write_metadata_partial(metadata_dict, absolute_path)
            _logger.info("Summarization completed for %s", filepath)

        # Step 3: Generate title via LLM using filename, extracted title,
        # and the extracted content (summarizer truncates to ~4000 chars).
        _logger.info("Starting title generation for %s (step 3/3)", filepath)
        generate_title_step = metadata_dict.get("generate_title") or {}
        if file_content and file_content.strip():
            # Initialize/update the generate_title step map before the LLM call.
            generate_title_step["status"] = "in_progress"
            generate_title_step["start_time"] = _now_timestamp()
            generate_title_step["input_size"] = len(file_content.encode("utf-8"))
            generate_title_step["output_size"] = 0
            generate_title_step["words"] = 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = 0
            generate_title_step["output_tokens"] = 0
            metadata_dict["generate_title"] = generate_title_step
            self._write_metadata_partial(metadata_dict, absolute_path)

            with _tag_step("generate_title"):
                generated_title, prompt_tokens, completion_tokens = self.summarizer.generate_title(
                    file_content,
                    filename=filename_title,
                    extracted=extracted,
                    filepath=filepath,
                )
            generated_title = _clean_title(generated_title) or generated_title

            # Update generate_title step with completion info.
            generate_title_step["status"] = "complete"
            generate_title_step["end_time"] = now_timestamp()
            generate_title_step["output_size"] = (
                len(generated_title.encode("utf-8")) if generated_title else 0
            )
            generate_title_step["words"] = len(generated_title.split()) if generated_title else 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = prompt_tokens
            generate_title_step["output_tokens"] = completion_tokens
            metadata_dict["generate_title"] = generate_title_step
        else:
            # No content available; fall back to filename-derived title.
            # The LLM is not called, so remove the generate_title step map
            # (initialized as in_progress in process_file) to avoid leaving
            # a stale in_progress entry in the final metadata.
            generated_title = filename_title
            metadata_dict.pop("generate_title", None)
        _logger.info("Title generation completed for %s", filepath)

        # Module prefix is applied in Metadata.final_title after precedence
        # resolution, so it is not applied here.

        # Build title dict for PDF with all three keys.
        title_dict = self.build_title_dict(
            filename_title=filename_title,
            extracted=extracted,
            generated_title=generated_title,
        )

        # On completion: update to final state.
        metadata_dict.update(
            {
                "title": title_dict,
                "description": description,
                "date": file_date,
                "mtime": stat_result.st_mtime,
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
            description=description,
            summary=summary,
            filename=filename,
            filetype=filetype,
            date=file_date,
            mtime=stat_result.st_mtime,
            publication_date=file_publication_date,
            size=file_size,
            content=file_content,
            hash=metadata_dict.get("hash"),
            extract=StepMap(**extract_step),
            summarize=StepMap(**summarize_step),
            generate_title=StepMap(**generate_title_step)
            if generate_title_step.get("status") == "complete"
            else None,
        )
        metadata._source_filepath = filepath
        return metadata

    def _process_html(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
        force: bool = False,
    ) -> Metadata:
        """Process an HTML file: extract metadata -> summarize -> generate title -> build metadata.

        Writes intermediate JSON states at each processing stage. Uses
        :meth:`_write_metadata_partial` to bypass :class:`Metadata`
        validation for in-progress states.

        Uses :class:`HTMLExtractor` to obtain title, description, date, and
        size. The extracted content is then summarized via the LLM to
        produce key takeaways stored in the ``summary`` field. The title is
        built using the filename-derived title and extracted title (from the
        HTML ``<title>`` tag) as context for LLM title generation. The
        ``title.generated`` field holds the LLM-generated title (module
        prefix is applied in ``Metadata.final_title``). The ``title.extracted``
        field stores the title from the HTML ``<title>`` tag but is NOT used
        as the final title.

        Args:
            filepath: Relative path from the data directory root.
            filename: The source filename (e.g. ``"page.html"``).
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).
            force: When ``True``, re-run extraction and summarization even
                if existing step maps show ``"complete"`` status.

        Returns:
            A :class:`Metadata` instance with extract and summarize steps
            complete.

        Raises:
            Exception: Any exception from extraction, summarization, or
                metadata construction is propagated to the caller.
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
                "generate_title": {
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

        _logger.info("Starting extraction for %s (step 1/3)", filepath)
        # Step 1: Extract metadata from the HTML.
        with _tag_step("extract"):
            extraction = self.html_extractor.extract(absolute_path)
        metadata_dict["hash"] = _compute_file_hash(absolute_path)

        # The filename-derived title is used as the actual title.
        # The extracted title from <title> tag is stored in the 'extracted'
        # key of the title dict for tracking and precedence.
        # Fall back to filename stem if extractor returns empty title.
        filename_title = extraction.get("title") or _sanitize_filename(filename) or filename
        extracted = extraction.get("extracted")
        description = extraction.get("description") or ""
        file_date = extraction.get("date", 0)
        file_publication_date = extraction.get("publication_date")
        file_size = extraction.get("size", 0)
        file_content = extraction.get("content")

        # After extraction: mark extract as complete.
        extract_step["status"] = "complete"
        extract_step["end_time"] = _now_timestamp()
        extract_step["output_size"] = len(file_content.encode("utf-8")) if file_content else 0
        extract_step["words"] = len(file_content.split()) if file_content else 0
        metadata_dict["extract"] = extract_step
        self._write_metadata_partial(metadata_dict, absolute_path)
        _logger.info(
            "Extraction completed for %s: title='%s', %d content chars",
            filepath,
            filename_title[:50],
            len(file_content) if file_content else 0,
        )

        # Get config for LLM model used in summarization.
        config = get_config()

        # Step 2: Summarize the extracted content via LLM.
        # Check for reflection.md in the source file's directory.
        reflection_questions = _check_reflection_md(absolute_path)
        summarize_step = metadata_dict.get("summarize") or {}
        if summarize_step.get("status") == "complete" and not force:
            # Summarization already complete; reuse existing summary.
            summary = metadata_dict.get("summary", "")
        else:
            summarize_step["status"] = "in_progress"
            summarize_step["start_time"] = _now_timestamp()
            summarize_step["input_size"] = len(file_content.encode("utf-8")) if file_content else 0
            summarize_step["output_size"] = 0
            summarize_step["words"] = 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = 0
            summarize_step["output_tokens"] = 0
            metadata_dict["summarize"] = summarize_step
            self._write_metadata_partial(metadata_dict, absolute_path)

            _logger.info("Starting summarization for %s (step 2/3)", filepath)
            if file_content and file_content.strip():
                with _tag_step("summarize"):
                    summary, prompt_tokens, completion_tokens = self.summarizer.summarize(
                        file_content,
                        reflection_questions=reflection_questions,
                        filepath=filepath,
                        filetype="html",
                    )
            else:
                # No content to summarize; use empty summary.
                summary = ""
                prompt_tokens = 0
                completion_tokens = 0

            summarize_step["status"] = "complete"
            summarize_step["end_time"] = _now_timestamp()
            summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
            summarize_step["words"] = len(summary.split()) if summary else 0
            summarize_step["model"] = config.llm_model
            summarize_step["input_tokens"] = prompt_tokens
            summarize_step["output_tokens"] = completion_tokens
            metadata_dict["summarize"] = summarize_step
            metadata_dict["summary"] = summary
            self._write_metadata_partial(metadata_dict, absolute_path)
            _logger.info("Summarization completed for %s", filepath)

        # Step 3: Generate title via LLM using filename, extracted title,
        # and the extracted content (summarizer truncates to ~4000 chars).
        _logger.info("Starting title generation for %s (step 3/3)", filepath)
        generate_title_step = metadata_dict.get("generate_title") or {}
        if file_content and file_content.strip():
            # Initialize/update the generate_title step map before the LLM call.
            generate_title_step["status"] = "in_progress"
            generate_title_step["start_time"] = _now_timestamp()
            generate_title_step["input_size"] = len(file_content.encode("utf-8"))
            generate_title_step["output_size"] = 0
            generate_title_step["words"] = 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = 0
            generate_title_step["output_tokens"] = 0
            metadata_dict["generate_title"] = generate_title_step
            self._write_metadata_partial(metadata_dict, absolute_path)

            with _tag_step("generate_title"):
                generated_title, prompt_tokens, completion_tokens = self.summarizer.generate_title(
                    file_content,
                    filename=filename_title,
                    extracted=extracted,
                    filepath=filepath,
                )
            generated_title = _clean_title(generated_title) or generated_title

            # Update generate_title step with completion info.
            generate_title_step["status"] = "complete"
            generate_title_step["end_time"] = now_timestamp()
            generate_title_step["output_size"] = (
                len(generated_title.encode("utf-8")) if generated_title else 0
            )
            generate_title_step["words"] = len(generated_title.split()) if generated_title else 0
            generate_title_step["model"] = config.llm_model
            generate_title_step["input_tokens"] = prompt_tokens
            generate_title_step["output_tokens"] = completion_tokens
            metadata_dict["generate_title"] = generate_title_step
        else:
            # No content available; fall back to filename-derived title.
            # The LLM is not called, so remove the generate_title step map
            # (initialized as in_progress in process_file) to avoid leaving
            # a stale in_progress entry in the final metadata.
            generated_title = filename_title
            metadata_dict.pop("generate_title", None)
        _logger.info("Title generation completed for %s", filepath)

        # Module prefix is applied in Metadata.final_title after precedence
        # resolution, so it is not applied here.

        # Build title dict for HTML with all three keys.
        title_dict = self.build_title_dict(
            filename_title=filename_title,
            extracted=extracted,
            generated_title=generated_title,
        )

        # On completion: update to final state.
        metadata_dict.update(
            {
                "title": title_dict,
                "description": description,
                "date": file_date,
                "mtime": stat_result.st_mtime,
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
            description=description,
            summary=summary,
            filename=filename,
            filetype="html",
            date=file_date,
            mtime=stat_result.st_mtime,
            publication_date=file_publication_date,
            size=file_size,
            content=file_content,
            hash=metadata_dict.get("hash"),
            extract=StepMap(**extract_step),
            summarize=StepMap(**summarize_step),
            generate_title=StepMap(**generate_title_step)
            if generate_title_step.get("status") == "complete"
            else None,
        )
        metadata._source_filepath = filepath
        return metadata
