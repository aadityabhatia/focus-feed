"""Media (audio/video) processing strategy."""

# These methods use attributes supplied by ProducerBase in the final MRO.
# mypy sees the mixin in isolation, before that MRO is composed.
# mypy: disable-error-code="attr-defined"

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any, cast

from focus_feed.errors import log_error
from focus_feed.filetypes import MediaFiletype
from focus_feed.metadata import Metadata, StepMap

from ._base import (
    _check_reflection_md,
    _clean_title,
    _compute_file_hash,
    _now_timestamp,
    _sanitize_filename,
    _tag_step,
    get_audio_metadata,
    get_config,
    now_timestamp,
)

_logger = logging.getLogger("focus_feed")


class MediaProcessingMixin:
    def _process_media(
        self,
        filepath: str,
        filename: str,
        absolute_path: str,
        json_path: str,
        filetype: str,
        force: bool = False,
    ) -> Metadata:
        """Process an audio/video file: transcribe -> summarize -> generate title -> build metadata.

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
            filename: The source filename, including its media extension.
            absolute_path: Absolute path to the source file.
            json_path: Absolute path to the JSON metadata file (written
                by :meth:`process_file` as the initial in-progress state).
            filetype: Lowercase source media extension without the leading dot.
            force: When ``True``, re-run transcription and summarization
                even if existing step maps show ``"complete"`` status.

        Returns:
            A :class:`Metadata` instance with content (transcript) and
            summary (key takeaways from LLM). The description field is left
            empty for media files.

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

        config = get_config()

        # Before transcription: set transcribe status to in_progress.
        transcribe_step = metadata_dict.get("transcribe") or {}

        # Determine input size for the transcribe step.
        stat_result = Path(absolute_path).stat()
        input_size = stat_result.st_size

        # Determine the transcription model to use.
        transcribe_model = config.transcription_remote_model or "remote"

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

            _logger.info("Starting transcription for %s (step 1/3)", filepath)
            # Step 1: Transcribe audio to markdown text.
            with _tag_step("transcribe"):
                transcript = self.transcriber.transcribe(absolute_path)

            # Update transcribe step with completion info.
            transcribe_step["status"] = "complete"
            transcribe_step["end_time"] = now_timestamp()
            transcribe_step["output_size"] = len(transcript.encode("utf-8")) if transcript else 0
            transcribe_step["words"] = len(transcript.split()) if transcript else 0
            metadata_dict["transcribe"] = transcribe_step

        # Hash the source during its processing pipeline, after transcription.
        metadata_dict["hash"] = _compute_file_hash(absolute_path)

        # After transcription, before summarization: update statuses.
        metadata_dict["content"] = transcript
        _logger.info(
            "Transcription completed for %s: %d words",
            filepath,
            len(transcript.split()) if transcript else 0,
        )
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

        _logger.info("Starting summarization for %s (step 2/3)", filepath)
        # Step 2: Summarize the transcript via LLM.
        # Check for reflection.md in the source file's directory.
        reflection_questions = _check_reflection_md(absolute_path)
        with _tag_step("summarize"):
            summary, prompt_tokens, completion_tokens = self.summarizer.summarize(
                transcript,
                reflection_questions=reflection_questions,
                filepath=filepath,
                filetype=filetype,
            )

        # Update summarize step with completion info.
        summarize_step["status"] = "complete"
        summarize_step["end_time"] = now_timestamp()
        summarize_step["output_size"] = len(summary.encode("utf-8")) if summary else 0
        summarize_step["words"] = len(summary.split()) if summary else 0
        summarize_step["model"] = config.llm_model
        summarize_step["input_tokens"] = prompt_tokens
        summarize_step["output_tokens"] = completion_tokens
        metadata_dict["summarize"] = summarize_step
        self._write_metadata_partial(metadata_dict, absolute_path)
        _logger.info("Summarization completed for %s", filepath)

        # Step 3: Extract media metadata via ffprobe for duration and creation_time.
        # This is done before title generation so the extracted title (if any)
        # can be passed to the LLM for context.
        file_size = stat_result.st_size
        file_mtime = int(stat_result.st_mtime * 1000) / 1000.0

        _logger.debug("Extracting audio metadata for %s", absolute_path)
        audio_meta = get_audio_metadata(absolute_path, filetype=filetype)

        # Use creation_time from media metadata as the date when available;
        # fall back to file modification time if probing fails or no
        # creation_time tag is present.
        file_date = file_mtime
        creation_time_str: str | None = None
        duration: float | None = None
        extracted: str | None = None
        if audio_meta is not None:
            creation_time_str = audio_meta.get("creation_time")
            duration = audio_meta.get("duration")
            extracted = audio_meta.get("title")
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
        else:
            log_error(
                RuntimeError("Audio metadata extraction failed"),
                filepath=absolute_path,
                stage="audio_metadata_extraction",
            )

        # Step 4: Generate a title. The transcript is the best available
        # content for title generation. Pass the filename
        # (without extension, underscores→spaces) and any extracted title
        # from audio metadata as context to the LLM.
        filename_title = _sanitize_filename(filename)
        _logger.info("Starting title generation for %s (step 3/3)", filepath)

        # Initialize/update the generate_title step map before the LLM call.
        generate_title_step = metadata_dict.get("generate_title") or {}
        generate_title_step["status"] = "in_progress"
        generate_title_step["start_time"] = _now_timestamp()
        generate_title_step["input_size"] = len(transcript.encode("utf-8")) if transcript else 0
        generate_title_step["output_size"] = 0
        generate_title_step["words"] = 0
        generate_title_step["model"] = config.llm_model
        generate_title_step["input_tokens"] = 0
        generate_title_step["output_tokens"] = 0
        metadata_dict["generate_title"] = generate_title_step
        self._write_metadata_partial(metadata_dict, absolute_path)

        with _tag_step("generate_title"):
            generated_title, prompt_tokens, completion_tokens = self.summarizer.generate_title(
                transcript,
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
        _logger.info("Title generation completed for %s", filepath)

        # Module prefix is applied in Metadata.final_title after precedence
        # resolution, so it is not applied here.

        # Build title dict for media with all three keys.
        title_dict = self.build_title_dict(
            filename_title=filename_title,
            extracted=extracted,
            generated_title=generated_title,
        )

        # On completion: update to final state with all fields.
        # For media files, description is left blank and the LLM-generated
        # summary is stored in the summary field instead.
        metadata_dict.update(
            {
                "title": title_dict,
                "description": "",  # Media: use summary instead
                "summary": summary,
                "date": file_date,
                "mtime": stat_result.st_mtime,
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
            description="",  # Media: use summary instead
            summary=summary,
            filename=filename,
            filetype=cast(MediaFiletype, filetype),
            date=file_date,
            mtime=stat_result.st_mtime,
            size=file_size,
            duration=duration,
            creation_time=creation_time_str,
            content=transcript,
            hash=metadata_dict.get("hash"),
            transcribe=StepMap(**transcribe_step),
            summarize=StepMap(**summarize_step),
            generate_title=StepMap(**generate_title_step),
        )
        metadata._source_filepath = filepath
        return metadata
