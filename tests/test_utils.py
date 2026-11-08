"""Tests for shared utility and metadata I/O helpers."""

import json
from math import isclose
from pathlib import Path

from focus_feed.metadata_io import (
    is_dict_feed_eligible,
    is_orphaned_json,
    is_step_complete,
    read_metadata_json,
)
from focus_feed.utils import (
    SPECIAL_CHAR_PATTERN,
    clean_title,
    is_audio_filetype,
    is_video_filetype,
    metadata_sidecar_path,
    now_timestamp,
    sanitize_content,
    should_ignore_filename,
)


def test_now_timestamp_has_millisecond_precision() -> None:
    timestamp = now_timestamp()
    assert isinstance(timestamp, float)
    assert isclose(timestamp * 1000, round(timestamp * 1000))


def test_clean_title() -> None:
    assert clean_title(None) is None
    assert clean_title("lecture_notes") == "lecture notes"
    assert clean_title("plain text") == "plain text"


def test_sanitize_content() -> None:
    assert sanitize_content("a\x00b\x08c\x0bd\x7f") == "abcd"
    assert sanitize_content("tab\tnewline\nreturn\r") == "tab\tnewline\nreturn\r"
    assert sanitize_content(None) is None


def test_metadata_sidecar_path_accepts_string_and_path() -> None:
    assert metadata_sidecar_path("course/lecture.mp4") == Path("course/lecture.mp4.json")
    assert metadata_sidecar_path(Path("course/lecture.mp4")) == Path("course/lecture.mp4.json")
    assert SPECIAL_CHAR_PATTERN.search('course<&".mp4')
    assert is_audio_filetype(".MP3")
    assert not is_audio_filetype("mp4")
    assert is_video_filetype(".MKV")
    assert is_video_filetype("Mp4")
    assert not is_video_filetype("mp3")
    assert should_ignore_filename("PODCAST.XML")
    assert should_ignore_filename(".hidden.mp4")
    assert should_ignore_filename("temporary.TMP")
    assert not should_ignore_filename("lecture.mp4")


def test_read_metadata_json_valid_dict(tmp_path: Path) -> None:
    json_path = tmp_path / "metadata.json"
    json_path.write_text(json.dumps({"filename": "lecture.mp4"}), encoding="utf-8")
    assert read_metadata_json(json_path) == {"filename": "lecture.mp4"}


def test_read_metadata_json_invalid_json(tmp_path: Path) -> None:
    json_path = tmp_path / "invalid.json"
    json_path.write_text("{invalid", encoding="utf-8")
    assert read_metadata_json(json_path) is None


def test_read_metadata_json_missing_file(tmp_path: Path) -> None:
    assert read_metadata_json(tmp_path / "missing.json") is None


def test_read_metadata_json_non_dict(tmp_path: Path) -> None:
    json_path = tmp_path / "list.json"
    json_path.write_text("[]", encoding="utf-8")
    assert read_metadata_json(json_path) is None


def test_is_orphaned_json(tmp_path: Path) -> None:
    assert not is_orphaned_json(tmp_path / "feed.json")

    source = tmp_path / "lecture.mp4"
    source.touch()
    assert not is_orphaned_json(tmp_path / "lecture.mp4.json")
    assert is_orphaned_json(tmp_path / "missing.pdf.json")
    assert is_orphaned_json(tmp_path / "notes.txt.json")


def test_is_step_complete() -> None:
    assert is_step_complete(
        {"summarize": {"status": "complete"}, "summary": "Summary text"}, "summarize"
    )
    assert not is_step_complete({"summarize": {"status": "complete"}}, "summarize")
    assert not is_step_complete(
        {"summarize": {"status": "complete"}, "summary": "   "}, "summarize"
    )
    assert not is_step_complete(
        {"summarize": {"status": "in_progress"}, "summary": "Summary text"}, "summarize"
    )
    assert not is_step_complete({}, "summarize")

    assert is_step_complete(
        {"extract": {"status": "complete"}, "content": "Extracted text"}, "extract"
    )
    assert not is_step_complete({"extract": {"status": "complete"}}, "extract")
    assert not is_step_complete({"extract": {"status": "complete"}, "content": "  "}, "extract")
    assert not is_step_complete(
        {"extract": {"status": "in_progress"}, "content": "Extracted text"}, "extract"
    )


def test_is_dict_feed_eligible_media() -> None:
    complete = {
        "transcribe": {"status": "complete"},
        "summarize": {"status": "complete"},
    }
    assert is_dict_feed_eligible(complete, "MP4")
    assert not is_dict_feed_eligible({**complete, "error": "failed"}, "mp4")
    assert not is_dict_feed_eligible(
        {"transcribe": {"status": "complete"}, "summarize": {"status": "in_progress"}},
        "mp4",
    )
    assert is_dict_feed_eligible(
        {"transcribe_status": "complete", "summarize_status": "complete"}, ".mp3"
    )


def test_is_dict_feed_eligible_document() -> None:
    complete = {
        "extract": {"status": "complete"},
        "summarize": {"status": "complete"},
    }
    assert is_dict_feed_eligible(complete, "pdf")
    assert not is_dict_feed_eligible(
        {"extract": {"status": "complete"}, "summarize": {"status": "in_progress"}},
        "html",
    )
    assert is_dict_feed_eligible(
        {"extract_status": "complete", "summarize_status": "complete"}, "docx"
    )
    assert not is_dict_feed_eligible(complete, "txt")
    assert not is_dict_feed_eligible(complete, 3)  # type: ignore[arg-type]
