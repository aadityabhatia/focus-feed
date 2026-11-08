"""Tests for the file scanner."""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import pytest

from focus_feed.scanner import (
    Scanner,
)
from focus_feed.services.extractors import DoclingExtractor


class TestIntegrationScenarios:
    """Integration-style tests for realistic directory structures."""

    def test_mixed_processed_unprocessed_files(self, tmp_path):
        """Scanner should return only unprocessed files in a mixed directory."""
        # Processed file (has JSON with all steps complete).
        (tmp_path / "processed.mp4").write_bytes(b"video")
        (tmp_path / "processed.mp4.json").write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        # Unprocessed file (no JSON).
        (tmp_path / "unprocessed.pdf").write_bytes(b"pdf")

        # Orphaned JSON.
        (tmp_path / "orphan.pdf.json").write_text('{"extract": {"status": "complete"}}')

        # Nested unprocessed.
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        filenames = {f.filename for f in result}
        assert filenames == {"unprocessed.pdf", "nested.html"}

    @pytest.mark.parametrize(
        "json_content,should_return,expected_filename",
        [
            (
                '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}',
                False,
                None,
            ),
            (
                '{"transcribe": {"status": "in_progress"}, "summarize": {"status": null}}',
                True,
                "lecture.mp4",
            ),
            ('{"title": "Some Title"}', True, "lecture.mp4"),
            ("{invalid json content", False, None),
        ],
    )
    def test_json_status_and_freshness(
        self, tmp_path, json_content, should_return, expected_filename
    ):
        """Test various JSON status and freshness scenarios."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        if should_return:
            assert len(result) == 1
            assert result[0].filename == expected_filename
        else:
            assert result == []

    @pytest.mark.parametrize(
        "source_newer,should_return",
        [
            (True, True),  # Source newer than JSON → needs processing
            (False, False),  # JSON newer than or equal to source → skip
        ],
    )
    def test_freshness_check(self, tmp_path, source_newer, should_return):
        """Source newer than JSON should trigger reprocessing; JSON newer should skip."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        old_time = time.time() - 100
        new_time = time.time()
        if source_newer:
            os.utime(json_file, (old_time, old_time))
            os.utime(source, (new_time, new_time))
        else:
            os.utime(source, (old_time, old_time))
            os.utime(json_file, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        if should_return:
            assert len(result) == 1
        else:
            assert result == []

    def test_stored_mtime_detects_source_change_even_when_json_is_newer(self, tmp_path):
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        old_mtime = time.time() - 100
        os.utime(source, (old_mtime, old_mtime))
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            json.dumps(
                {
                    "mtime": old_mtime,
                    "transcribe": {"status": "complete"},
                    "summarize": {"status": "complete"},
                }
            )
        )
        source.write_bytes(b"updated video data")
        new_mtime = time.time()
        os.utime(source, (new_mtime, new_mtime))
        os.utime(json_file, (new_mtime + 100, new_mtime + 100))

        result = Scanner(tmp_path).scan_directory()

        assert len(result) == 1

    def test_invalid_stored_mtime_falls_back_to_json_timestamp(self, tmp_path):
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"mtime": "invalid", "transcribe": {"status": "complete"}, '
            '"summarize": {"status": "complete"}}'
        )
        old_time = time.time() - 100
        new_time = time.time()
        os.utime(source, (old_time, old_time))
        os.utime(json_file, (new_time, new_time))

        assert Scanner(tmp_path).scan_directory() == []


class TestStepStatusBasedScanning:
    """Tests for step status-based scanning logic."""

    @pytest.mark.parametrize(
        "filetype,step_map",
        [
            ("mp4", {"status": "failed", "error": "transcription failed"}),
            ("mp3", {"status": "failed", "error": "transcription failed"}),
            ("pdf", {"status": "failed", "error": "extraction failed"}),
        ],
    )
    def test_step_error_with_zero_output_returns_file_for_processing(
        self, tmp_path, filetype, step_map
    ):
        source = tmp_path / f"lecture.{filetype}"
        source.write_bytes(b"data")
        step_name = "extract" if filetype == "pdf" else "transcribe"
        metadata = {step_name: {**step_map, "output_size": 0}, "error": "step failed"}
        (tmp_path / f"lecture.{filetype}.json").write_text(json.dumps(metadata))

        result = Scanner(tmp_path).scan_directory()

        assert len(result) == 1
        assert result[0].filename == f"lecture.{filetype}"

    def test_step_error_with_nonzero_output_does_not_override_top_level_error(self, tmp_path):
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"data")
        metadata = {
            "transcribe": {
                "status": "failed",
                "error": "transcription failed",
                "output_size": 12,
            },
            "error": "step failed",
        }
        (tmp_path / "lecture.mp4.json").write_text(json.dumps(metadata))

        assert Scanner(tmp_path).scan_directory() == []

    @pytest.mark.parametrize(
        "filetype,step_field",
        [
            ("mp4", "transcribe"),
            ("mp4", "summarize"),
            ("pdf", "extract"),
            ("pdf", "summarize"),
            ("html", "extract"),
            ("html", "summarize"),
        ],
    )
    def test_incomplete_step_error_returns_file_for_processing(
        self, tmp_path, filetype, step_field
    ):
        """An error in an incomplete applicable step prevents feed eligibility."""
        source = tmp_path / f"lecture.{filetype}"
        source.write_bytes(b"data")
        metadata: dict[str, Any] = {
            "transcribe": {"status": "complete"},
            "extract": {"status": "complete"},
            "summarize": {"status": "complete"},
        }
        metadata[step_field] = {"status": "failed", "error": "step failed"}
        (tmp_path / f"lecture.{filetype}.json").write_text(json.dumps(metadata))

        result = Scanner(tmp_path).scan_directory()

        assert len(result) == 1
        assert result[0].filename == f"lecture.{filetype}"

    @pytest.mark.parametrize(
        "filetype", ["mp4", "pdf", "html", "docx", "doc", "pptx", "ppt", "xlsx", "xls"]
    )
    def test_non_dict_step_maps_are_safe(self, tmp_path, filetype):
        """Non-dict step maps do not cause error checking to fail."""
        source = tmp_path / f"lecture.{filetype}"
        source.write_bytes(b"data")
        metadata: dict[str, Any] = {
            "transcribe": {"status": "complete"},
            "extract": {"status": "complete"},
            "summarize": {"status": "complete"},
        }
        metadata["transcribe" if filetype == "mp4" else "extract"] = "not a step map"
        (tmp_path / f"lecture.{filetype}.json").write_text(json.dumps(metadata))

        # A non-dict map is not treated as an error map. The existing status
        # rules still determine eligibility (the substituted required map is
        # incomplete, so the file is returned for processing).
        result = Scanner(tmp_path).scan_directory()

        assert len(result) == 1
        assert result[0].filename == f"lecture.{filetype}"

    @pytest.mark.parametrize(
        "step_field,step_status,json_content",
        [
            (
                "transcribe",
                "in_progress",
                '{"transcribe": {"status": "in_progress"}, "summarize": {"status": null}}',
            ),
            (
                "summarize",
                "in_progress",
                '{"transcribe": {"status": "complete"}, "summarize": {"status": "in_progress"}}',
            ),
            ("extract", "in_progress", '{"extract": {"status": "in_progress"}}'),
        ],
    )
    def test_in_progress_status_returned(self, tmp_path, step_field, step_status, json_content):
        """Files with in_progress step status should be returned for resuming."""
        ext = "mp4" if step_field != "extract" else "pdf"
        source = tmp_path / f"lecture.{ext}"
        source.write_bytes(b"video data" if step_field != "extract" else b"pdf data")
        json_file = tmp_path / f"lecture.{ext}.json"
        json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == f"lecture.{ext}"

    @pytest.mark.parametrize(
        "filetype,json_content",
        [
            ("mp4", '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'),
            *[
                (
                    extension.removeprefix("."),
                    '{"extract": {"status": "complete"}, "summarize": {"status": "complete"}}',
                )
                for extension in sorted(DoclingExtractor.SUPPORTED_EXTENSIONS)
            ],
            ("html", '{"extract": {"status": "complete"}, "summarize": {"status": "complete"}}'),
        ],
    )
    def test_complete_files_skipped(self, tmp_path, filetype, json_content):
        """Files with all steps complete should be skipped."""
        source = tmp_path / f"lecture.{filetype}"
        source.write_bytes(b"data")
        json_file = tmp_path / f"lecture.{filetype}.json"
        json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    @pytest.mark.parametrize(
        "filetype,json_content",
        [
            ("pdf", '{"extract": {"status": "complete"}}'),
            ("html", '{"extract": {"status": "complete"}}'),
        ],
    )
    def test_pdf_html_without_summarize_returned(self, tmp_path, filetype, json_content):
        """PDF/HTML files with extract complete but no summarize should be returned.

        Files that have extract complete but no summarize step should be
        returned for processing.
        """
        source = tmp_path / f"lecture.{filetype}"
        source.write_bytes(b"data")
        json_file = tmp_path / f"lecture.{filetype}.json"
        json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == f"lecture.{filetype}"

    def test_partial_mp4_returned(self, tmp_path):
        """MP4 files with only transcribe complete but not summarize should be returned."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": null}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_error_returned_when_source_is_newer(self, tmp_path):
        """Files with error should be returned if source is newer.

        This allows retrying after the user fixes the source file.
        """
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}, "error": "old error"}')

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_complete_mp4_returned_when_source_newer(self, tmp_path):
        """MP4 files with all steps complete should be returned if source is newer."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1

    def test_missing_step_statuses_skipped(self, tmp_path):
        """Files with JSON missing status should be skipped (conservative)."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"title": "Some Title"}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        # No status dict → not feed-eligible → needs processing
        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_invalid_json_skipped_conservatively(self, tmp_path, caplog):
        """Invalid JSON metadata should be skipped (conservative) if source not newer."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text("{invalid json content")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_directory()

        assert result == []
        assert any(
            "Could not parse JSON metadata" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_invalid_json_returned_when_source_newer(self, tmp_path, caplog):
        """Invalid JSON should still trigger reprocessing if source is newer."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text("{invalid json content")

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"
        assert result[0].json_exists is True
        assert result[0].current_stage is None


class TestJsonExistsField:
    """Tests for the json_exists field on ScannedFile."""

    @pytest.mark.parametrize(
        "json_content,should_exist",
        [
            ('{"transcribe": {"status": "in_progress"}}', True),
            (None, False),
        ],
    )
    def test_json_exists_field(self, tmp_path, json_content, should_exist):
        """json_exists should correctly reflect whether JSON metadata exists."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        if json_content:
            json_file = tmp_path / "lecture.mp4.json"
            json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].json_exists is should_exist

    def test_json_exists_true_when_source_newer_than_error_json(self, tmp_path):
        """json_exists should be True even when returning due to freshness."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}, "error": "some error"}')

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].json_exists is True


class TestStageTracking:
    """Tests for stage tracking via current_stage field."""

    @pytest.mark.parametrize(
        "json_content,expected_stage",
        [
            (
                '{"transcribe": {"status": "in_progress"}, '
                '"summarize": {"status": null}, "stage": "transcribing"}',
                "transcribing",
            ),
            (
                '{"extract": {"status": "in_progress"}, "stage": "extracting_metadata"}',
                "extracting_metadata",
            ),
            ('{"status": {"transcribe": "in_progress"}, "stage": null}', None),
            ('{"transcribe": {"status": "in_progress"}, "stage": 123}', None),
            (
                '{"status": {"transcribe": "in_progress"}, "current_stage": "legacy_stage_value"}',
                "legacy_stage_value",
            ),
            (
                '{"status": {"transcribe": "in_progress"}, '
                '"stage": "new_field", "current_stage": "old_field"}',
                "new_field",
            ),
        ],
    )
    def test_stage_tracking(self, tmp_path, json_content, expected_stage):
        """current_stage should be correctly extracted from JSON metadata."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == expected_stage

    def test_current_stage_none_when_not_in_json(self, tmp_path):
        """current_stage should be None when not present in JSON metadata or no JSON exists."""
        # No JSON file
        (tmp_path / "lecture.mp4").write_bytes(b"video data")
        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()
        assert len(result) == 1
        assert result[0].current_stage is None
        assert result[0].json_exists is False

        # JSON without stage field
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}}')
        result2 = scanner.scan_directory()
        assert len(result2) == 1
        assert result2[0].current_stage is None

    def test_source_newer_than_complete_json_includes_stage(self, tmp_path):
        """When source is newer, stage from JSON should still be included."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, '
            '"summarize": {"status": "complete"}, "stage": "summarizing"}'
        )

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "summarizing"

    @pytest.mark.parametrize(
        "json_content,should_skip",
        [
            ('{"transcribe_status": "in_progress", "summarize_status": null}', False),
            ('{"transcribe_status": "complete", "summarize_status": "complete"}', True),
            ('{"extract_status": "complete"}', False),  # Missing summarize_status
            ('{"extract_status": "complete", "summarize_status": "complete"}', True),
        ],
    )
    def test_legacy_status_fields(self, tmp_path, json_content, should_skip):
        """Scanner should handle old JSON files with individual status fields."""
        filetype = "pdf" if "extract_status" in json_content else "mp4"
        source = tmp_path / f"lecture.{filetype}"
        source.write_bytes(b"data")
        json_file = tmp_path / f"lecture.{filetype}.json"
        json_file.write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        if should_skip:
            assert result == []
        else:
            assert len(result) == 1
            assert result[0].filename == f"lecture.{filetype}"


class TestScanPaths:
    """Tests for the scan_paths() method."""

    def test_empty_paths_returns_empty_list(self, tmp_path):
        """An empty paths list should return an empty list."""
        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([])
        assert result == []

    def test_single_file_needing_processing(self, tmp_path):
        """A single source file without JSON should be returned."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([source])

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"
        assert result[0].filetype == "mp4"
        assert result[0].filepath == "lecture.mp4"
        assert result[0].absolute_path == str(source.resolve())
        assert result[0].json_exists is False
        assert result[0].current_stage is None

    def test_single_file_with_complete_json_skipped(self, tmp_path):
        """A source file with complete JSON should be skipped."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([source])

        assert result == []

    def test_single_directory(self, tmp_path):
        """A single directory should be scanned recursively."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "subdir" / "doc.pdf").write_bytes(b"pdf")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path])

        assert len(result) == 2
        filepaths = {f.filepath for f in result}
        assert "lecture.mp4" in filepaths
        assert "subdir/doc.pdf" in filepaths

    def test_mixed_files_and_directories(self, tmp_path):
        """Mix of file and directory paths should be scanned together."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "subdir" / "doc.pdf").write_bytes(b"pdf")
        (tmp_path / "article.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path / "lecture.mp4", tmp_path / "subdir"])

        assert len(result) == 2
        filenames = {f.filename for f in result}
        assert filenames == {"lecture.mp4", "doc.pdf"}

    def test_filepaths_relative_to_common_parent(self, tmp_path):
        """Filepaths should be relative to the common parent of all input paths."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "subdir" / "doc.pdf").write_bytes(b"pdf")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path / "lecture.mp4", tmp_path / "subdir" / "doc.pdf"])

        assert len(result) == 2
        filepaths = {f.filepath for f in result}
        assert "lecture.mp4" in filepaths
        assert "subdir/doc.pdf" in filepaths

    def test_common_parent_is_subdirectory(self, tmp_path):
        """When all paths are under a subdirectory, filepaths should be relative to that subdir."""
        (tmp_path / "Course1").mkdir()
        (tmp_path / "Course1" / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "Course1" / "doc.pdf").write_bytes(b"pdf")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path / "Course1"])

        assert len(result) == 2
        filepaths = {f.filepath for f in result}
        assert "lecture.mp4" in filepaths
        assert "doc.pdf" in filepaths

    def test_nonexistent_path_skipped(self, tmp_path, caplog):
        """Nonexistent paths should be skipped with a warning."""
        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_paths([tmp_path / "nonexistent.mp4"])

        assert result == []
        assert any(
            "does not exist" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_unsupported_file_skipped(self, tmp_path):
        """Unsupported file types should be skipped."""
        (tmp_path / "data.txt").write_text("some text")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path / "data.txt"])

        assert result == []

    def test_ignored_filenames_skipped(self, tmp_path):
        """Ignored filenames (feed.json, index.html, etc.) should be skipped."""
        (tmp_path / "feed.json").write_text('{"title": "Feed"}')
        (tmp_path / "index.html").write_text("<html></html>")
        (tmp_path / "lecture.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path])

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    @pytest.mark.parametrize("filename", [".hidden.mp4", "video.mp4.tmp", "INDEX.HTML"])
    def test_ignored_files_skipped_in_scan_paths(self, tmp_path, filename):
        ignored = tmp_path / filename
        ignored.write_bytes(b"data")

        assert Scanner(tmp_path).scan_paths([ignored]) == []

    @pytest.mark.parametrize("filename", [".hidden.mp4", "video.mp4.tmp"])
    def test_ignored_files_skipped_in_directory_scan(self, tmp_path, filename):
        (tmp_path / filename).write_bytes(b"data")

        assert Scanner(tmp_path).scan_directory() == []

    def test_results_sorted_by_filepath(self, tmp_path):
        """Results should be sorted by relative filepath."""
        (tmp_path / "zebra.mp4").write_bytes(b"video")
        (tmp_path / "alpha.pdf").write_bytes(b"pdf")
        (tmp_path / "mid").mkdir()
        (tmp_path / "mid" / "file.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path])

        filepaths = [f.filepath for f in result]
        assert filepaths == sorted(filepaths)

    def test_json_status_evaluated_for_scanned_files(self, tmp_path):
        """scan_paths should evaluate JSON status like scan_directory."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "in_progress"}, "summarize": {"status": null}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([source])

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"
        assert result[0].json_exists is True

    def test_orphaned_json_detected_in_directory_scan(self, tmp_path, caplog):
        """Orphaned JSON files should be detected during directory scanning."""
        (tmp_path / "orphan.pdf.json").write_text('{"extract": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_paths([tmp_path])

        assert result == []
        assert any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_current_stage_preserved(self, tmp_path):
        """current_stage should be preserved from JSON metadata."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "in_progress"}, '
            '"summarize": {"status": null}, "stage": "transcribing"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([source])

        assert len(result) == 1
        assert result[0].current_stage == "transcribing"

    def test_multiple_files_in_directory(self, tmp_path):
        """Multiple source files in a directory should all be returned."""
        (tmp_path / "a.mp4").write_bytes(b"video")
        (tmp_path / "b.pdf").write_bytes(b"pdf")
        (tmp_path / "c.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([tmp_path])

        assert len(result) == 3
        filenames = {f.filename for f in result}
        assert filenames == {"a.mp4", "b.pdf", "c.html"}

    def test_scan_paths_with_relative_paths(self, tmp_path, monkeypatch):
        """scan_paths should work with relative paths."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        monkeypatch.chdir(tmp_path)
        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([Path("lecture.mp4")])

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"
        assert result[0].filepath == "lecture.mp4"

    def test_scan_paths_mixed_relative_and_absolute(self, tmp_path, monkeypatch):
        """scan_paths should handle a mix of relative and absolute paths."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "subdir" / "doc.pdf").write_bytes(b"pdf")

        monkeypatch.chdir(tmp_path)
        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_paths([Path("lecture.mp4"), tmp_path / "subdir"])

        assert len(result) == 2
        filenames = {f.filename for f in result}
        assert filenames == {"lecture.mp4", "doc.pdf"}
