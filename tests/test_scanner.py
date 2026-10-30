"""Tests for the file scanner."""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from unittest import mock

import pytest

from focus_feed.scanner import ScannedFile, Scanner


class TestScannerConstruction:
    """Tests for Scanner initialization."""

    @pytest.mark.parametrize("data_dir_arg", [None, str])
    def test_init_with_data_dir(self, tmp_path, data_dir_arg):
        """Scanner should accept a data_dir argument (Path or string) and store as absolute."""
        dir_arg = tmp_path if data_dir_arg is None else str(tmp_path)
        scanner = Scanner(data_dir=dir_arg)
        assert scanner.data_dir == tmp_path.resolve()

    def test_init_requires_data_dir(self):
        """Scanner should require data_dir to be provided (no config fallback)."""
        with pytest.raises(TypeError):
            Scanner()  # type: ignore[call-arg]


class TestScanDirectory:
    """Tests for scan_directory() core functionality."""

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """An empty data directory should return an empty list."""
        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()
        assert result == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        """A nonexistent directory should return an empty list with a warning."""
        scanner = Scanner(data_dir=tmp_path / "nonexistent")
        result = scanner.scan_directory()
        assert result == []

    def test_nonexistent_directory_logs_warning(self, tmp_path, caplog):
        """Scanning a nonexistent directory should log a warning."""
        scanner = Scanner(data_dir=tmp_path / "nonexistent")
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            scanner.scan_directory()
        assert any(
            "does not exist" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_finds_files_without_json(self, tmp_path):
        """Source files without JSON metadata should be returned with correct fields."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")
        (tmp_path / "document.pdf").write_bytes(b"pdf data")
        (tmp_path / "article.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 3
        filetypes = {f.filetype for f in result}
        assert filetypes == {"mp4", "pdf", "html"}
        # Verify filename and filetype fields
        by_name = {f.filename: f for f in result}
        assert by_name["lecture.mp4"].filetype == "mp4"
        assert by_name["document.pdf"].filetype == "pdf"
        assert by_name["article.html"].filetype == "html"

    def test_skips_files_with_existing_json(self, tmp_path):
        """Source files with existing JSON metadata should not be returned."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    @pytest.mark.parametrize(
        "json_newer,should_return",
        [
            (False, True),  # Source newer than JSON → needs processing
            (True, False),  # JSON newer than source → skip
        ],
    )
    def test_json_freshness_check(self, tmp_path, json_newer, should_return):
        """Source newer than JSON should trigger reprocessing; JSON newer should skip."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        old_time = time.time() - 3600
        new_time = time.time()
        if json_newer:
            os.utime(source, (old_time, old_time))
            os.utime(json_file, (new_time, new_time))
        else:
            os.utime(json_file, (old_time, old_time))
            os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        if should_return:
            assert len(result) == 1
            assert result[0].filename == "lecture.mp4"
        else:
            assert result == []

    def test_recursive_directory_walk(self, tmp_path):
        """Scanner should find files in nested subdirectories."""
        (tmp_path / "Course1").mkdir()
        (tmp_path / "Course1" / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "Course2").mkdir()
        (tmp_path / "Course2" / "subdir").mkdir()
        (tmp_path / "Course2" / "subdir" / "doc.pdf").write_bytes(b"pdf")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 2
        filepaths = {f.filepath for f in result}
        assert "Course1/lecture.mp4" in filepaths
        assert "Course2/subdir/doc.pdf" in filepaths

    def test_returns_absolute_and_relative_paths(self, tmp_path):
        """Result should contain both absolute and relative paths."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        entry = result[0]
        assert entry.absolute_path == str(source.resolve())
        assert entry.filepath == "lecture.mp4"

    @pytest.mark.parametrize(
        "filename,should_ignore",
        [
            ("index.html", True),
            ("INDEX.HTML", True),
            ("podcast.xml", True),
            ("articles.xml", True),
            ("feed.json", True),
            ("lecture.mp4", False),
        ],
    )
    def test_ignores_generated_and_config_files(self, tmp_path, filename, should_ignore):
        """Generated feed files and config files should be ignored."""
        (tmp_path / filename).write_bytes(b"data")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        if should_ignore:
            assert result == []
        else:
            assert len(result) == 1
            assert result[0].filename == filename

    def test_results_sorted_by_filepath(self, tmp_path):
        """Results should be sorted by relative filepath for determinism."""
        (tmp_path / "zebra.mp4").write_bytes(b"video")
        (tmp_path / "alpha.pdf").write_bytes(b"pdf")
        (tmp_path / "mid").mkdir()
        (tmp_path / "mid" / "file.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        filepaths = [f.filepath for f in result]
        assert filepaths == sorted(filepaths)


class TestSpecialCharacterWarnings:
    """Tests for filename special character validation."""

    @pytest.mark.parametrize(
        "filename",
        [
            "file<test>.mp4",
            "file&name.mp4",
            'file"name.mp4',
            "file'name.mp4",
            " leading.mp4",
            "trailing.mp4 ",
        ],
    )
    def test_warns_on_special_characters(self, tmp_path, caplog, filename):
        """Filenames with special characters should log a warning."""
        (tmp_path / filename).write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            scanner.scan_directory()

        assert any(
            "special characters" in record.message or "leading/trailing" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_no_warning_for_clean_filenames(self, tmp_path, caplog):
        """Clean filenames should not trigger special character warnings."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "my-document.pdf").write_bytes(b"pdf")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            scanner.scan_directory()

        assert not any(
            "special characters" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )
        assert not any(
            "leading/trailing" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )


class TestOrphanedJsonDetection:
    """Tests for orphaned JSON file detection."""

    @pytest.mark.parametrize(
        "json_filename,json_content,should_warn",
        [
            ("orphan.mp4.json", '{"transcribe": {"status": "complete"}}', True),
            ("config.json", '{"extract": {"status": "complete"}}', True),
            ("feed.json", '{"title": "My Feed"}', False),
        ],
    )
    def test_orphaned_json_detection(
        self, tmp_path, caplog, json_filename, json_content, should_warn
    ):
        """Orphaned JSON files should be detected and warned about (except feed.json)."""
        (tmp_path / json_filename).write_text(json_content)

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_directory()

        if should_warn:
            assert any(
                "Orphaned JSON" in record.message
                for record in caplog.records
                if record.levelno == logging.WARNING
            )
            # Warning should include the filename
            assert any(
                json_filename in record.message
                for record in caplog.records
                if record.levelno == logging.WARNING
            )
        else:
            assert not any(
                "Orphaned JSON" in record.message
                for record in caplog.records
                if record.levelno == logging.WARNING
            )
        assert result == []

    def test_json_with_source_not_orphaned(self, tmp_path, caplog):
        """A JSON file with a matching source should not be flagged as orphaned."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_directory()

        assert result == []
        assert not any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_orphaned_json_in_subdirectory(self, tmp_path, caplog):
        """Orphaned JSON in subdirectories should be detected."""
        (tmp_path / "Course1").mkdir()
        (tmp_path / "Course1" / "orphan.pdf.json").write_text('{"extract": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="focus_feed"):
            result = scanner.scan_directory()

        assert result == []
        assert any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )


class TestScanDirectoryErrorHandling:
    """Tests for error handling during directory scanning."""

    @pytest.mark.parametrize(
        "exception_cls",
        [
            PermissionError,
            OSError,
        ],
    )
    def test_handles_filesystem_error_gracefully(self, tmp_path, caplog, exception_cls):
        """Scanner should not crash on filesystem errors for individual files."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)

        original_stat = Path.stat

        def mock_stat(self, *args, **kwargs):
            if self == source:
                raise exception_cls("Filesystem error")
            return original_stat(self, *args, **kwargs)

        with (
            mock.patch.object(Path, "stat", mock_stat),
            caplog.at_level(logging.WARNING, logger="focus_feed"),
        ):
            result = scanner.scan_directory()

        # File should still be returned (treated as needing processing).
        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"


class TestScannedFileDataclass:
    """Tests for the ScannedFile dataclass."""

    @pytest.mark.parametrize(
        "with_optional,expected_stage,expected_json_exists",
        [
            (False, None, False),
            (True, "transcribing", True),
        ],
    )
    def test_scanned_file_fields(self, with_optional, expected_stage, expected_json_exists):
        """ScannedFile should be immutable and have all required fields."""
        kwargs: dict[str, Any] = {
            "filepath": "videos/lecture.mp4",
            "absolute_path": "/data/videos/lecture.mp4",
            "filename": "lecture.mp4",
            "filetype": "mp4",
        }
        if with_optional:
            kwargs["current_stage"] = expected_stage
            kwargs["json_exists"] = expected_json_exists
        entry = ScannedFile(**kwargs)
        with pytest.raises(AttributeError):
            entry.filepath = "other.mp4"  # type: ignore[misc]
        assert entry.filepath == "videos/lecture.mp4"
        assert entry.absolute_path == "/data/videos/lecture.mp4"
        assert entry.filename == "lecture.mp4"
        assert entry.filetype == "mp4"
        assert entry.current_stage == expected_stage
        assert entry.json_exists is expected_json_exists


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
            ("pdf", '{"extract": {"status": "complete"}, "summarize": {"status": "complete"}}'),
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
