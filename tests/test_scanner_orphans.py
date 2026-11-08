"""Tests for the file scanner."""

import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from focus_feed.scanner import (
    ScannedFile,
    Scanner,
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
