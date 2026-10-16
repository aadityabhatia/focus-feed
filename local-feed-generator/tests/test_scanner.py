"""Tests for the file scanner."""

import logging
import os
from pathlib import Path
from unittest import mock

import pytest

from local_feed_generator.scanner import ScannedFile, Scanner


class TestScannerConstruction:
    """Tests for Scanner initialization."""

    def test_init_with_explicit_data_dir(self, tmp_path):
        """Scanner should accept a data_dir argument and store it as absolute Path."""
        scanner = Scanner(data_dir=tmp_path)
        assert scanner.data_dir == tmp_path.resolve()

    def test_init_with_string_data_dir(self, tmp_path):
        """Scanner should accept a string path for data_dir."""
        scanner = Scanner(data_dir=str(tmp_path))
        assert scanner.data_dir == tmp_path.resolve()

    def test_init_data_dir_is_absolute(self, tmp_path):
        """Scanner should store data_dir as an absolute path."""
        scanner = Scanner(data_dir=tmp_path)
        assert scanner.data_dir.is_absolute()

    def test_init_uses_config_data_dir_when_none(self, tmp_path):
        """Scanner should fall back to config DATA_DIR when data_dir is None."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": str(tmp_path),
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_SYSTEM_PROMPT": "Test prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            from local_feed_generator.config import reload_config

            reload_config()
            scanner = Scanner()
            assert scanner.data_dir == tmp_path.resolve()


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
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()
        assert any(
            "does not exist" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_path_is_file_not_dir_returns_empty(self, tmp_path):
        """A path pointing to a file (not a directory) should return empty."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        scanner = Scanner(data_dir=file_path)
        result = scanner.scan_directory()
        assert result == []

    def test_finds_files_without_json(self, tmp_path):
        """Source files without JSON metadata should be returned."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")
        (tmp_path / "document.pdf").write_bytes(b"pdf data")
        (tmp_path / "article.html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 3
        filetypes = {f.filetype for f in result}
        assert filetypes == {"mp4", "pdf", "html"}

    def test_skips_files_with_existing_json(self, tmp_path):
        """Source files with existing JSON metadata should not be returned."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    def test_detects_outdated_json(self, tmp_path):
        """Files whose source was modified after JSON should be returned."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        # Make the JSON older than the source.
        import os
        import time

        old_time = time.time() - 3600
        os.utime(json_file, (old_time, old_time))
        # Ensure source is newer.
        new_time = time.time()
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_fresh_json_skips_processing(self, tmp_path):
        """Files with JSON newer than source should not be returned."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        # Make the JSON newer than the source.
        import os
        import time

        old_time = time.time() - 3600
        os.utime(source, (old_time, old_time))
        new_time = time.time()
        os.utime(json_file, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

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

    def test_returns_filename_and_filetype(self, tmp_path):
        """Result should include filename and filetype fields."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "notes.PDF").write_bytes(b"pdf")
        (tmp_path / "page.HTML").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 3
        by_name = {f.filename: f for f in result}
        assert by_name["lecture.mp4"].filetype == "mp4"
        assert by_name["notes.PDF"].filetype == "pdf"
        assert by_name["page.HTML"].filetype == "html"

    def test_case_insensitive_extensions(self, tmp_path):
        """Scanner should match extensions case-insensitively."""
        (tmp_path / "video.MP4").write_bytes(b"video")
        (tmp_path / "doc.Pdf").write_bytes(b"pdf")
        (tmp_path / "page.Html").write_text("<html></html>")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 3
        filetypes = {f.filetype for f in result}
        assert filetypes == {"mp4", "pdf", "html"}

    def test_ignores_unsupported_extensions(self, tmp_path):
        """Files with unsupported extensions should not be returned."""
        (tmp_path / "image.jpg").write_bytes(b"image")
        (tmp_path / "data.txt").write_text("text")
        (tmp_path / "video.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "video.mp4"

    def test_ignores_feed_xml_files(self, tmp_path):
        """Generated feed XML files should not be treated as source files."""
        (tmp_path / "podcast.xml").write_text("<rss></rss>")
        (tmp_path / "articles.xml").write_text("<rss></rss>")
        (tmp_path / "video.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "video.mp4"

    def test_ignores_index_html_files(self, tmp_path):
        """index.html files should not be treated as source files."""
        (tmp_path / "index.html").write_text("<html></html>")
        (tmp_path / "lecture.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_ignores_index_html_case_insensitive(self, tmp_path):
        """index.html should be ignored regardless of case."""
        (tmp_path / "INDEX.HTML").write_text("<html></html>")
        (tmp_path / "lecture.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_ignores_index_html_in_subdirectory(self, tmp_path):
        """index.html in subdirectories should also be ignored."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "index.html").write_text("<html></html>")
        (tmp_path / "subdir" / "lecture.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_ignores_index_html_with_orphaned_json(self, tmp_path):
        """index.html.json should not be treated as orphaned JSON."""
        (tmp_path / "index.html").write_text("<html></html>")
        (tmp_path / "index.html.json").write_text("{}")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

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

    def test_warns_on_xml_special_chars(self, tmp_path, caplog):
        """Filenames with XML special characters should log a warning."""
        (tmp_path / "file<test>.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "special characters" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_warns_on_ampersand(self, tmp_path, caplog):
        """Filenames with '&' should log a warning."""
        (tmp_path / "file&name.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "special characters" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_warns_on_quotes(self, tmp_path, caplog):
        """Filenames with quotes should log a warning."""
        (tmp_path / 'file"name.mp4').write_bytes(b"video")
        (tmp_path / "file'name.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        warning_count = sum(
            1
            for record in caplog.records
            if record.levelno == logging.WARNING and "special characters" in record.message
        )
        assert warning_count >= 2

    def test_warns_on_leading_trailing_spaces(self, tmp_path, caplog):
        """Filenames with leading/trailing whitespace should log a warning."""
        (tmp_path / " leading.mp4").write_bytes(b"video")
        (tmp_path / "trailing .mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "leading/trailing whitespace" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_no_warning_for_clean_filenames(self, tmp_path, caplog):
        """Clean filenames should not trigger special character warnings."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "my-document.pdf").write_bytes(b"pdf")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
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

    def test_warning_includes_filename(self, tmp_path, caplog):
        """Warning message should include the problematic filename."""
        (tmp_path / "file<test>.mp4").write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "file<test>.mp4" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )


class TestOrphanedJsonDetection:
    """Tests for orphaned JSON file detection."""

    def test_orphaned_json_logs_warning(self, tmp_path, caplog):
        """An orphaned JSON file should log a warning."""
        (tmp_path / "orphan.mp4.json").write_text('{"transcribe": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_orphaned_json_not_in_results(self, tmp_path, caplog):
        """Orphaned JSON files should not appear in the result list."""
        (tmp_path / "orphan.mp4.json").write_text('{"transcribe": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            result = scanner.scan_directory()

        assert result == []

    def test_json_with_source_not_orphaned(self, tmp_path, caplog):
        """A JSON file with a matching source should not be flagged as orphaned."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            result = scanner.scan_directory()

        assert result == []
        assert not any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_json_with_unsupported_extension_is_orphaned(self, tmp_path, caplog):
        """A JSON file whose stripped name has no supported extension is orphaned."""
        (tmp_path / "config.json").write_text('{"extract": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_orphaned_json_in_subdirectory(self, tmp_path, caplog):
        """Orphaned JSON in subdirectories should be detected."""
        (tmp_path / "Course1").mkdir()
        (tmp_path / "Course1" / "orphan.pdf.json").write_text(
            '{"extract": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            result = scanner.scan_directory()

        assert result == []
        assert any(
            "Orphaned JSON" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_warning_includes_json_path(self, tmp_path, caplog):
        """Orphaned JSON warning should include the file path."""
        (tmp_path / "orphan.mp4.json").write_text('{"transcribe": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            scanner.scan_directory()

        assert any(
            "orphan.mp4.json" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )


class TestScanDirectoryErrorHandling:
    """Tests for error handling during directory scanning."""

    def test_handles_permission_error_gracefully(self, tmp_path, caplog):
        """Scanner should not crash on PermissionError for individual files."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)

        original_stat = Path.stat

        def mock_stat(self, *args, **kwargs):
            if self == source:
                raise PermissionError("Permission denied")
            return original_stat(self, *args, **kwargs)

        with (
            mock.patch.object(Path, "stat", mock_stat),
            caplog.at_level(logging.WARNING, logger="local_feed_generator"),
        ):
            result = scanner.scan_directory()

        # File should still be returned (treated as needing processing).
        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_handles_os_error_on_stat(self, tmp_path, caplog):
        """Scanner should handle OSError when stat-ing files."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video")

        scanner = Scanner(data_dir=tmp_path)

        original_stat = Path.stat

        def mock_stat(self, *args, **kwargs):
            if self == source:
                raise OSError("I/O error")
            return original_stat(self, *args, **kwargs)

        with (
            mock.patch.object(Path, "stat", mock_stat),
            caplog.at_level(logging.WARNING, logger="local_feed_generator"),
        ):
            result = scanner.scan_directory()

        assert len(result) == 1

    def test_no_filesystem_side_effects(self, tmp_path):
        """Scanner should not create, modify, or delete any files."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "lecture.mp4.json").write_text("{}")

        initial_files = set()
        for root, _dirs, files in os.walk(tmp_path):
            for f in files:
                initial_files.add(os.path.join(root, f))

        scanner = Scanner(data_dir=tmp_path)
        scanner.scan_directory()

        final_files = set()
        for root, _dirs, files in os.walk(tmp_path):
            for f in files:
                final_files.add(os.path.join(root, f))

        assert initial_files == final_files


class TestScannedFileDataclass:
    """Tests for the ScannedFile dataclass."""

    def test_scanned_file_is_frozen(self):
        """ScannedFile should be immutable (frozen dataclass)."""
        entry = ScannedFile(
            filepath="lecture.mp4",
            absolute_path="/data/lecture.mp4",
            filename="lecture.mp4",
            filetype="mp4",
        )
        with pytest.raises(AttributeError):
            entry.filepath = "other.mp4"

    def test_scanned_file_fields(self):
        """ScannedFile should have all required fields."""
        entry = ScannedFile(
            filepath="videos/lecture.mp4",
            absolute_path="/data/videos/lecture.mp4",
            filename="lecture.mp4",
            filetype="mp4",
        )
        assert entry.filepath == "videos/lecture.mp4"
        assert entry.absolute_path == "/data/videos/lecture.mp4"
        assert entry.filename == "lecture.mp4"
        assert entry.filetype == "mp4"

    def test_scanned_file_default_optional_fields(self):
        """ScannedFile should have sensible defaults for optional fields."""
        entry = ScannedFile(
            filepath="lecture.mp4",
            absolute_path="/data/lecture.mp4",
            filename="lecture.mp4",
            filetype="mp4",
        )
        assert entry.current_stage is None
        assert entry.json_exists is False

    def test_scanned_file_with_optional_fields(self):
        """ScannedFile should accept current_stage and json_exists."""
        entry = ScannedFile(
            filepath="lecture.mp4",
            absolute_path="/data/lecture.mp4",
            filename="lecture.mp4",
            filetype="mp4",
            current_stage="transcribing",
            json_exists=True,
        )
        assert entry.current_stage == "transcribing"
        assert entry.json_exists is True


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

    def test_empty_json_file_still_marks_as_processed(self, tmp_path):
        """An empty JSON file should still count as 'has metadata'."""
        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "lecture.mp4.json").write_text("")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    def test_json_newer_than_source_not_returned(self, tmp_path):
        """A source file with a newer JSON should not be returned."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        # Make JSON newer.
        import os
        import time

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(source, (old_time, old_time))
        os.utime(json_file, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()
        assert result == []

    def test_source_newer_than_json_returned(self, tmp_path):
        """A source file newer than its JSON should be returned."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        # Make source newer.
        import os
        import time

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()
        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"


class TestStepStatusBasedScanning:
    """Tests for step status-based scanning logic."""

    def test_in_progress_transcribe_status_returned(self, tmp_path):
        """Files with status['transcribe']='in_progress' should be returned for resuming."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "in_progress"}, "summarize": {"status": null}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"
        assert result[0].json_exists is True

    def test_in_progress_summarize_status_returned(self, tmp_path):
        """Files with status['summarize']='in_progress' should be returned for resuming."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "in_progress"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_in_progress_extract_status_returned(self, tmp_path):
        """Files with status['extract']='in_progress' should be returned for resuming."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        json_file = tmp_path / "doc.pdf.json"
        json_file.write_text('{"extract": {"status": "in_progress"}}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "doc.pdf"

    def test_error_skipped_when_source_not_newer(self, tmp_path):
        """Files with error should be skipped if source is not newer."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}, "error": "transcription failed"}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    def test_error_returned_when_source_is_newer(self, tmp_path):
        """Files with error should be returned if source is newer.

        This allows retrying after the user fixes the source file.
        """
        import os
        import time

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}, "error": "old error"}')

        # Make source newer than JSON.
        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_complete_mp4_skipped(self, tmp_path):
        """MP4 files with all steps complete should be skipped."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    def test_complete_pdf_skipped(self, tmp_path):
        """PDF files with status['extract']='complete' should be skipped."""
        source = tmp_path / "lecture.pdf"
        source.write_bytes(b"pdf data")
        json_file = tmp_path / "lecture.pdf.json"
        json_file.write_text('{"extract": {"status": "complete"}}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

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

    def test_complete_mp4_returned_when_source_newer(self, tmp_path):
        """MP4 files with all steps complete should be returned if source is newer."""
        import os
        import time

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

    def test_stage_extracted_from_json(self, tmp_path):
        """The stage field should be extracted from JSON metadata."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "in_progress"}, "summarize": {"status": null}, "stage": "transcribing"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "transcribing"

    def test_current_stage_none_when_not_in_json(self, tmp_path):
        """current_stage should be None when not present in JSON metadata."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage is None

    def test_current_stage_none_when_no_json(self, tmp_path):
        """current_stage should be None when no JSON metadata exists."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage is None
        assert result[0].json_exists is False

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
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            result = scanner.scan_directory()

        assert result == []
        assert any(
            "Could not parse JSON metadata" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_invalid_json_returned_when_source_newer(self, tmp_path, caplog):
        """Invalid JSON should still trigger reprocessing if source is newer."""
        import os
        import time

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text("{invalid json content")

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="local_feed_generator"):
            result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"
        assert result[0].json_exists is True
        assert result[0].current_stage is None


class TestJsonExistsField:
    """Tests for the json_exists field on ScannedFile."""

    def test_json_exists_true_for_file_with_json(self, tmp_path):
        """json_exists should be True when JSON metadata exists."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text('{"transcribe": {"status": "in_progress"}}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].json_exists is True

    def test_json_exists_false_for_file_without_json(self, tmp_path):
        """json_exists should be False when no JSON metadata exists."""
        (tmp_path / "lecture.mp4").write_bytes(b"video data")

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].json_exists is False

    def test_json_exists_true_when_source_newer_than_error_json(self, tmp_path):
        """json_exists should be True even when returning due to freshness."""
        import os
        import time

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

    def test_in_progress_with_stage_returned_with_stage(self, tmp_path):
        """An in_progress file with stage should carry that stage."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"transcribe": {"status": "in_progress"}, "summarize": {"status": null}, "stage": "transcribing"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "transcribing"

    def test_in_progress_extract_with_stage_returned_with_stage(self, tmp_path):
        """An in_progress PDF with stage should carry that stage."""
        source = tmp_path / "lecture.pdf"
        source.write_bytes(b"pdf data")
        (tmp_path / "lecture.pdf.json").write_text(
            '{"extract": {"status": "in_progress"}, "stage": "extracting_metadata"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "extracting_metadata"

    def test_source_newer_than_complete_json_includes_stage(self, tmp_path):
        """When source is newer, stage from JSON should still be included."""
        import os
        import time

        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe": {"status": "complete"}, "summarize": {"status": "complete"}, "stage": "summarizing"}'
        )

        old_time = time.time() - 100
        new_time = time.time()
        os.utime(json_file, (old_time, old_time))
        os.utime(source, (new_time, new_time))

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "summarizing"

    def test_stage_not_extracted_from_non_string(self, tmp_path):
        """current_stage should be None when JSON stage value is not a string."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"transcribe": {"status": "in_progress"}, "stage": 123}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage is None

    def test_stage_not_extracted_from_null(self, tmp_path):
        """current_stage should be None when JSON stage value is null."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"status": {"transcribe": "in_progress"}, "stage": null}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage is None

    def test_current_stage_fallback_to_legacy_field(self, tmp_path):
        """Scanner should also read legacy 'current_stage' field from JSON."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"status": {"transcribe": "in_progress"}, "current_stage": "legacy_stage_value"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "legacy_stage_value"

    def test_stage_preferred_over_current_stage(self, tmp_path):
        """When both 'stage' and 'current_stage' are present, 'stage' wins."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        (tmp_path / "lecture.mp4.json").write_text(
            '{"status": {"transcribe": "in_progress"}, "stage": "new_field", "current_stage": "old_field"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert len(result) == 1
        assert result[0].current_stage == "new_field"

    def test_legacy_status_fields_backward_compatible(self, tmp_path):
        """Scanner should handle old JSON files with individual status fields."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe_status": "in_progress", "summarize_status": null}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        # Should be returned because transcribe is not complete
        assert len(result) == 1
        assert result[0].filename == "lecture.mp4"

    def test_legacy_status_fields_complete_skipped(self, tmp_path):
        """Scanner should skip files with legacy status fields all complete."""
        source = tmp_path / "lecture.mp4"
        source.write_bytes(b"video data")
        json_file = tmp_path / "lecture.mp4.json"
        json_file.write_text(
            '{"transcribe_status": "complete", "summarize_status": "complete"}'
        )

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []

    def test_legacy_extract_status_complete_skipped(self, tmp_path):
        """Scanner should skip PDF files with legacy extract_status='complete'."""
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"pdf data")
        json_file = tmp_path / "doc.pdf.json"
        json_file.write_text('{"extract_status": "complete"}')

        scanner = Scanner(data_dir=tmp_path)
        result = scanner.scan_directory()

        assert result == []
