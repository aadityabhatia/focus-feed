"""Tests for the file scanner."""

import logging
import os
import time

import pytest

from focus_feed.scanner import (
    DIRECT_TRANSCRIBE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTRACTION_EXTENSIONS,
    Scanner,
    _is_audio_filetype,
    _is_video_filetype,
    _should_ignore_filename,
)
from focus_feed.services.extractors import DoclingExtractor


class TestSupportedMediaExtensions:
    """Tests for supported audio and video extension classification."""

    def test_supported_extensions_include_audio_video_and_documents(self):
        assert (
            *DIRECT_TRANSCRIBE_EXTENSIONS,
            *VIDEO_EXTRACTION_EXTENSIONS,
            ".html",
            *sorted(DoclingExtractor.SUPPORTED_EXTENSIONS),
        ) == SUPPORTED_EXTENSIONS

    @pytest.mark.parametrize("filetype", ["mp3", "M4A", ".WAV", "flac", "ogg", "opus"])
    def test_audio_filetype_is_case_insensitive(self, filetype):
        assert _is_audio_filetype(filetype)
        assert not _is_video_filetype(filetype)

    @pytest.mark.parametrize("filetype", ["mp4", "MP4", ".MkV", "mov", "AVI", "wmv"])
    def test_video_filetype_is_case_insensitive(self, filetype):
        assert _is_video_filetype(filetype)
        assert not _is_audio_filetype(filetype)

    @pytest.mark.parametrize(
        "extension", [*DIRECT_TRANSCRIBE_EXTENSIONS, *VIDEO_EXTRACTION_EXTENSIONS]
    )
    def test_scanner_discovers_supported_media_extensions(self, tmp_path, extension):
        source = tmp_path / f"media{extension.upper()}"
        source.write_bytes(b"media")

        result = Scanner(tmp_path).scan_directory()

        assert len(result) == 1
        assert result[0].filetype == extension.removeprefix(".")


class TestShouldIgnoreFilename:
    @pytest.mark.parametrize(
        "filename",
        [
            "index.html",
            "INDEX.HTML",
            ".hidden.mp4",
            ".DS_Store",
            "video.mp4.tmp",
            "video.mp4.Tmp",
            "video.mp4.TMP",
        ],
    )
    def test_ignores_generated_hidden_and_temporary_names(self, filename):
        assert _should_ignore_filename(filename)

    @pytest.mark.parametrize("filename", ["lecture.mp4", "report.PDF", "lecture.tmp.mp4"])
    def test_keeps_other_filenames(self, filename):
        assert not _should_ignore_filename(filename)

    @pytest.mark.parametrize("extension", sorted(DoclingExtractor.SUPPORTED_EXTENSIONS))
    def test_scanner_discovers_docling_extensions(self, tmp_path, extension):
        source = tmp_path / f"document{extension}"
        source.write_bytes(b"document")

        result = Scanner(tmp_path).scan_directory()

        assert len(result) == 1
        assert result[0].filetype == extension.removeprefix(".")


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
