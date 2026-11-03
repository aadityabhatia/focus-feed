"""Tests for audio format compatibility and extraction cleanup."""

from pathlib import Path

import pytest

from focus_feed.services import audio_extractor


@pytest.mark.parametrize("extension", audio_extractor.DIRECT_TRANSCRIBE_FORMATS)
def test_direct_transcribe_format_is_returned_unchanged(tmp_path: Path, extension: str) -> None:
    source = tmp_path / f"lecture{extension}"
    source.touch()

    assert audio_extractor.ensure_compatible_format(source) == str(source)


@pytest.mark.parametrize("extension", audio_extractor.NEARLY_SUPPORTED_FORMATS)
def test_nearly_supported_format_is_extracted(
    tmp_path: Path, extension: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / f"lecture{extension}"
    source.touch()
    expected = tmp_path / "lecture.mp3"
    monkeypatch.setattr(
        audio_extractor, "_extract_audio_to_supported_format", lambda path: expected
    )

    assert audio_extractor.ensure_compatible_format(source) == str(expected)


def test_missing_source_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audio_extractor.ensure_compatible_format(tmp_path / "missing.mkv")


def test_unsupported_format_raises_value_error(tmp_path: Path) -> None:
    source = tmp_path / "lecture.3gp"
    source.touch()

    with pytest.raises(ValueError, match="Unsupported transcription format"):
        audio_extractor.ensure_compatible_format(source)


def test_partial_output_is_removed_when_extraction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "lecture.mkv"
    source.touch()
    output = tmp_path / "lecture.flac"

    def fail_after_writing(_source: Path, destination: Path, _codec: str, _format: str) -> None:
        destination.write_bytes(b"partial")
        raise OSError("encoding failed")

    monkeypatch.setattr(audio_extractor, "_convert_audio", fail_after_writing)

    with pytest.raises(RuntimeError, match="Unable to extract audio"):
        monkeypatch.setattr(audio_extractor, "_get_audio_codec", lambda _path: "unknown")
        audio_extractor._extract_audio_to_supported_format(source, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        ("mp3", (".mp3", "mp3")),
        ("aac", (".m4a", "mp4")),
        ("flac", (".flac", "flac")),
        ("opus", (".ogg", "ogg")),
        ("vorbis", (".ogg", "ogg")),
        ("pcm_s16le", (".wav", "wav")),
        ("unknown", (".flac", "flac")),
    ],
)
def test_output_format_is_selected_by_codec(codec: str, expected: tuple[str, str]) -> None:
    assert audio_extractor._output_format_for_codec(codec) == expected
