"""Extract audio from video formats for direct transcription API compatibility."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile

import av

_logger = logging.getLogger("focus_feed")

DIRECT_TRANSCRIBE_FORMATS = (
    ".mp3",
    ".m4a",
    ".wav",
    ".webm",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mpeg",
    ".mpga",
)
NEARLY_SUPPORTED_FORMATS = (".mkv", ".mov", ".avi", ".wmv")
_SUPPORTED_AUDIO_CODECS = {
    "mp3": (".mp3", "mp3"),
    "aac": (".m4a", "mp4"),
    "flac": (".flac", "flac"),
    "opus": (".ogg", "ogg"),
    "vorbis": (".ogg", "ogg"),
}


def _is_direct_transcribe_format(filepath: str | Path) -> bool:
    """Return whether a file extension is supported by direct transcription.

    Args:
        filepath: Path to the audio or video file.

    Returns:
        ``True`` if the extension is supported for direct transcription.
    """
    return Path(filepath).suffix.lower() in DIRECT_TRANSCRIBE_FORMATS


def ensure_compatible_format(filepath: str | os.PathLike[str]) -> str:
    """Return a path in a format accepted by the transcription API.

    Supported formats are returned unchanged. Video formats requiring
    conversion are extracted to a temporary file in a supported audio format,
    which the caller must remove when finished.

    Args:
        filepath: Path to an audio or video file.

    Returns:
        The original path for a supported format, or a temporary audio path.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input extension is neither supported nor convertible.
        RuntimeError: If audio extraction fails.
    """
    source_path = Path(filepath)
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio/video file not found: {source_path}")

    if _is_direct_transcribe_format(source_path):
        return str(source_path)
    if source_path.suffix.lower() not in NEARLY_SUPPORTED_FORMATS:
        supported = ", ".join(DIRECT_TRANSCRIBE_FORMATS)
        raise ValueError(
            f"Unsupported transcription format '{source_path.suffix}'. "
            f"Supported formats are: {supported}"
        )

    return str(_extract_audio_to_supported_format(source_path))


def _extract_audio_to_supported_format(source_path: Path, output_path: Path | None = None) -> Path:
    """Extract the first audio stream without loss when its codec is supported.

    Args:
        source_path: Path to the source video.
        output_path: Destination audio path. If omitted, a temporary file with
            an extension appropriate to the source codec is created.

    Returns:
        Path to the generated audio file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        RuntimeError: If the source contains no usable audio or extraction fails.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio/video file not found: {source_path}")
    if output_path is not None and output_path.resolve() == source_path.resolve():
        raise ValueError("Audio output path must not be the same as the source video")

    try:
        codec_name = _get_audio_codec(source_path)
        output_suffix, output_format = _output_format_for_codec(codec_name)
        if output_path is None:
            descriptor, temp_name = tempfile.mkstemp(suffix=output_suffix)
            os.close(descriptor)
            output_path = Path(temp_name)

        _convert_audio(source_path, output_path, codec_name, output_format)
        if output_path.stat().st_size == 0:
            raise RuntimeError("Audio extraction produced an empty file")
    except Exception as exc:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                _logger.warning(
                    "Could not remove partial audio file %s: %s",
                    output_path,
                    cleanup_error,
                )
        error = RuntimeError(
            f"Unable to extract audio from '{source_path}' "
            f"(unsupported codecs fall back to FLAC): {exc}"
        )
        _logger.error("%s", error)
        raise error from exc

    _logger.info("Extracted audio from %s to %s", source_path, output_path)
    return output_path


def _get_audio_codec(source_path: Path) -> str:
    """Return the codec name of the source's first audio stream."""
    with av.open(str(source_path), mode="r") as input_container:
        audio_streams = input_container.streams.audio
        if not audio_streams:
            raise RuntimeError("Unable to extract audio: source contains no audio stream")
        return audio_streams[0].codec.name.lower()


def _output_format_for_codec(codec_name: str) -> tuple[str, str]:
    """Return the output extension and container format for an audio codec."""
    if codec_name in _SUPPORTED_AUDIO_CODECS:
        return _SUPPORTED_AUDIO_CODECS[codec_name]
    if codec_name.startswith("pcm_"):
        return ".wav", "wav"
    return ".flac", "flac"


def _convert_audio(
    source_path: Path,
    output_path: Path,
    codec_name: str,
    output_format: str,
) -> None:
    """Copy supported encoded packets or losslessly encode unsupported audio."""
    with av.open(str(source_path), mode="r") as input_container:
        audio_streams = input_container.streams.audio
        if not audio_streams:
            raise RuntimeError("Unable to extract audio: source contains no audio stream")

        input_stream = audio_streams[0]
        with av.open(str(output_path), mode="w", format=output_format) as output_container:
            if codec_name in _SUPPORTED_AUDIO_CODECS or codec_name.startswith("pcm_"):
                output_stream = output_container.add_stream(
                    template=input_stream  # type: ignore[call-overload]
                )
                for packet in input_container.demux(input_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = output_stream
                    output_container.mux(packet)
                return

            output_stream = output_container.add_stream("flac", rate=input_stream.rate or 44100)
            for frame in input_container.decode(input_stream):
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
            for packet in output_stream.encode(None):
                output_container.mux(packet)
