"""Transcription service for audio files.

Uses faster-whisper with CUDA GPU acceleration for high-quality
audio transcription. The model is loaded once and reused for all
files via a singleton pattern.

Configuration is read from environment variables via the config module:
- TRANSCRIPTION_MODEL (default: large-v3-turbo)
- TRANSCRIPTION_DEVICE (default: cuda)
- TRANSCRIPTION_COMPUTE_TYPE (default: float16)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from local_feed_generator.config import get_config
from local_feed_generator.errors import TranscriptionError, log_error
from local_feed_generator.services.remote_transcriber import RemoteTranscriber

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# Runtime import with graceful fallback for optional dependency.
# When faster-whisper is not installed (e.g. remote-only mode), WhisperModel
# is None and LocalTranscriber._load_model() will raise a clear error.
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

_logger = logging.getLogger("local_feed_generator")

# Error message shown when CUDA libraries are missing but a GPU device is
# requested.  Kept as a module-level constant so it can be reused by the
# check utility, the model loader, and callers that want to surface the same
# actionable guidance.
CUDA_ERROR_MESSAGE = (
    "CUDA libraries not found. "
    "Transcription with GPU requires Nvidia CUDA Toolkit to be installed.\n\n"
    "To fix this:\n"
    "1. Install Nvidia GPU drivers for your graphics card\n"
    "2. Install CUDA Toolkit (version 12.x recommended for large-v3-turbo)\n"
    "   Download from: https://developer.nvidia.com/cuda-downloads\n"
    "3. Ensure CUDA libraries are in your LD_LIBRARY_PATH\n\n"
    "Alternatively, you can try transcription on CPU by setting:\n"
    "  TRANSCRIPTION_DEVICE=cpu\n"
    "Note: CPU transcription is much slower and not recommended for production use."
)


def check_cuda_available() -> bool:
    """Check if CUDA libraries (specifically cuBLAS) are available and loadable.

    Uses :func:`ctypes.CDLL` to actually attempt loading the cuBLAS shared
    library, which is more reliable than :func:`ctypes.util.find_library`
    (which only checks the linker cache).  This catches cases where the
    library exists in the cache but cannot actually be loaded (e.g.,
    missing dependencies, broken symlinks, wrong architecture, etc.).

    The search order is:
    1. Standard library paths (via bare ``ctypes.CDLL`` with versioned names)
    2. Python site-packages ``nvidia`` package directories (where
       ``nvidia-cublas-cu12`` and similar packages bundle the libraries)

    Returns:
        ``True`` if a CUDA cuBLAS library can be loaded, ``False`` otherwise.
    """
    try:
        import ctypes
    except ImportError:
        return False

    # Common cuBLAS library names across CUDA versions, ordered by preference.
    # Version-specific names are tried first since CTranslate2 looks for
    # specific versions (e.g., libcublas.so.12 for CUDA 12.x).
    lib_names = [
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so.10",
        "libcublas.so",
    ]

    # 1. Try loading from standard library paths (LD_LIBRARY_PATH, ldconfig cache, etc.)
    for name in lib_names:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue

    # 2. Try loading from Python site-packages (nvidia-cublas-cu12, etc.)
    #    These packages bundle CUDA libraries in <site-packages>/nvidia/cublas/lib/
    try:
        import site

        search_dirs: list[str] = []
        with contextlib.suppress(AttributeError):
            search_dirs.extend(site.getsitepackages())
        with contextlib.suppress(AttributeError):
            search_dirs.append(site.getusersitepackages())

        for site_dir in search_dirs:
            nvidia_lib_dir = Path(site_dir) / "nvidia" / "cublas" / "lib"
            if nvidia_lib_dir.is_dir():
                for name in lib_names:
                    lib_path = nvidia_lib_dir / name
                    if lib_path.is_file():
                        try:
                            ctypes.CDLL(str(lib_path))
                            return True
                        except OSError:
                            continue
    except Exception:
        pass

    return False


class LocalTranscriber:
    """Transcribes audio files to markdown text using faster-whisper.

    The WhisperModel is loaded once (singleton pattern) and reused
    across all :meth:`transcribe` calls. Use :meth:`cleanup` to
    unload the model and release GPU/CPU resources.

    Attributes:
        model: The loaded ``faster_whisper.WhisperModel`` instance,
            or ``None`` if not yet loaded or after :meth:`cleanup`.
    """

    def __init__(self) -> None:
        """Initialize the Transcriber with configuration from settings."""
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel | None:
        """Return the loaded model instance, or ``None`` if not loaded."""
        return self._model

    def _load_model(self) -> WhisperModel:
        """Load the faster-whisper model using config settings.

        Reads ``TRANSCRIPTION_MODEL``, ``TRANSCRIPTION_DEVICE``, and
        ``TRANSCRIPTION_COMPUTE_TYPE`` from the application config
        (which in turn reads from environment variables).

        Returns:
            The loaded ``WhisperModel`` instance.

        Raises:
            TranscriptionError: If model loading fails (e.g. CUDA not
                available, invalid model name, or network error during
                model download).
        """
        config = get_config()
        model_name = config.transcription_model
        device = config.transcription_device
        compute_type = config.transcription_compute_type

        _logger.info(
            "Loading transcription model '%s' on device '%s' with compute_type '%s'",
            model_name,
            device,
            compute_type,
        )

        # Check that faster-whisper is installed before attempting to load
        # the model.  This provides a clear, actionable error message instead
        # of an opaque TypeError when WhisperModel is None.
        if WhisperModel is None:
            raise TranscriptionError(
                "faster-whisper is not installed. "
                "Install with: pip install local-feed-generator[local-transcription]"
            )

        # Pre-emptive CUDA availability check: surface a clear, actionable
        # error before faster-whisper attempts to load the model and fails
        # with an opaque low-level exception.
        if device == "cuda" and not check_cuda_available():
            raise TranscriptionError(CUDA_ERROR_MESSAGE)

        try:
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            # Catch the specific case where faster-whisper fails because the
            # cuBLAS shared library cannot be loaded, and re-raise with the
            # helpful message instead of letting the opaque error bubble up.
            if "libcublas.so" in str(exc) or "cublas" in str(exc).lower():
                raise TranscriptionError(CUDA_ERROR_MESSAGE) from exc
            raise TranscriptionError(
                f"Failed to load transcription model '{model_name}': {exc}",
            ) from exc

        return model

    def _ensure_model(self) -> WhisperModel:
        """Return the loaded model, loading it if necessary.

        Implements the singleton pattern: the model is loaded only once
        and reused for all subsequent transcriptions.

        Returns:
            The loaded ``WhisperModel`` instance.

        Raises:
            TranscriptionError: If model loading fails.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def transcribe(self, filepath: str | os.PathLike[str]) -> str:
        """Transcribe an audio file to markdown text without timestamps.

        The file must exist. The model is loaded lazily on first call
        and reused for subsequent calls.

        Args:
            filepath: Path to the audio file (e.g. an MP4 file containing
                audio) to transcribe.

        Returns:
            A markdown string containing the full transcript without
            any timestamp annotations. Segments are joined with newlines
            to form a clean, readable transcript.

        Raises:
            TranscriptionError: If the file does not exist, the model
                cannot be loaded, or transcription fails for any reason.
        """
        path = Path(filepath)

        if not path.exists():
            raise TranscriptionError(
                f"File not found: {filepath}",
                filepath=str(filepath),
            )

        # Safety net: if the adjacent JSON metadata already marks transcription
        # as complete, return the existing transcript instead of re-transcribing.
        # This guards against callers that bypass the CLI/producer filtering.
        json_path = path.with_name(path.name + ".json")
        if json_path.exists():
            try:
                existing_data = json.loads(json_path.read_text(encoding="utf-8"))
                if (
                    isinstance(existing_data, dict)
                    and existing_data.get("status", {}).get("transcribe") == "complete"
                ):
                    existing_transcript = existing_data.get("transcript")
                    if isinstance(existing_transcript, str) and existing_transcript:
                        return existing_transcript
            except json.JSONDecodeError, OSError:
                pass

        # Get audio file info for user-facing output.
        file_size = self._get_file_size(path)
        audio_length = self._get_audio_duration(path)

        if audio_length is not None and file_size is not None:
            print(f"Transcribing {path.name}: {audio_length:.1f}s, {file_size:,} bytes")
        elif file_size is not None:
            print(f"Transcribing {path.name}: {file_size:,} bytes")
        else:
            print(f"Transcribing {path.name}")

        start_time = time.time()

        model = self._ensure_model()

        try:
            segments, _info = model.transcribe(str(path))

            # Collect all segment texts and join them into a clean
            # markdown transcript without timestamps.
            transcript_parts: list[str] = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    transcript_parts.append(text)

            transcript = "\n".join(transcript_parts).strip()

            elapsed = time.time() - start_time
            word_count = len(transcript.split())
            print(f"Completed in {elapsed:.2f}s ({word_count} words)")

            return transcript

        except TranscriptionError:
            raise
        except Exception as exc:
            # If the error is CUDA/cuBLAS related (e.g., the library was
            # found by check_cuda_available but fails during actual use),
            # surface the helpful CUDA error message instead of a generic
            # transcription failure.
            if "libcublas" in str(exc).lower() or "cublas" in str(exc).lower():
                raise TranscriptionError(
                    CUDA_ERROR_MESSAGE,
                    filepath=str(filepath),
                ) from exc
            error = TranscriptionError(
                f"Transcription failed for '{filepath}': {exc}",
                filepath=str(filepath),
            )
            log_error(error, filepath=str(filepath), logger=_logger)
            raise error from exc

    @staticmethod
    def _get_file_size(path: Path) -> int | None:
        """Return the file size in bytes, or ``None`` if it cannot be determined."""
        try:
            return path.stat().st_size
        except OSError:
            return None

    @staticmethod
    def _get_audio_duration(path: Path) -> float | None:
        """Return the audio duration in seconds, or ``None`` if it cannot be determined.

        Uses the ``av`` library (already a dependency via faster-whisper) to
        read the container's audio stream metadata without decoding the full
        audio into memory.  All exceptions are caught so that transcription
        can proceed even when metadata cannot be retrieved.
        """
        try:
            import av

            with av.open(str(path), mode="r") as container:
                if container.streams.audio:
                    # duration_seconds is a runtime property on av streams;
                    # use getattr because av stubs may not include it.
                    duration = getattr(container.streams.audio[0], "duration_seconds", None)
                    if duration is not None:
                        return float(duration)
        except Exception:
            pass
        return None

    def cleanup(self) -> None:
        """Unload the model and release associated resources.

        After calling this method, the next call to :meth:`transcribe`
        will reload the model from disk. This is useful for long-running
        processes that want to free GPU memory when transcription is
        not actively needed.
        """
        if self._model is not None:
            _logger.info("Unloading transcription model")
            try:
                # CTranslate2 models may have a delete method for
                # explicit cleanup; fall back to del if not available.
                if hasattr(self._model, "delete"):
                    self._model.delete()
            except Exception as exc:
                _logger.warning("Error during model cleanup: %s", exc)
            finally:
                del self._model
                self._model = None


class Transcriber(LocalTranscriber):
    """Factory/adapter that delegates to LocalTranscriber or RemoteTranscriber.

    Checks if ``transcription_endpoint`` is configured in the application
    config.  If an endpoint is set, transcription is performed via
    :class:`RemoteTranscriber` using an OpenAI-compatible API.  Otherwise,
    the local faster-whisper model (:class:`LocalTranscriber`) is used.

    This class maintains the exact same public interface as the original
    :class:`LocalTranscriber` so that existing code using ``Transcriber``
    continues to work without modification.

    Attributes:
        model: The loaded ``faster_whisper.WhisperModel`` instance (local mode
            only), or ``None`` if not yet loaded or after :meth:`cleanup`.
        _remote_transcriber: The :class:`RemoteTranscriber` instance used in
            remote mode, or ``None`` in local mode.
    """

    def __init__(self) -> None:
        """Initialize the Transcriber, detecting mode from config.

        If ``TRANSCRIPTION_ENDPOINT`` is configured, a
        :class:`RemoteTranscriber` is created for API-based transcription.
        Otherwise, the local faster-whisper transcriber is used (via the
        parent :class:`LocalTranscriber` initializer).
        """
        config = get_config()
        if config.transcription_endpoint:
            self._remote_transcriber: RemoteTranscriber | None = RemoteTranscriber(config=config)
        else:
            self._remote_transcriber = None
        super().__init__()

    @property
    def model(self) -> WhisperModel | None:
        """Return the loaded model instance, or ``None`` if not loaded.

        In remote mode, always returns ``None`` since no local model is
        loaded.
        """
        return self._model

    def transcribe(self, filepath: str | os.PathLike[str]) -> str:
        """Transcribe an audio file, delegating to the appropriate transcriber.

        If a remote transcription endpoint is configured, delegates to
        :class:`RemoteTranscriber`.  Otherwise, uses the local faster-whisper
        model via the parent :class:`LocalTranscriber`.

        Args:
            filepath: Path to the audio file (e.g. an MP4 file containing
                audio) to transcribe.

        Returns:
            A markdown string containing the full transcript without
            any timestamp annotations.

        Raises:
            TranscriptionError: If the file does not exist, the model
                cannot be loaded, or transcription fails for any reason.
        """
        if self._remote_transcriber is not None:
            return self._remote_transcriber.transcribe(filepath)
        return super().transcribe(filepath)

    def cleanup(self) -> None:
        """Release resources held by the underlying transcriber.

        For local transcription, unloads the faster-whisper model and
        releases GPU/CPU resources.  For remote transcription, this is
        a no-op since no local resources are held.
        """
        if self._remote_transcriber is not None:
            # RemoteTranscriber has no model to unload
            return
        super().cleanup()
