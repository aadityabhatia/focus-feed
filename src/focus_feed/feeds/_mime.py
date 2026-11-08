"""Canonical MIME types for podcast media enclosures."""

from focus_feed.filetypes import MEDIA_FILETYPES


def get_mime_type(filetype: str) -> str:
    """Return the appropriate MIME type for a media filetype."""
    mime_map: dict[str, str] = {
        "mp4": "audio/mp4",
        "m4a": "audio/mp4",
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "mpga": "audio/mpeg",
        "wav": "audio/wav",
        "webm": "audio/webm",
        "aac": "audio/aac",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "opus": "audio/opus",
        # Audio is extracted from video formats for podcast playback.
        "mkv": "audio/mp4",
        "mov": "audio/mp4",
        "avi": "audio/mp4",
        "wmv": "audio/mp4",
    }
    return mime_map.get(filetype, "application/octet-stream")


PODCAST_MIME_TYPES = frozenset(get_mime_type(filetype) for filetype in MEDIA_FILETYPES)
_PODCAST_MIME_TYPES = PODCAST_MIME_TYPES
