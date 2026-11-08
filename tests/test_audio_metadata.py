"""Tests for the PDF, HTML, and audio metadata extractors."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest import mock

from pypdf import PdfWriter
import pytest

from focus_feed.services.extractors import (
    get_audio_metadata,
)

# ---------------------------------------------------------------------------
# Helper functions for creating test fixtures
# ---------------------------------------------------------------------------


def create_test_pdf(
    path: Path,
    title: str | None = "Test PDF Title",
    author: str | None = "Test Author",
    creation_date: datetime | None = None,
) -> None:
    """Create a minimal PDF file with optional metadata.

    Args:
        path: Where to write the PDF file.
        title: Title to embed in PDF metadata, or None to omit.
        author: Author to embed in PDF metadata, or None to omit.
        creation_date: Creation date to embed, or None to omit.
            Converted to PDF date format string (D:YYYYMMDDHHMMSS).
    """
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    metadata: dict[str, Any] = {}
    if title is not None:
        metadata["/Title"] = title
    if author is not None:
        metadata["/Author"] = author
    if creation_date is not None:
        # PDF date format: D:YYYYMMDDHHMMSS
        metadata["/CreationDate"] = creation_date.strftime("D:%Y%m%d%H%M%S")

    if metadata:
        writer.add_metadata(metadata)

    with open(path, "wb") as f:
        writer.write(f)


def create_test_pdf_with_text(
    path: Path,
    text: str,
    title: str | None = "Test PDF Title",
    author: str | None = "Test Author",
) -> None:
    """Create a PDF file with text content and optional metadata.

    Uses pypdf's PdfWriter to create a page with a content stream
    containing the given text so that ``extract_text()`` can read it back.

    Args:
        path: Where to write the PDF file.
        text: The text content to embed in the PDF page.
        title: Title to embed in PDF metadata, or None to omit.
        author: Author to embed in PDF metadata, or None to omit.
    """
    from pypdf import PageObject
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    page = PageObject.create_blank_page(width=200, height=200)

    # Build a minimal content stream with the text.
    content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject("/Contents")] = stream

    # Add a Helvetica font resource so text is extractable.
    font_dict = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_indirect = writer._add_object(font_dict)
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): font_indirect,
                }
            ),
        }
    )
    page[NameObject("/Resources")] = resources

    writer.add_page(page)

    metadata: dict[str, Any] = {}
    if title is not None:
        metadata["/Title"] = title
    if author is not None:
        metadata["/Author"] = author
    if metadata:
        writer.add_metadata(metadata)

    with open(path, "wb") as f:
        writer.write(f)


def create_test_html(
    path: Path,
    title: str | None = "Test HTML Title",
    description: str | None = "Test description from meta tag",
    date: str | None = "2026-10-14T12:00:00Z",
    pubdate: str | None = None,
    time_datetime: str | None = None,
) -> None:
    """Create an HTML file with optional metadata.

    Args:
        path: Where to write the HTML file.
        title: Content of the <title> tag, or None to omit.
        description: Content of <meta name="description">, or None to omit.
        date: Content of <meta name="date">, or None to omit.
        pubdate: Content of <meta name="pubdate">, or None to omit.
        time_datetime: Value of <time datetime="...">, or None to omit.
    """
    html_parts = ["<!DOCTYPE html>", "<html>", "<head>"]

    if title is not None:
        html_parts.append(f"<title>{title}</title>")

    if description is not None:
        html_parts.append(f'<meta name="description" content="{description}">')

    if date is not None:
        html_parts.append(f'<meta name="date" content="{date}">')

    if pubdate is not None:
        html_parts.append(f'<meta name="pubdate" content="{pubdate}">')

    html_parts.append("</head>")
    html_parts.append("<body>")

    if time_datetime is not None:
        html_parts.append(f'<time datetime="{time_datetime}">October 14, 2026</time>')

    html_parts.append("<p>Test content</p>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    path.write_text("\n".join(html_parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# _clean_title helper tests
# ---------------------------------------------------------------------------


class TestGetAudioMetadata:
    """Tests for the get_audio_metadata function."""

    @pytest.fixture
    def mock_probe(self, tmp_path):
        """Set up a mock audio file and return helper for patching."""
        audio_path = tmp_path / "test.mp4"
        audio_path.write_bytes(b"fake audio data")
        return audio_path

    def test_returns_none_when_ffprobe_not_found(self, mock_probe):
        """Should return None when ffprobe is not on PATH."""
        with mock.patch("shutil.which", return_value=None):
            result = get_audio_metadata(mock_probe)
        assert result is None

    def test_returns_metadata_dict_when_ffprobe_succeeds(self, mock_probe):
        """Should return a dict with expected keys when ffprobe succeeds."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "3600.500000",
                "bit_rate": "128000",
                "tags": {
                    "creation_time": "2024-01-15T10:30:00.000000Z",
                },
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "3600.500000",
                    "bit_rate": "128000",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["type"] == "mov,mp4,m4a,3gp,3g2,mj2"
        assert result["duration"] == 3600.5
        assert result["creation_time"] == "2024-01-15T10:30:00.000000Z"
        assert result["bit_rate"] == 128000

    @pytest.mark.parametrize(
        "side_effect,expected",
        [
            (subprocess.CalledProcessError(1, "ffprobe"), None),
            (OSError("command not found"), None),
        ],
    )
    def test_returns_none_on_subprocess_error(self, mock_probe, side_effect, expected):
        """Should return None when ffprobe command fails."""
        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = side_effect
            result = get_audio_metadata(mock_probe)
        assert result is expected

    def test_returns_none_on_json_decode_error(self, mock_probe):
        """Should return None when ffprobe output is not valid JSON."""
        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout="not valid json",
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)
        assert result is None

    def test_handles_missing_creation_time(self, mock_probe):
        """Should return None for creation_time when tag is absent."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "120.000000",
                "tags": {},
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "120.000000",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["creation_time"] is None
        assert result["duration"] == 120.0
        assert result["bit_rate"] is None

    def test_falls_back_to_format_duration_when_no_audio_stream(self, mock_probe):
        """Should use format-level duration when no audio stream exists."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "600.000000",
                "bit_rate": "256000",
                "tags": {
                    "creation_time": "2023-06-01T08:00:00Z",
                },
            },
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "600.000000",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["duration"] == 600.0
        assert result["bit_rate"] == 256000
        assert result["creation_time"] == "2023-06-01T08:00:00Z"

    def test_handles_empty_streams_list(self, mock_probe):
        """Should return 0 duration when no streams and no format duration."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert result["duration"] == 0.0
        assert result["creation_time"] is None
        assert result["bit_rate"] is None

    def test_accepts_string_path(self, mock_probe):
        """Should accept a string path, not just Path."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "100.0",
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(str(mock_probe))

        assert result is not None
        assert result["type"] == "mov,mp4,m4a,3gp,3g2,mj2"
        # Verify the path was passed to ffprobe as a string
        call_args = mock_run.call_args
        assert str(mock_probe) in call_args[0][0]

    @pytest.mark.parametrize(
        "duration_str,expected_duration",
        [
            ("180.5", 180.5),
            ("100.0", 100.0),
            ("0.0", 0.0),
        ],
    )
    def test_duration_is_float(self, mock_probe, duration_str, expected_duration):
        """Duration should be a float."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": duration_str,
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert isinstance(result["duration"], float)
        assert result["duration"] == expected_duration

    @pytest.mark.parametrize(
        "bit_rate_str,expected_bit_rate",
        [
            ("320000", 320000),
            ("128000", 128000),
        ],
    )
    def test_bit_rate_is_int(self, mock_probe, bit_rate_str, expected_bit_rate):
        """Bit rate should be an int."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "bit_rate": bit_rate_str,
                "tags": {},
            },
            "streams": [],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        assert isinstance(result["bit_rate"], int)
        assert result["bit_rate"] == expected_bit_rate

    def test_returns_all_required_keys(self, mock_probe):
        """Result dict should contain all required keys."""
        mock_probe_output = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "100.0",
                "bit_rate": "128000",
                "tags": {
                    "creation_time": "2024-01-15T10:30:00.000000Z",
                },
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "100.0",
                }
            ],
        }

        with (
            mock.patch("shutil.which", return_value="/usr/bin/ffprobe"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout=json.dumps(mock_probe_output),
                stderr="",
                returncode=0,
            )
            result = get_audio_metadata(mock_probe)

        assert result is not None
        required_keys = {"type", "duration", "creation_time", "bit_rate"}
        assert required_keys.issubset(set(result.keys()))
