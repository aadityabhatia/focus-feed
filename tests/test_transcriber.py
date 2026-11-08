"""Tests for the remote-only Transcriber adapter."""

from unittest import mock

import pytest

from focus_feed.errors import ConfigurationError
from focus_feed.services.transcriber import Transcriber


def test_transcriber_delegates_to_remote_service(valid_env):
    with mock.patch("focus_feed.services.transcriber.RemoteTranscriber") as remote_cls:
        remote = remote_cls.return_value
        remote.transcribe.return_value = "Remote transcript"

        transcriber = Transcriber()
        result = transcriber.transcribe("lecture.mp4")

    assert result == "Remote transcript"
    remote.transcribe.assert_called_once_with("lecture.mp4")


@pytest.mark.parametrize(
    "missing_field",
    ["TRANSCRIPTION_ENDPOINT", "TRANSCRIPTION_API_KEY"],
)
def test_missing_remote_configuration_raises_clear_error(monkeypatch, missing_field):
    incomplete_config = mock.Mock(transcription_endpoint="", transcription_api_key="")
    with (
        mock.patch("focus_feed.services.transcriber.get_config", return_value=incomplete_config),
        pytest.raises(ConfigurationError, match=missing_field),
    ):
        Transcriber()
