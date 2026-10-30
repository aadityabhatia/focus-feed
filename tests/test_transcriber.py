"""Tests for the remote-only Transcriber adapter."""

from unittest import mock

import pytest

from focus_feed.errors import ConfigurationError
from focus_feed.services.transcriber import Transcriber


def _set_valid_env(monkeypatch) -> None:
    """Set all required application settings for a remote transcription test."""
    variables = {
        "BASE_URL": "https://example.com/feeds",
        "LLM_ENDPOINT": "https://llm.example.com/v1",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-model",
        "PROMPT_ARTICLE": "Test article prompt",
        "PROMPT_PODCAST": "Test podcast prompt",
        "PROMPT_TITLE": "Test title prompt",
        "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
        "TRANSCRIPTION_API_KEY": "transcription-test-key",
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    import focus_feed.config as config_module

    config_module._config_instance = None


@pytest.fixture
def valid_env(monkeypatch):
    _set_valid_env(monkeypatch)
    yield
    import focus_feed.config as config_module

    config_module._config_instance = None


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
