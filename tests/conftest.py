"""Pytest configuration for Focus Feed tests."""

from collections.abc import Callable, Iterator
from io import StringIO
import logging
import os
from unittest import mock

import pytest

from focus_feed.config import Settings
from focus_feed.errors import _configure_error_log

# Required env vars that may be loaded into os.environ by load_dotenv() in
# CLI tests.  These are saved and restored around each test to prevent
# leakage between tests.
_REQUIRED_ENV_VARS: tuple[str, ...] = (
    "BASE_URL",
    "LLM_ENDPOINT",
    "LLM_API_KEY",
    "LLM_MODEL",
    "PROMPT_ARTICLE",
    "PROMPT_ARTICLE_REFLECTION",
    "PROMPT_PODCAST",
    "PROMPT_PODCAST_REFLECTION",
    "PROMPT_TITLE",
    "DOCLING_API",
    "TRANSCRIPTION_ENDPOINT",
    "TRANSCRIPTION_API_KEY",
)


@pytest.fixture(autouse=True)
def restore_required_env_vars(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Save and restore required env vars around each test.

    Tests in test_cli.py call ``_load_config()`` which invokes
    ``load_dotenv()``, injecting .env values directly into ``os.environ``.
    Because ``load_dotenv`` is not tracked by ``monkeypatch``, those values
    persist across tests and can cause unrelated tests (e.g. ones that
    intentionally omit a required field) to pass or fail unexpectedly.

    This fixture clears required env vars before each test and restores their
    original values afterwards, preventing ambient shell state and
    cross-test leakage from affecting configuration tests.
    """
    saved: dict[str, str | None] = {key: os.environ.get(key) for key in _REQUIRED_ENV_VARS}
    for key in _REQUIRED_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def reset_config() -> Iterator[None]:
    """Reset the cached configuration before and after a test."""
    import focus_feed.config as config_module

    config_module._config_instance = None
    yield
    config_module._config_instance = None


@pytest.fixture
def valid_env(monkeypatch: pytest.MonkeyPatch, reset_config: None) -> Iterator[None]:
    """Provide valid environment variables for config-dependent tests."""
    env = {
        "BASE_URL": "https://example.com/feeds",
        "LLM_ENDPOINT": "https://llm.example.com/v1",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-model",
        "PROMPT_ARTICLE": "Test article prompt",
        "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
        "PROMPT_PODCAST": "Test podcast prompt",
        "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
        "PROMPT_TITLE": "Test title prompt",
        "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
        "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        "TRANSCRIPTION_REMOTE_MODEL": "",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield


@pytest.fixture
def mock_config() -> Settings:
    """Provide the shared Settings instance used by summarizer tests."""
    return Settings(
        base_url="https://example.com",
        llm_endpoint="https://llm.example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        prompt_article="Summarize key takeaways from this document.",
        prompt_article_reflection="Summarize key takeaways from this document with reflection.",
        prompt_podcast="Summarize key takeaways from this podcast.",
        prompt_podcast_reflection="Summarize key takeaways from this podcast with reflection.",
        prompt_title="Generate a title for the given content.",
        transcription_endpoint="https://transcribe.example.com/v1",
        transcription_api_key="transcribe-test-key",
    )


@pytest.fixture
def capture_logger() -> Callable[..., tuple[logging.Logger, StringIO]]:
    """Return a logger-capture factory, optionally given a logger name."""

    def _capture_logger(
        logger_name: str = "test_log_error",
    ) -> tuple[logging.Logger, StringIO]:
        log = logging.getLogger(logger_name)
        log.handlers = []
        log.setLevel(logging.DEBUG)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handler.setLevel(logging.DEBUG)
        log.addHandler(handler)
        return log, stream

    return _capture_logger


@pytest.fixture(autouse=True)
def provide_default_transcription_env() -> Iterator[None]:
    """Provide valid remote transcription credentials unless a test overrides them."""
    endpoint = os.environ.get("TRANSCRIPTION_ENDPOINT")
    api_key = os.environ.get("TRANSCRIPTION_API_KEY")
    os.environ.setdefault("TRANSCRIPTION_ENDPOINT", "https://transcribe.example.com/v1")
    os.environ.setdefault("TRANSCRIPTION_API_KEY", "transcribe-test-key")
    yield
    if endpoint is None:
        os.environ.pop("TRANSCRIPTION_ENDPOINT", None)
    else:
        os.environ["TRANSCRIPTION_ENDPOINT"] = endpoint
    if api_key is None:
        os.environ.pop("TRANSCRIPTION_API_KEY", None)
    else:
        os.environ["TRANSCRIPTION_API_KEY"] = api_key


@pytest.fixture(autouse=True)
def isolate_error_log(tmp_path_factory):
    """Redirect error.log to a temp directory to prevent test pollution."""
    tmp_dir = tmp_path_factory.mktemp("error_log")
    _configure_error_log(data_dir=tmp_dir, force=True)
    yield tmp_dir
    # Redirect to a fresh temp dir rather than CWD to avoid creating
    # error.log in the workspace root during teardown.
    tmp_dir_cleanup = tmp_path_factory.mktemp("error_log_cleanup")
    _configure_error_log(data_dir=tmp_dir_cleanup, force=True)


@pytest.fixture(autouse=True)
def disable_env_file_loading(request: pytest.FixtureRequest) -> Iterator[None]:
    """Disable loading .env file for tests that check config validation."""
    # Skip patching for env file loading tests
    test_node = request.node.nodeid
    if "TestEnvFileLoading" in test_node:
        yield
        return
    # Patch Settings model_config to use a non-existent env file
    from focus_feed.config import Settings

    original_config = Settings.model_config.copy()
    # Use a path that doesn't exist
    patched_config = {**original_config, "env_file": "/nonexistent/.env"}
    with mock.patch.object(Settings, "model_config", patched_config):
        yield
