"""Pytest configuration for Focus Feed tests."""

from collections.abc import Iterator
import os
from unittest import mock

import pytest

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
    "TRANSCRIPTION_ENDPOINT",
    "TRANSCRIPTION_API_KEY",
)


@pytest.fixture(autouse=True)
def restore_required_env_vars() -> Iterator[None]:
    """Save and restore required env vars around each test.

    Tests in test_cli.py call ``_load_config()`` which invokes
    ``load_dotenv()``, injecting .env values directly into ``os.environ``.
    Because ``load_dotenv`` is not tracked by ``monkeypatch``, those values
    persist across tests and can cause unrelated tests (e.g. ones that
    intentionally omit a required field) to pass or fail unexpectedly.

    This fixture snapshots the required env vars before each test and
    restores them afterwards, preventing cross-test leakage.
    """
    saved: dict[str, str | None] = {key: os.environ.get(key) for key in _REQUIRED_ENV_VARS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
    yield
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
