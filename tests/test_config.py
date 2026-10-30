"""Tests for the configuration system."""

import contextlib
import logging
import os
from unittest import mock

from pydantic import ValidationError
import pytest

from focus_feed.config import Settings, get_config, reload_config


class TestRequiredFields:
    """Tests for required field validation."""

    def _get_valid_env(self) -> dict[str, str]:
        return {
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
        }

    def test_missing_all_required_fields_raises_error(self) -> None:
        """Config validation should fail when all required fields are missing."""
        env = {
            "BASE_URL": "",
            "LLM_ENDPOINT": "",
            "LLM_API_KEY": "",
            "LLM_MODEL": "",
            "PROMPT_ARTICLE": "",
            "PROMPT_ARTICLE_REFLECTION": "",
            "PROMPT_PODCAST": "",
            "PROMPT_PODCAST_REFLECTION": "",
            "PROMPT_TITLE": "",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_missing_base_url_raises_error(self) -> None:
        """Missing BASE_URL should raise a validation error."""
        env = {
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
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_missing_llm_endpoint_raises_error(self) -> None:
        """Missing LLM_ENDPOINT should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_missing_llm_api_key_raises_error(self) -> None:
        """Missing LLM_API_KEY should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_missing_llm_model_raises_error(self) -> None:
        """Missing LLM_MODEL should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_missing_prompt_article_raises_error(self) -> None:
        """Missing PROMPT_ARTICLE should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]


class TestOptionalFields:
    """Tests for optional field defaults."""

    def _get_valid_env(self) -> dict[str, str]:
        return {
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
        }

    def test_llm_model_required(self) -> None:
        """LLM_MODEL should be required (no default)."""
        env = self._get_valid_env()
        del env["LLM_MODEL"]
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_llm_model_override(self) -> None:
        """LLM_MODEL should accept a custom model name."""
        env = {**self._get_valid_env(), "LLM_MODEL": "mistralai/Mistral-7B-Instruct"}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.llm_model == "mistralai/Mistral-7B-Instruct"

    @pytest.mark.parametrize("field", ["TRANSCRIPTION_ENDPOINT", "TRANSCRIPTION_API_KEY"])
    def test_transcription_credentials_required(self, field: str) -> None:
        """Remote transcription endpoint and key should both be required."""
        env = self._get_valid_env()
        del env[field]
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    @pytest.mark.parametrize("field", ["TRANSCRIPTION_ENDPOINT", "TRANSCRIPTION_API_KEY"])
    def test_config_loading_fails_without_transcription_credentials(self, field: str) -> None:
        """get_config() should reject missing remote transcription settings."""
        env = self._get_valid_env()
        del env[field]
        import focus_feed.config as config_module

        config_module._config_instance = None
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
            get_config()

    def test_prompt_title_required(self) -> None:
        """PROMPT_TITLE should be required (no default)."""
        env = self._get_valid_env()
        del env["PROMPT_TITLE"]
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_prompt_title_override(self):
        """PROMPT_TITLE should accept a custom prompt."""
        custom_prompt = "Extract key points from this text."
        env = {**self._get_valid_env(), "PROMPT_TITLE": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.prompt_title == custom_prompt

    def test_prompt_article_reflection_optional(self):
        """PROMPT_ARTICLE_REFLECTION should be optional (defaults to None)."""
        env = self._get_valid_env()
        del env["PROMPT_ARTICLE_REFLECTION"]
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.prompt_article_reflection is None

    def test_prompt_podcast_reflection_optional(self):
        """PROMPT_PODCAST_REFLECTION should be optional (defaults to None)."""
        env = self._get_valid_env()
        del env["PROMPT_PODCAST_REFLECTION"]
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.prompt_podcast_reflection is None

    def test_prompt_article_reflection_override(self):
        """PROMPT_ARTICLE_REFLECTION should accept a custom prompt."""
        custom_prompt = "Custom article reflection prompt."
        env = {**self._get_valid_env(), "PROMPT_ARTICLE_REFLECTION": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.prompt_article_reflection == custom_prompt

    def test_prompt_podcast_reflection_override(self):
        """PROMPT_PODCAST_REFLECTION should accept a custom prompt."""
        custom_prompt = "Custom podcast reflection prompt."
        env = {**self._get_valid_env(), "PROMPT_PODCAST_REFLECTION": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.prompt_podcast_reflection == custom_prompt


class TestEnvFileLoading:
    """Tests for loading configuration from .env file."""

    def test_load_from_env_file(self, tmp_path):
        """Config should be loadable from a .env file."""
        env_content = (
            "BASE_URL=https://feed.example.com\n"
            "LLM_ENDPOINT=https://llm.example.com/v1\n"
            "LLM_API_KEY=secret-key\n"
            "LLM_MODEL=my-model\n"
            "PROMPT_ARTICLE=My article prompt\n"
            "PROMPT_ARTICLE_REFLECTION=My article reflection prompt\n"
            "PROMPT_PODCAST=My podcast prompt\n"
            "PROMPT_PODCAST_REFLECTION=My podcast reflection prompt\n"
            "PROMPT_TITLE=My title prompt\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        # Change to tmp_path so .env is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            settings = Settings()  # type: ignore[call-arg]
            assert settings.base_url == "https://feed.example.com"
            assert settings.llm_endpoint == "https://llm.example.com/v1"
            assert settings.llm_api_key == "secret-key"
            assert settings.llm_model == "my-model"
            assert settings.prompt_article == "My article prompt"
            assert settings.prompt_title == "My title prompt"
        finally:
            os.chdir(original_cwd)

    def test_env_vars_override_env_file(self, tmp_path):
        """Environment variables should take precedence over .env file."""
        env_content = (
            "BASE_URL=https://from-env-file.com\n"
            "LLM_ENDPOINT=https://llm.example.com/v1\n"
            "LLM_API_KEY=secret-key\n"
            "LLM_MODEL=model-from-file\n"
            "PROMPT_ARTICLE=Prompt from file\n"
            "PROMPT_ARTICLE_REFLECTION=Reflection prompt from file\n"
            "PROMPT_PODCAST=Podcast prompt from file\n"
            "PROMPT_PODCAST_REFLECTION=Podcast reflection prompt from file\n"
            "PROMPT_TITLE=Prompt from file\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            env_override = {
                "BASE_URL": "https://from-env-var.com",
            }
            with mock.patch.dict(os.environ, env_override, clear=False):
                settings = Settings()  # type: ignore[call-arg]
                assert settings.base_url == "https://from-env-var.com"
                assert settings.llm_model == "model-from-file"
                assert settings.prompt_article == "Prompt from file"
                assert settings.prompt_title == "Prompt from file"
        finally:
            os.chdir(original_cwd)


class TestEnvironmentVariables:
    """Tests for loading configuration from environment variables."""

    def test_load_from_environment_variables(self):
        """Config should be loadable from environment variables directly."""
        env = {
            "BASE_URL": "https://env.example.com",
            "LLM_ENDPOINT": "https://env-llm.example.com/v1",
            "LLM_API_KEY": "env-secret-key",
            "LLM_MODEL": "env-model",
            "PROMPT_ARTICLE": "Custom article prompt from env",
            "PROMPT_ARTICLE_REFLECTION": "Custom article reflection prompt from env",
            "PROMPT_PODCAST": "Custom podcast prompt from env",
            "PROMPT_PODCAST_REFLECTION": "Custom podcast reflection prompt from env",
            "PROMPT_TITLE": "Custom title prompt from env",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
            "TRANSCRIPTION_REMOTE_MODEL": "whisper-1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.base_url == "https://env.example.com"
            assert settings.llm_endpoint == "https://env-llm.example.com/v1"
            assert settings.llm_api_key == "env-secret-key"
            assert settings.llm_model == "env-model"
            assert settings.prompt_article == "Custom article prompt from env"
            assert settings.prompt_title == "Custom title prompt from env"
            assert settings.transcription_endpoint == "https://transcribe.example.com/v1"
            assert settings.transcription_api_key == "transcribe-test-key"
            assert settings.transcription_remote_model == "whisper-1"


class TestSingletonPattern:
    """Tests for the singleton config accessor."""

    def test_get_config_returns_same_instance(self):
        """get_config() should return the same instance on repeated calls."""
        env = {
            "BASE_URL": "https://singleton.example.com",
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
        }
        with mock.patch.dict(os.environ, env, clear=False):
            # Reset singleton
            import focus_feed.config as config_module

            config_module._config_instance = None

            config1 = get_config()
            config2 = get_config()
            assert config1 is config2

    def test_reload_config_creates_new_instance(self):
        """reload_config() should create a new Settings instance."""
        env = {
            "BASE_URL": "https://reload.example.com",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            import focus_feed.config as config_module

            config_module._config_instance = None

            config1 = get_config()
            config2 = reload_config()
            assert config1 is not config2
            assert config2.base_url == "https://reload.example.com"

            # After reload, get_config should return the new instance
            config3 = get_config()
            assert config3 is config2

    def test_singleton_reflects_env_changes_after_reload(self):
        """Singleton should reflect environment changes after reload."""
        env1 = {
            "BASE_URL": "https://first.example.com",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        env2 = {
            "BASE_URL": "https://second.example.com",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }

        import focus_feed.config as config_module

        config_module._config_instance = None

        with mock.patch.dict(os.environ, env1, clear=False):
            config1 = get_config()
            assert config1.base_url == "https://first.example.com"

        with mock.patch.dict(os.environ, env2, clear=False):
            config2 = reload_config()
            assert config2.base_url == "https://second.example.com"

        # Reset for other tests
        config_module._config_instance = None


class TestBackwardsCompatibility:
    """Tests that legacy prompt environment names are not used as fallbacks."""

    def test_legacy_prompt_names_do_not_replace_required_prompts(self):
        """Legacy prompt names should not be used when modern names are missing."""
        env = {
            "BASE_URL": "https://example.com",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "SUMMARY_PROMPT": "Old summary prompt",
            "SUMMARY_REFLECTION_PROMPT": "Old reflection prompt",
            "TITLE_PROMPT": "Old title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_legacy_reflection_prompt_does_not_replace_required_prompts(self):
        """Legacy names do not satisfy missing modern prompt configuration."""
        env = {
            "BASE_URL": "https://example.com",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "SUMMARY_PROMPT": "Old summary prompt",
            "SUMMARY_REFLECTION_PROMPT": "Old reflection prompt",
            "TITLE_PROMPT": "Old title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]


class TestCriticalLogging:
    """Tests for CRITICAL-level logging when required config is missing."""

    def _get_valid_env(self) -> dict[str, str]:
        return {
            "BASE_URL": "https://example.com/feeds",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_TITLE": "Test title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }

    def test_missing_required_field_logs_critical(self, caplog):
        """A CRITICAL log should be emitted when a required field is missing."""
        env = self._get_valid_env()
        env["BASE_URL"] = ""
        with (
            mock.patch.dict(os.environ, env, clear=False),
            caplog.at_level(logging.CRITICAL, logger="focus_feed"),
            pytest.raises(ValidationError),
        ):
            get_config()
        assert any(
            "BASE_URL" in record.message and record.levelno == logging.CRITICAL
            for record in caplog.records
        )

    def test_missing_multiple_required_fields_logs_critical(self, caplog):
        """CRITICAL log should list all missing fields."""
        env = {
            "BASE_URL": "",
            "LLM_ENDPOINT": "",
            "LLM_MODEL": "test-model",
            "PROMPT_TITLE": "Test title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            caplog.at_level(logging.CRITICAL, logger="focus_feed"),
            pytest.raises(ValidationError),
        ):
            get_config()
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) >= 1
        assert "BASE_URL" in critical_records[0].message
        assert "LLM_ENDPOINT" in critical_records[0].message

    def test_all_required_present_no_critical_log(self, caplog):
        """No CRITICAL log should be emitted when all required fields are present."""
        env = self._get_valid_env()
        with (
            mock.patch.dict(os.environ, env, clear=False),
            caplog.at_level(logging.CRITICAL, logger="focus_feed"),
            contextlib.suppress(ValidationError),
        ):
            get_config()
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) == 0

    def test_reload_config_logs_critical_on_missing(self, caplog):
        """reload_config should also log CRITICAL when config is missing."""
        env = self._get_valid_env()
        env["LLM_API_KEY"] = ""
        with (
            mock.patch.dict(os.environ, env, clear=False),
            caplog.at_level(logging.CRITICAL, logger="focus_feed"),
            pytest.raises(ValidationError),
        ):
            reload_config()
        assert any(
            "LLM_API_KEY" in record.message and record.levelno == logging.CRITICAL
            for record in caplog.records
        )
