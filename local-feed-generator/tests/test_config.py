"""Tests for the configuration system."""

import os
from unittest import mock

from local_feed_generator.config import Settings, get_config, reload_config
from pydantic import ValidationError
import pytest


class TestRequiredFields:
    """Tests for required field validation."""

    def _get_valid_env(self):
        return {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }

    def test_missing_all_required_fields_raises_error(self):
        """Config validation should fail when all required fields are missing."""
        env = {
            "BASE_URL": "",
            "DATA_DIR": "",
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
            Settings()

    def test_missing_base_url_raises_error(self):
        """Missing BASE_URL should raise a validation error."""
        env = {
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_data_dir_raises_error(self):
        """Missing DATA_DIR should raise a validation error."""
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
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_llm_endpoint_raises_error(self):
        """Missing LLM_ENDPOINT should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_llm_api_key_raises_error(self):
        """Missing LLM_API_KEY should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_llm_model_raises_error(self):
        """Missing LLM_MODEL should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_prompt_article_raises_error(self):
        """Missing PROMPT_ARTICLE should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()


class TestOptionalFields:
    """Tests for optional field defaults."""

    def _get_valid_env(self):
        return {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }

    def test_llm_model_required(self):
        """LLM_MODEL should be required (no default)."""
        env = self._get_valid_env()
        del env["LLM_MODEL"]
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_llm_model_override(self):
        """LLM_MODEL should accept a custom model name."""
        env = {**self._get_valid_env(), "LLM_MODEL": "mistralai/Mistral-7B-Instruct"}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.llm_model == "mistralai/Mistral-7B-Instruct"

    def test_transcription_model_default(self):
        """TRANSCRIPTION_MODEL should default to 'large-v3-turbo'."""
        with mock.patch.dict(os.environ, self._get_valid_env(), clear=False):
            settings = Settings()
            assert settings.transcription_model == "large-v3-turbo"

    def test_transcription_device_default(self):
        """TRANSCRIPTION_DEVICE should default to 'cuda'."""
        with mock.patch.dict(os.environ, self._get_valid_env(), clear=False):
            settings = Settings()
            assert settings.transcription_device == "cuda"

    def test_transcription_compute_type_default(self):
        """TRANSCRIPTION_COMPUTE_TYPE should default to 'float16'."""
        with mock.patch.dict(os.environ, self._get_valid_env(), clear=False):
            settings = Settings()
            assert settings.transcription_compute_type == "float16"

    def test_prompt_title_required(self):
        """PROMPT_TITLE should be required (no default)."""
        env = self._get_valid_env()
        del env["PROMPT_TITLE"]
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_prompt_title_override(self):
        """PROMPT_TITLE should accept a custom prompt."""
        custom_prompt = "Extract key points from this text."
        env = {**self._get_valid_env(), "PROMPT_TITLE": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_title == custom_prompt

    def test_prompt_article_reflection_optional(self):
        """PROMPT_ARTICLE_REFLECTION should be optional (defaults to None)."""
        env = self._get_valid_env()
        del env["PROMPT_ARTICLE_REFLECTION"]
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_article_reflection is None

    def test_prompt_podcast_reflection_optional(self):
        """PROMPT_PODCAST_REFLECTION should be optional (defaults to None)."""
        env = self._get_valid_env()
        del env["PROMPT_PODCAST_REFLECTION"]
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_podcast_reflection is None

    def test_prompt_article_reflection_override(self):
        """PROMPT_ARTICLE_REFLECTION should accept a custom prompt."""
        custom_prompt = "Custom article reflection prompt."
        env = {**self._get_valid_env(), "PROMPT_ARTICLE_REFLECTION": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_article_reflection == custom_prompt

    def test_prompt_podcast_reflection_override(self):
        """PROMPT_PODCAST_REFLECTION should accept a custom prompt."""
        custom_prompt = "Custom podcast reflection prompt."
        env = {**self._get_valid_env(), "PROMPT_PODCAST_REFLECTION": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_podcast_reflection == custom_prompt


class TestEnvFileLoading:
    """Tests for loading configuration from .env file."""

    def test_load_from_env_file(self, tmp_path):
        """Config should be loadable from a .env file."""
        env_content = (
            "BASE_URL=https://feed.example.com\n"
            "DATA_DIR=./mydata\n"
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
            settings = Settings()
            assert settings.base_url == "https://feed.example.com"
            assert settings.data_dir == "./mydata"
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
            "DATA_DIR=./from-env-file\n"
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
                settings = Settings()
                assert settings.base_url == "https://from-env-var.com"
                assert settings.data_dir == "./from-env-file"
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
            "DATA_DIR": "/tmp/envdata",
            "LLM_ENDPOINT": "https://env-llm.example.com/v1",
            "LLM_API_KEY": "env-secret-key",
            "LLM_MODEL": "env-model",
            "PROMPT_ARTICLE": "Custom article prompt from env",
            "PROMPT_ARTICLE_REFLECTION": "Custom article reflection prompt from env",
            "PROMPT_PODCAST": "Custom podcast prompt from env",
            "PROMPT_PODCAST_REFLECTION": "Custom podcast reflection prompt from env",
            "PROMPT_TITLE": "Custom title prompt from env",
            "TRANSCRIPTION_MODEL": "tiny",
            "TRANSCRIPTION_DEVICE": "cpu",
            "TRANSCRIPTION_COMPUTE_TYPE": "int8",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.base_url == "https://env.example.com"
            assert settings.data_dir == "/tmp/envdata"
            assert settings.llm_endpoint == "https://env-llm.example.com/v1"
            assert settings.llm_api_key == "env-secret-key"
            assert settings.llm_model == "env-model"
            assert settings.prompt_article == "Custom article prompt from env"
            assert settings.prompt_title == "Custom title prompt from env"
            assert settings.transcription_model == "tiny"
            assert settings.transcription_device == "cpu"
            assert settings.transcription_compute_type == "int8"


class TestSingletonPattern:
    """Tests for the singleton config accessor."""

    def test_get_config_returns_same_instance(self):
        """get_config() should return the same instance on repeated calls."""
        env = {
            "BASE_URL": "https://singleton.example.com",
            "DATA_DIR": "./data",
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
            # Reset singleton
            import local_feed_generator.config as config_module

            config_module._config_instance = None

            config1 = get_config()
            config2 = get_config()
            assert config1 is config2

    def test_reload_config_creates_new_instance(self):
        """reload_config() should create a new Settings instance."""
        env = {
            "BASE_URL": "https://reload.example.com",
            "DATA_DIR": "./data",
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
            import local_feed_generator.config as config_module

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
            "DATA_DIR": "./data",
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
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
        }

        import local_feed_generator.config as config_module

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
    """Tests for backwards compatibility with old environment variable names."""

    def test_fallback_to_summary_prompt_for_article(self):
        """If PROMPT_ARTICLE is unset but SUMMARY_PROMPT is set, use the old value."""
        env = {
            "BASE_URL": "https://example.com",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "SUMMARY_PROMPT": "Old summary prompt",
            "SUMMARY_REFLECTION_PROMPT": "Old reflection prompt",
            "TITLE_PROMPT": "Old title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_article == "Old summary prompt"
            assert settings.prompt_podcast == "Old summary prompt"
            assert settings.prompt_title == "Old title prompt"

    def test_fallback_to_summary_reflection_prompt(self):
        """If PROMPT_*_REFLECTION is unset but SUMMARY_REFLECTION_PROMPT is set, use old value."""
        env = {
            "BASE_URL": "https://example.com",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "SUMMARY_PROMPT": "Old summary prompt",
            "SUMMARY_REFLECTION_PROMPT": "Old reflection prompt",
            "TITLE_PROMPT": "Old title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_article_reflection == "Old reflection prompt"
            assert settings.prompt_podcast_reflection == "Old reflection prompt"

    def test_new_vars_take_priority_over_old(self):
        """New env vars should take priority over old ones."""
        env = {
            "BASE_URL": "https://example.com",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "SUMMARY_PROMPT": "Old summary prompt",
            "SUMMARY_REFLECTION_PROMPT": "Old reflection prompt",
            "TITLE_PROMPT": "Old title prompt",
            "PROMPT_ARTICLE": "New article prompt",
            "PROMPT_PODCAST": "New podcast prompt",
            "PROMPT_TITLE": "New title prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.prompt_article == "New article prompt"
            assert settings.prompt_podcast == "New podcast prompt"
            assert settings.prompt_title == "New title prompt"
