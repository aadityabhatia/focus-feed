"""Tests for the configuration system."""

import os
from unittest import mock

from pydantic import ValidationError
import pytest

from local_feed_generator.config import Settings, get_config, reload_config


class TestRequiredFields:
    """Tests for required field validation."""

    def _get_valid_env(self):
        return {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_SYSTEM_PROMPT": "Test system prompt",
        }

    def test_missing_all_required_fields_raises_error(self):
        """Config validation should fail when all required fields are missing."""
        env = {
            "BASE_URL": "",
            "DATA_DIR": "",
            "LLM_ENDPOINT": "",
            "LLM_API_KEY": "",
            "LLM_MODEL": "",
            "LLM_SYSTEM_PROMPT": "",
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
            "LLM_SYSTEM_PROMPT": "Test system prompt",
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
            "LLM_SYSTEM_PROMPT": "Test system prompt",
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
            "LLM_SYSTEM_PROMPT": "Test system prompt",
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
            "LLM_SYSTEM_PROMPT": "Test system prompt",
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
            "LLM_SYSTEM_PROMPT": "Test system prompt",
        }
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_llm_system_prompt_raises_error(self):
        """Missing LLM_SYSTEM_PROMPT should raise a validation error."""
        env = {
            "BASE_URL": "https://example.com/feeds",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
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
            "LLM_SYSTEM_PROMPT": "Test system prompt",
        }

    def test_feed_format_default(self):
        """FEED_FORMAT should default to 'rss'."""
        with mock.patch.dict(os.environ, self._get_valid_env(), clear=False):
            settings = Settings()
            assert settings.feed_format == "rss"

    def test_feed_format_override(self):
        """FEED_FORMAT should accept 'atom' as a value."""
        env = {**self._get_valid_env(), "FEED_FORMAT": "atom"}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.feed_format == "atom"

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

    def test_llm_system_prompt_required(self):
        """LLM_SYSTEM_PROMPT should be required (no default)."""
        env = self._get_valid_env()
        del env["LLM_SYSTEM_PROMPT"]
        with mock.patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_llm_system_prompt_override(self):
        """LLM_SYSTEM_PROMPT should accept a custom prompt."""
        custom_prompt = "Extract key points from this text."
        env = {**self._get_valid_env(), "LLM_SYSTEM_PROMPT": custom_prompt}
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.llm_system_prompt == custom_prompt


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
            "LLM_SYSTEM_PROMPT=My custom prompt\n"
            "FEED_FORMAT=atom\n"
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
            assert settings.llm_system_prompt == "My custom prompt"
            assert settings.feed_format == "atom"
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
            "LLM_SYSTEM_PROMPT=Prompt from file\n"
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
                assert settings.llm_system_prompt == "Prompt from file"
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
            "LLM_SYSTEM_PROMPT": "Custom prompt from env",
            "FEED_FORMAT": "atom",
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
            assert settings.llm_system_prompt == "Custom prompt from env"
            assert settings.feed_format == "atom"
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
            "LLM_SYSTEM_PROMPT": "Test prompt",
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
            "LLM_SYSTEM_PROMPT": "Test prompt",
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
            "LLM_SYSTEM_PROMPT": "Test prompt",
        }
        env2 = {
            "BASE_URL": "https://second.example.com",
            "DATA_DIR": "./data",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_SYSTEM_PROMPT": "Test prompt",
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
