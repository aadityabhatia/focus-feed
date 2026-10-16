"""Pytest configuration for local-feed-generator tests."""

import pytest
from unittest import mock

@pytest.fixture(autouse=True)
def disable_env_file_loading(request):
    """Disable loading .env file for tests that check config validation."""
    # Skip patching for env file loading tests
    test_node = request.node.nodeid
    if "TestEnvFileLoading" in test_node:
        yield
        return
    # Patch Settings model_config to use a non-existent env file
    from local_feed_generator.config import Settings
    original_config = Settings.model_config.copy()
    # Use a path that doesn't exist
    with mock.patch.object(Settings, 'model_config', {**original_config, 'env_file': '/nonexistent/.env'}):
        yield
