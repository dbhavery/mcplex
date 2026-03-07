"""Tests for mcplex configuration."""

import os
from unittest import mock

import pytest

from mcplex.config import McplexConfig, get_config, reset_config


@pytest.fixture(autouse=True)
def _reset():
    """Reset config singleton before each test."""
    reset_config()
    yield
    reset_config()


class TestMcplexConfig:
    """Test McplexConfig dataclass defaults and env var loading."""

    def test_default_values(self) -> None:
        """Config uses sensible defaults when no env vars are set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = McplexConfig()
        assert cfg.ollama_url == "http://localhost:11434"
        assert cfg.default_model == "qwen3:8b"
        assert cfg.default_embed_model == "nomic-embed-text"
        assert cfg.default_vision_model == "llava"
        assert cfg.chroma_path == "./mcplex_data/chroma"
        assert cfg.default_temperature == 0.7
        assert cfg.default_max_tokens == 2048

    def test_env_var_override(self) -> None:
        """Config loads values from environment variables."""
        env = {
            "MCPLEX_OLLAMA_URL": "http://gpu-server:11434",
            "MCPLEX_DEFAULT_MODEL": "llama3.1:70b",
            "MCPLEX_EMBED_MODEL": "mxbai-embed-large",
            "MCPLEX_VISION_MODEL": "bakllava",
            "MCPLEX_CHROMA_PATH": "/data/chroma",
            "MCPLEX_DEFAULT_TEMPERATURE": "0.3",
            "MCPLEX_DEFAULT_MAX_TOKENS": "4096",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = McplexConfig()
        assert cfg.ollama_url == "http://gpu-server:11434"
        assert cfg.default_model == "llama3.1:70b"
        assert cfg.default_embed_model == "mxbai-embed-large"
        assert cfg.default_vision_model == "bakllava"
        assert cfg.chroma_path == "/data/chroma"
        assert cfg.default_temperature == 0.3
        assert cfg.default_max_tokens == 4096

    def test_get_config_singleton(self) -> None:
        """get_config returns the same instance on repeated calls."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config(self) -> None:
        """reset_config clears the singleton so next call creates a new one."""
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2
