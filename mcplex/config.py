"""Configuration for mcplex server."""

import os
from dataclasses import dataclass, field


@dataclass
class McplexConfig:
    """Server configuration, loaded from environment variables with sensible defaults."""

    ollama_url: str = field(
        default_factory=lambda: os.environ.get(
            "MCPLEX_OLLAMA_URL", "http://localhost:11434"
        )
    )
    default_model: str = field(
        default_factory=lambda: os.environ.get("MCPLEX_DEFAULT_MODEL", "qwen3:8b")
    )
    default_embed_model: str = field(
        default_factory=lambda: os.environ.get(
            "MCPLEX_EMBED_MODEL", "nomic-embed-text"
        )
    )
    default_vision_model: str = field(
        default_factory=lambda: os.environ.get("MCPLEX_VISION_MODEL", "llava")
    )
    chroma_path: str = field(
        default_factory=lambda: os.environ.get(
            "MCPLEX_CHROMA_PATH", "./mcplex_data/chroma"
        )
    )
    default_temperature: float = field(
        default_factory=lambda: float(
            os.environ.get("MCPLEX_DEFAULT_TEMPERATURE", "0.7")
        )
    )
    default_max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get("MCPLEX_DEFAULT_MAX_TOKENS", "2048")
        )
    )


# Singleton config instance
_config: McplexConfig | None = None


def get_config() -> McplexConfig:
    """Return the global config singleton, creating it on first call."""
    global _config
    if _config is None:
        _config = McplexConfig()
    return _config


def reset_config() -> None:
    """Reset the global config singleton. Useful for testing."""
    global _config
    _config = None
