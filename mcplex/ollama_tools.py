"""Ollama integration tools for mcplex."""

from __future__ import annotations

import json
from typing import Any

import httpx

from mcplex.config import get_config


async def generate(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a prompt to a local Ollama model and return the response.

    Args:
        prompt: The text prompt to send to the model.
        model: Ollama model name. Defaults to config default_model.
        temperature: Sampling temperature (0.0-2.0). Defaults to config default.
        max_tokens: Maximum tokens to generate. Defaults to config default.

    Returns:
        The model's generated text response.
    """
    cfg = get_config()
    model = model or cfg.default_model
    temperature = temperature if temperature is not None else cfg.default_temperature
    max_tokens = max_tokens if max_tokens is not None else cfg.default_max_tokens

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{cfg.ollama_url}/api/generate", json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Multi-turn chat with a local Ollama model.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
            Roles: 'system', 'user', 'assistant'.
        model: Ollama model name. Defaults to config default_model.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.

    Returns:
        The assistant's response text.
    """
    cfg = get_config()
    model = model or cfg.default_model
    temperature = temperature if temperature is not None else cfg.default_temperature
    max_tokens = max_tokens if max_tokens is not None else cfg.default_max_tokens

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{cfg.ollama_url}/api/chat", json=payload
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        return message.get("content", "")


async def embed(
    text: str | list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for text using a local Ollama model.

    Args:
        text: A single string or list of strings to embed.
        model: Embedding model name. Defaults to config default_embed_model.

    Returns:
        List of embedding vectors (list of floats).
    """
    cfg = get_config()
    model = model or cfg.default_embed_model

    # Ollama /api/embed accepts "input" as string or list
    input_val = text if isinstance(text, list) else text

    payload: dict[str, Any] = {
        "model": model,
        "input": input_val,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{cfg.ollama_url}/api/embed", json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])


async def list_models() -> list[dict[str, Any]]:
    """List all models available in the local Ollama instance.

    Returns:
        List of model info dicts with keys like 'name', 'size', 'modified_at', etc.
    """
    cfg = get_config()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{cfg.ollama_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
