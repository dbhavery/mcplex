"""Tests for mcplex Ollama tools — all HTTP calls are mocked."""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from mcplex.config import reset_config
from mcplex.ollama_tools import chat, embed, generate, list_models


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response with JSON data."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://test"),
    )
    return response


class TestGenerate:
    """Tests for the generate tool."""

    @pytest.mark.asyncio
    async def test_generate_basic(self) -> None:
        """generate sends correct payload and returns response text."""
        mock_resp = _mock_response({"response": "Hello, world!"})

        with mock.patch("mcplex.ollama_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            result = await generate("Say hello")

        assert result == "Hello, world!"
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["prompt"] == "Say hello"
        assert payload["model"] == "qwen3:8b"
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_custom_params(self) -> None:
        """generate passes custom model, temperature, and max_tokens."""
        mock_resp = _mock_response({"response": "Custom response"})

        with mock.patch("mcplex.ollama_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            result = await generate(
                "Test", model="llama3.1", temperature=0.2, max_tokens=100
            )

        assert result == "Custom response"
        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert payload["model"] == "llama3.1"
        assert payload["options"]["temperature"] == 0.2
        assert payload["options"]["num_predict"] == 100


class TestChat:
    """Tests for the chat tool."""

    @pytest.mark.asyncio
    async def test_chat_returns_content(self) -> None:
        """chat extracts message content from Ollama response."""
        mock_resp = _mock_response(
            {"message": {"role": "assistant", "content": "I can help with that."}}
        )

        with mock.patch("mcplex.ollama_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            messages = [
                {"role": "user", "content": "Help me"},
            ]
            result = await chat(messages)

        assert result == "I can help with that."
        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert payload["messages"] == messages


class TestEmbed:
    """Tests for the embed tool."""

    @pytest.mark.asyncio
    async def test_embed_single_string(self) -> None:
        """embed sends single text and returns embedding vectors."""
        mock_resp = _mock_response(
            {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}
        )

        with mock.patch("mcplex.ollama_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            result = await embed("hello world")

        assert result == [[0.1, 0.2, 0.3, 0.4]]
        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert payload["input"] == "hello world"
        assert payload["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embed_list_of_strings(self) -> None:
        """embed accepts a list of strings."""
        mock_resp = _mock_response(
            {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        )

        with mock.patch("mcplex.ollama_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            result = await embed(["hello", "world"])

        assert len(result) == 2
        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert payload["input"] == ["hello", "world"]


class TestListModels:
    """Tests for the list_models tool."""

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        """list_models returns parsed model info."""
        mock_resp = _mock_response(
            {
                "models": [
                    {"name": "qwen3:8b", "size": 4_500_000_000},
                    {"name": "llama3.1:8b", "size": 4_700_000_000},
                ]
            }
        )

        with mock.patch("mcplex.ollama_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            models = await list_models()

        assert len(models) == 2
        assert models[0]["name"] == "qwen3:8b"
