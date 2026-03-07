"""Tests for mcplex vision tools — file validation and mocked Ollama calls."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import httpx
import pytest

from mcplex.config import reset_config
from mcplex.vision_tools import (
    SUPPORTED_EXTENSIONS,
    _validate_image_path,
    analyze_image,
    ocr_image,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://test"),
    )


class TestValidateImagePath:
    """Tests for image path validation."""

    def test_nonexistent_file_raises(self) -> None:
        """FileNotFoundError for paths that don't exist."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            _validate_image_path("/nonexistent/image.png")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """ValueError for unsupported file extensions."""
        bad_file = tmp_path / "document.pdf"
        bad_file.write_text("fake pdf")
        with pytest.raises(ValueError, match="Unsupported image format"):
            _validate_image_path(str(bad_file))

    def test_valid_image_path(self, tmp_path: Path) -> None:
        """Valid image paths pass validation."""
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]:
            img_file = tmp_path / f"test{ext}"
            img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            result = _validate_image_path(str(img_file))
            assert result.exists()


class TestAnalyzeImage:
    """Tests for analyze_image with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_analyze_image_sends_base64(self, tmp_path: Path) -> None:
        """analyze_image reads the file, base64 encodes it, and sends to Ollama."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG" + b"\x00" * 50)

        mock_resp = _mock_response({"response": "A test image with black pixels."})

        with mock.patch("mcplex.vision_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            result = await analyze_image(str(img_file), prompt="What is this?")

        assert result == "A test image with black pixels."
        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert payload["model"] == "llava"
        assert payload["prompt"] == "What is this?"
        assert len(payload["images"]) == 1
        assert isinstance(payload["images"][0], str)  # base64 string

    @pytest.mark.asyncio
    async def test_analyze_image_file_not_found(self) -> None:
        """analyze_image raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            await analyze_image("/does/not/exist.png")


class TestOcrImage:
    """Tests for ocr_image."""

    @pytest.mark.asyncio
    async def test_ocr_uses_ocr_prompt(self, tmp_path: Path) -> None:
        """ocr_image sends an OCR-focused prompt to the vision model."""
        img_file = tmp_path / "screenshot.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        mock_resp = _mock_response({"response": "Hello World\nLine 2"})

        with mock.patch("mcplex.vision_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            result = await ocr_image(str(img_file))

        assert result == "Hello World\nLine 2"
        payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
        assert "Extract ALL text" in payload["prompt"]
