"""Vision model tools for mcplex — image analysis via local Ollama vision models."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from mcplex.config import get_config

# Supported image extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


def _validate_image_path(image_path: str) -> Path:
    """Validate that the image path exists and has a supported extension.

    Args:
        image_path: Path to the image file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def _encode_image(path: Path) -> str:
    """Read and base64-encode an image file.

    Args:
        path: Path to the image file.

    Returns:
        Base64-encoded string of the image bytes.
    """
    return base64.b64encode(path.read_bytes()).decode("utf-8")


async def analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    model: str | None = None,
) -> str:
    """Analyze an image using a local vision model via Ollama.

    Args:
        image_path: Absolute or relative path to the image file.
        prompt: The question or instruction for the vision model.
            Defaults to a general description request.
        model: Vision model name (e.g., 'llava', 'llava:13b', 'bakllava').
            Defaults to config default_vision_model.

    Returns:
        The model's text description/analysis of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image format is not supported.
    """
    cfg = get_config()
    model = model or cfg.default_vision_model

    path = _validate_image_path(image_path)
    image_b64 = _encode_image(path)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{cfg.ollama_url}/api/generate", json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


async def ocr_image(
    image_path: str,
    model: str | None = None,
) -> str:
    """Extract text content from an image using a local vision model.

    Uses an OCR-focused prompt to extract all visible text from the image.

    Args:
        image_path: Absolute or relative path to the image file.
        model: Vision model name. Defaults to config default_vision_model.

    Returns:
        Extracted text content from the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image format is not supported.
    """
    ocr_prompt = (
        "Extract ALL text visible in this image. Return only the text content, "
        "preserving the original layout and structure as closely as possible. "
        "If no text is visible, respond with 'No text found in image.'"
    )
    return await analyze_image(
        image_path=image_path,
        prompt=ocr_prompt,
        model=model,
    )
