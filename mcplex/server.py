"""mcplex MCP server — expose local AI models to any MCP client.

Entry point: mcplex.server:main
Transport: stdio (default)
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcplex import __version__
from mcplex.config import get_config
from mcplex.ollama_tools import chat as ollama_chat
from mcplex.ollama_tools import embed as ollama_embed
from mcplex.ollama_tools import generate as ollama_generate
from mcplex.ollama_tools import list_models as ollama_list_models
from mcplex.memory_tools import list_collections as memory_list_collections
from mcplex.memory_tools import search as memory_search
from mcplex.memory_tools import store as memory_store
from mcplex.vision_tools import analyze_image as vision_analyze
from mcplex.vision_tools import ocr_image as vision_ocr

# ---------------------------------------------------------------------------
# Create the FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="mcplex",
    instructions=(
        "mcplex bridges local AI models to MCP clients. "
        "Use these tools to run Ollama models, generate embeddings, "
        "analyze images with vision models, and store/search vector memories."
    ),
)


# ---------------------------------------------------------------------------
# Ollama tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="generate",
    description=(
        "Send a prompt to a local Ollama model and get a text response. "
        "Good for one-shot completions, summarization, analysis, and code generation."
    ),
)
async def tool_generate(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate text from a local Ollama model.

    Args:
        prompt: The text prompt to send to the model.
        model: Ollama model name (e.g., 'qwen3:8b', 'llama3.1'). Uses server default if omitted.
        temperature: Sampling temperature 0.0-2.0. Lower = more deterministic.
        max_tokens: Maximum tokens to generate.
    """
    return await ollama_generate(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@mcp.tool(
    name="chat",
    description=(
        "Multi-turn chat with a local Ollama model. "
        "Send a list of messages with roles (system/user/assistant) for context-aware responses."
    ),
)
async def tool_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Chat with a local Ollama model using message history.

    Args:
        messages: List of message objects with 'role' (system/user/assistant) and 'content'.
        model: Ollama model name. Uses server default if omitted.
        temperature: Sampling temperature 0.0-2.0.
        max_tokens: Maximum tokens to generate.
    """
    return await ollama_chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@mcp.tool(
    name="embed",
    description=(
        "Generate vector embeddings for text using a local Ollama embedding model. "
        "Useful for semantic similarity, clustering, and feeding into vector databases."
    ),
)
async def tool_embed(
    text: str | list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for one or more text strings.

    Args:
        text: A single string or list of strings to embed.
        model: Embedding model name (e.g., 'nomic-embed-text'). Uses server default if omitted.
    """
    return await ollama_embed(text=text, model=model)


@mcp.tool(
    name="list_models",
    description="List all models currently available in the local Ollama instance.",
)
async def tool_list_models() -> str:
    """List available Ollama models with their sizes and modification dates."""
    models = await ollama_list_models()
    if not models:
        return "No models found. Is Ollama running?"
    lines = []
    for m in models:
        name = m.get("name", "unknown")
        size_gb = m.get("size", 0) / (1024 ** 3)
        lines.append(f"  {name} ({size_gb:.1f} GB)")
    return "Available models:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Vision tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="analyze_image",
    description=(
        "Analyze an image using a local vision model (e.g., LLaVA). "
        "Describe contents, answer questions about the image, or extract information."
    ),
)
async def tool_analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    model: str | None = None,
) -> str:
    """Analyze an image with a local vision model.

    Args:
        image_path: Absolute path to the image file (.png, .jpg, .gif, .bmp, .webp).
        prompt: Question or instruction for the vision model.
        model: Vision model name (e.g., 'llava', 'bakllava'). Uses server default if omitted.
    """
    return await vision_analyze(
        image_path=image_path, prompt=prompt, model=model
    )


@mcp.tool(
    name="ocr_image",
    description=(
        "Extract text from an image using a local vision model. "
        "Works with screenshots, documents, signs, handwriting, etc."
    ),
)
async def tool_ocr_image(
    image_path: str,
    model: str | None = None,
) -> str:
    """Extract text content from an image.

    Args:
        image_path: Absolute path to the image file.
        model: Vision model name. Uses server default if omitted.
    """
    return await vision_ocr(image_path=image_path, model=model)


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="memory_store",
    description=(
        "Store text in a local ChromaDB vector database for later semantic search. "
        "Attach optional metadata (tags, source, timestamp, etc.)."
    ),
)
async def tool_memory_store(
    text: str,
    metadata: dict[str, Any] | None = None,
    collection: str = "default",
) -> str:
    """Store text with optional metadata in vector memory.

    Args:
        text: The text content to store.
        metadata: Optional key-value metadata dict (e.g., {"source": "chat", "topic": "python"}).
        collection: ChromaDB collection name. Defaults to 'default'.
    """
    doc_id = await memory_store(
        text=text, metadata=metadata, collection=collection
    )
    return f"Stored in collection '{collection}' with id: {doc_id}"


@mcp.tool(
    name="memory_search",
    description=(
        "Semantic search over stored memories in a ChromaDB collection. "
        "Returns the most similar stored texts ranked by relevance."
    ),
)
async def tool_memory_search(
    query: str,
    n_results: int = 5,
    collection: str = "default",
) -> str:
    """Search vector memory for semantically similar content.

    Args:
        query: The search query text.
        n_results: Maximum number of results to return (default 5).
        collection: ChromaDB collection name. Defaults to 'default'.
    """
    results = await memory_search(
        query=query, n_results=n_results, collection=collection
    )
    if not results:
        return f"No results found in collection '{collection}'."

    lines = []
    for i, r in enumerate(results, 1):
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        meta = r.get("metadata", {})
        meta_str = f" | metadata: {json.dumps(meta)}" if meta else ""
        lines.append(f"{i}. [{r['id']}]{dist_str}{meta_str}\n   {r['document']}")
    return "\n\n".join(lines)


@mcp.tool(
    name="memory_list_collections",
    description="List all ChromaDB vector memory collections.",
)
async def tool_memory_list_collections() -> str:
    """List all available memory collections."""
    collections = await memory_list_collections()
    if not collections:
        return "No collections found."
    return "Collections:\n" + "\n".join(f"  - {c}" for c in collections)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the mcplex MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
