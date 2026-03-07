# mcplex

**MCP server for local AI models** -- expose Ollama, embeddings, vision, and vector memory to Claude Code and other MCP clients.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

---

## What is this?

mcplex is a [Model Context Protocol](https://modelcontextprotocol.io) server that bridges your **local AI models** to any MCP client. It gives Claude Code (or any MCP-compatible tool) direct access to:

- **Ollama models** -- generate text, chat, and list available models
- **Embeddings** -- generate vector embeddings via local embedding models
- **Vision** -- analyze images and extract text using local vision models (LLaVA, etc.)
- **Vector memory** -- store and semantically search text using ChromaDB

Everything runs locally. No API keys needed. No data leaves your machine.

## Features

| Category | Tools | Description |
|----------|-------|-------------|
| **Text Generation** | `generate` | One-shot text generation with any Ollama model |
| **Chat** | `chat` | Multi-turn conversation with message history |
| **Embeddings** | `embed` | Generate vector embeddings for text |
| **Model Management** | `list_models` | List all available Ollama models |
| **Vision** | `analyze_image` | Describe/analyze images with a vision model |
| **OCR** | `ocr_image` | Extract text from images |
| **Memory Store** | `memory_store` | Store text + metadata in ChromaDB |
| **Memory Search** | `memory_search` | Semantic search over stored memories |
| **Memory List** | `memory_list_collections` | List all memory collections |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (default: `http://localhost:11434`)
- At least one Ollama model pulled (e.g., `ollama pull qwen3:8b`)

## Installation

```bash
# From PyPI (when published)
pip install mcplex

# With vector memory support
pip install mcplex[memory]

# From source
git clone https://github.com/dbhavery/mcplex.git
cd mcplex
pip install -e ".[memory,dev]"
```

## Claude Code Integration

Add mcplex to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "mcplex": {
      "command": "mcplex",
      "args": []
    }
  }
}
```

Or if running from source:

```json
{
  "mcpServers": {
    "mcplex": {
      "command": "python",
      "args": ["-m", "mcplex.server"]
    }
  }
}
```

Once configured, Claude Code can use your local models directly:

> "Use the generate tool to summarize this file with qwen3:8b"
>
> "Embed these three paragraphs and store them in the 'research' collection"
>
> "Analyze this screenshot and extract all visible text"

## Tool Reference

### generate

Send a prompt to a local Ollama model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | The text prompt |
| `model` | `str` | `qwen3:8b` | Ollama model name |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `int` | `2048` | Maximum tokens to generate |

### chat

Multi-turn chat with message history.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[{role, content}]` | *required* | Message history |
| `model` | `str` | `qwen3:8b` | Ollama model name |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Maximum tokens |

### embed

Generate vector embeddings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Text to embed |
| `model` | `str` | `nomic-embed-text` | Embedding model |

### list_models

List all available Ollama models. No parameters.

### analyze_image

Analyze an image with a local vision model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` | *required* | Path to image file |
| `prompt` | `str` | `"Describe this image in detail."` | Question/instruction |
| `model` | `str` | `llava` | Vision model name |

### ocr_image

Extract text from an image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` | *required* | Path to image file |
| `model` | `str` | `llava` | Vision model name |

### memory_store

Store text in vector memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Text to store |
| `metadata` | `dict` | `None` | Optional key-value metadata |
| `collection` | `str` | `"default"` | ChromaDB collection name |

### memory_search

Semantic search over stored memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Search query |
| `n_results` | `int` | `5` | Max results to return |
| `collection` | `str` | `"default"` | ChromaDB collection name |

### memory_list_collections

List all ChromaDB collections. No parameters.

## Configuration

All configuration is via environment variables (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `MCPLEX_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MCPLEX_DEFAULT_MODEL` | `qwen3:8b` | Default text model |
| `MCPLEX_EMBED_MODEL` | `nomic-embed-text` | Default embedding model |
| `MCPLEX_VISION_MODEL` | `llava` | Default vision model |
| `MCPLEX_CHROMA_PATH` | `./mcplex_data/chroma` | ChromaDB storage path |
| `MCPLEX_DEFAULT_TEMPERATURE` | `0.7` | Default sampling temperature |
| `MCPLEX_DEFAULT_MAX_TOKENS` | `2048` | Default max tokens |

## Architecture

```
MCP Client (Claude Code, etc.)
    |
    | stdio (JSON-RPC)
    |
mcplex server (FastMCP)
    |
    +-- ollama_tools -----> Ollama API (HTTP)
    |                        localhost:11434
    +-- vision_tools -----> Ollama API (with images)
    |
    +-- memory_tools -----> ChromaDB (local persistent)
```

- **Transport:** stdio (standard for CLI-based MCP clients)
- **Ollama communication:** async HTTP via httpx
- **Vector storage:** ChromaDB with persistent client (lazy-loaded)
- **No API keys required** -- everything runs locally

## Development

```bash
git clone https://github.com/dbhavery/mcplex.git
cd mcplex
pip install -e ".[memory,dev]"

# Run tests
python -m pytest tests/ -v

# Run the server
mcplex
# or
python -m mcplex.server
```

## License

[MIT](LICENSE) -- Copyright (c) 2026 Donald Havery
