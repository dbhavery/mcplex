# McPlex

MCP server that bridges local Ollama models to Claude Code and other MCP clients -- text generation, embeddings, vision, and vector memory, all running locally.

## Why I Built This

Claude Code is powerful but cloud-only. Local models via Ollama are private and free but disconnected from MCP tooling. I needed a bridge: expose local models as MCP tools so Claude Code can delegate tasks to local inference (summarization, embedding, image analysis) without API costs or data leaving my machine. Any MCP-compatible client (Claude Code, Cursor, etc.) gets access with zero custom integration.

## What It Does

- **9 MCP tools** -- `generate`, `chat`, `embed`, `list_models`, `analyze_image`, `ocr_image`, `memory_store`, `memory_search`, `memory_list_collections`
- **Zero cloud dependency** -- all inference runs locally via Ollama; no API keys needed
- **ChromaDB vector memory** -- store and semantically search text with persistent local storage
- **Vision and OCR** -- analyze images and extract text using local vision models (LLaVA)
- **Drop-in MCP config** -- add 3 lines to Claude Code's MCP config and local models are available immediately

## Key Technical Decisions

- **MCP protocol over custom API** -- standard protocol means any MCP client works without custom integration code. When a new MCP client launches, McPlex works with it automatically.
- **Ollama over vLLM** -- simpler setup, built-in model management (`ollama pull`), runs on consumer hardware. vLLM is faster at scale but requires manual model configuration and more VRAM.
- **Lazy ChromaDB loading** -- memory tools are optional. Core text/vision tools work without ChromaDB installed. `pip install mcplex[memory]` adds vector storage only when needed.
- **Async HTTP via httpx** -- non-blocking Ollama API calls. Multiple tools can query different models concurrently without blocking the MCP event loop.

## Quick Start

```bash
pip install mcplex              # Core (text + vision)
pip install mcplex[memory]      # With ChromaDB vector memory

# Requires Ollama running locally
ollama pull qwen3:8b            # Pull a model
```

Add to Claude Code MCP config:

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

Then ask Claude Code: *"Use the generate tool to summarize this file with qwen3:8b"*

Configuration via environment variables:

```bash
MCPLEX_OLLAMA_URL=http://localhost:11434
MCPLEX_DEFAULT_MODEL=qwen3:8b
MCPLEX_CHROMA_PATH=./mcplex_data/chroma
```

## Lessons Learned

**MCP tool schema design matters more than implementation quality.** Overly flexible schemas (e.g., a single `query` tool that accepts model, prompt, temperature, max_tokens, format, and system prompt) confuse LLM clients -- they don't know which parameters to set. Specific, well-documented tool signatures with sensible defaults (`generate` takes a prompt and optional model) produce much better tool-calling accuracy. I went through three schema iterations before landing on the current 9-tool design, and each simplification improved Claude Code's ability to use the tools correctly.

## Tests

```bash
pip install -e ".[memory,dev]"
pytest tests/ -v    # 24 tests
```

---

MIT License. See [LICENSE](LICENSE).
