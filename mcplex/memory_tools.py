"""Vector memory tools for mcplex — ChromaDB-backed semantic storage and search.

ChromaDB is lazily imported: it is only loaded when a memory tool is actually called.
This keeps startup fast and allows the server to run without chromadb installed
(the memory tools will raise ImportError if used without it).
"""

from __future__ import annotations

import uuid
from typing import Any

from mcplex.config import get_config

# Lazy-loaded ChromaDB client
_chroma_client: Any = None


def _get_chroma_client() -> Any:
    """Return a persistent ChromaDB client, creating it on first call.

    Raises:
        ImportError: If chromadb is not installed.
    """
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for memory tools. "
                "Install it with: pip install mcplex[memory]"
            ) from exc
        cfg = get_config()
        _chroma_client = chromadb.PersistentClient(path=cfg.chroma_path)
    return _chroma_client


def reset_chroma_client() -> None:
    """Reset the ChromaDB client singleton. Useful for testing."""
    global _chroma_client
    _chroma_client = None


def set_chroma_client(client: Any) -> None:
    """Inject a ChromaDB client. Useful for testing with mocks.

    Args:
        client: A ChromaDB client or mock object.
    """
    global _chroma_client
    _chroma_client = client


async def store(
    text: str,
    metadata: dict[str, Any] | None = None,
    collection: str = "default",
    doc_id: str | None = None,
) -> str:
    """Store text with optional metadata in a ChromaDB collection.

    Args:
        text: The text content to store.
        metadata: Optional key-value metadata to associate with the text.
        collection: Name of the ChromaDB collection. Defaults to 'default'.
        doc_id: Optional document ID. Auto-generated if not provided.

    Returns:
        The document ID of the stored entry.
    """
    client = _get_chroma_client()
    coll = client.get_or_create_collection(name=collection)

    doc_id = doc_id or str(uuid.uuid4())
    add_kwargs: dict[str, Any] = {
        "documents": [text],
        "ids": [doc_id],
    }
    if metadata:
        add_kwargs["metadatas"] = [metadata]

    coll.add(**add_kwargs)
    return doc_id


async def search(
    query: str,
    n_results: int = 5,
    collection: str = "default",
) -> list[dict[str, Any]]:
    """Semantic search over stored memories in a ChromaDB collection.

    Args:
        query: The search query text.
        n_results: Maximum number of results to return. Defaults to 5.
        collection: Name of the ChromaDB collection. Defaults to 'default'.

    Returns:
        List of result dicts, each containing 'id', 'document', 'metadata',
        and 'distance' keys.
    """
    client = _get_chroma_client()

    try:
        coll = client.get_collection(name=collection)
    except Exception:
        return []

    results = coll.query(query_texts=[query], n_results=n_results)

    output: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, doc_id in enumerate(ids):
        output.append(
            {
                "id": doc_id,
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else None,
            }
        )

    return output


async def list_collections() -> list[str]:
    """List all ChromaDB collection names.

    Returns:
        List of collection name strings.
    """
    client = _get_chroma_client()
    collections = client.list_collections()
    # ChromaDB may return Collection objects or strings depending on version
    names: list[str] = []
    for c in collections:
        if isinstance(c, str):
            names.append(c)
        elif hasattr(c, "name"):
            names.append(c.name)
        else:
            names.append(str(c))
    return names
