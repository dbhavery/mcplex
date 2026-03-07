"""Tests for mcplex memory tools — ChromaDB is fully mocked."""

from __future__ import annotations

from unittest import mock

import pytest

from mcplex.config import reset_config
from mcplex.memory_tools import (
    list_collections,
    reset_chroma_client,
    search,
    set_chroma_client,
    store,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    reset_chroma_client()
    yield
    reset_chroma_client()
    reset_config()


def _make_mock_client() -> mock.MagicMock:
    """Create a mock ChromaDB client with get_or_create_collection and friends."""
    mock_collection = mock.MagicMock()
    mock_collection.add = mock.MagicMock()
    mock_collection.query = mock.MagicMock(
        return_value={
            "ids": [["id1", "id2"]],
            "documents": [["doc one", "doc two"]],
            "metadatas": [[{"tag": "a"}, {"tag": "b"}]],
            "distances": [[0.1, 0.5]],
        }
    )

    mock_client = mock.MagicMock()
    mock_client.get_or_create_collection = mock.MagicMock(
        return_value=mock_collection
    )
    mock_client.get_collection = mock.MagicMock(return_value=mock_collection)
    mock_client.list_collections = mock.MagicMock(return_value=["default", "notes"])

    return mock_client


class TestStore:
    """Tests for the store tool."""

    @pytest.mark.asyncio
    async def test_store_returns_id(self) -> None:
        """store adds a document and returns its ID."""
        mock_client = _make_mock_client()
        set_chroma_client(mock_client)

        doc_id = await store("Test document", collection="test_coll")

        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
        mock_client.get_or_create_collection.assert_called_once_with(name="test_coll")
        mock_coll = mock_client.get_or_create_collection.return_value
        mock_coll.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_metadata(self) -> None:
        """store passes metadata to ChromaDB."""
        mock_client = _make_mock_client()
        set_chroma_client(mock_client)

        await store("Text", metadata={"source": "chat", "topic": "test"})

        mock_coll = mock_client.get_or_create_collection.return_value
        call_kwargs = mock_coll.add.call_args.kwargs
        assert call_kwargs["metadatas"] == [{"source": "chat", "topic": "test"}]

    @pytest.mark.asyncio
    async def test_store_with_custom_id(self) -> None:
        """store uses a provided document ID."""
        mock_client = _make_mock_client()
        set_chroma_client(mock_client)

        doc_id = await store("Text", doc_id="custom-123")

        assert doc_id == "custom-123"
        mock_coll = mock_client.get_or_create_collection.return_value
        call_kwargs = mock_coll.add.call_args.kwargs
        assert call_kwargs["ids"] == ["custom-123"]


class TestSearch:
    """Tests for the search tool."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        """search returns structured results from ChromaDB query."""
        mock_client = _make_mock_client()
        set_chroma_client(mock_client)

        results = await search("test query", n_results=2)

        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["document"] == "doc one"
        assert results[0]["metadata"] == {"tag": "a"}
        assert results[0]["distance"] == 0.1
        assert results[1]["id"] == "id2"

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection(self) -> None:
        """search returns empty list for collections that don't exist."""
        mock_client = mock.MagicMock()
        mock_client.get_collection = mock.MagicMock(
            side_effect=ValueError("Collection not found")
        )
        set_chroma_client(mock_client)

        results = await search("query", collection="nonexistent")

        assert results == []


class TestListCollections:
    """Tests for list_collections."""

    @pytest.mark.asyncio
    async def test_list_collections_returns_names(self) -> None:
        """list_collections returns collection names as strings."""
        mock_client = _make_mock_client()
        set_chroma_client(mock_client)

        names = await list_collections()

        assert names == ["default", "notes"]

    @pytest.mark.asyncio
    async def test_list_collections_with_objects(self) -> None:
        """list_collections handles Collection objects with .name attribute."""
        mock_client = mock.MagicMock()
        coll_obj = mock.MagicMock()
        coll_obj.name = "my_collection"
        mock_client.list_collections = mock.MagicMock(return_value=[coll_obj])
        set_chroma_client(mock_client)

        names = await list_collections()

        assert names == ["my_collection"]


class TestLazyImport:
    """Test that ChromaDB is lazily imported."""

    def test_import_error_without_chromadb(self) -> None:
        """_get_chroma_client raises ImportError with helpful message if chromadb missing."""
        reset_chroma_client()
        with mock.patch.dict("sys.modules", {"chromadb": None}):
            from mcplex.memory_tools import _get_chroma_client

            with pytest.raises(ImportError, match="chromadb is required"):
                _get_chroma_client()
