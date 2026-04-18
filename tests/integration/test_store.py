"""Integration tests for document store backends.

Chroma tests use tmp_path — no Docker required. Runs in CI.
Weaviate tests are skipped automatically when Docker is not running.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from rag.interfaces.store import Chunk, DocumentStoreProtocol


def _no_weaviate_running() -> bool:
    """Check if Weaviate is reachable on localhost:8080."""
    try:
        with socket.create_connection(("localhost", 8080), timeout=1):
            return False
    except OSError:
        return True


# ---------------------------------------------------------------------------
# Chroma tests — no Docker required
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestChromaStore:
    """Integration tests for ChromaStore using a temporary directory."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> Any:
        from rag.store.chroma import ChromaStore

        return ChromaStore(
            persist_dir=tmp_path / "chroma",
            collection_name="test",
            dimensions=4,
        )

    def _make_child(self, parent_id: str, idx: int) -> Chunk:
        return Chunk(
            id=f"child-{idx}",
            text=f"child text {idx}",
            metadata={"source": "test.pdf", "page": idx},
            parent_id=parent_id,
            token_count=10,
        )

    def _make_parent(self, idx: int) -> Chunk:
        return Chunk(
            id=f"parent-{idx}",
            text=f"parent text {idx}",
            metadata={"source": "test.pdf", "page": idx},
            parent_id=None,
            token_count=50,
        )

    def test_upsert_and_search(self, store: Any) -> None:
        """Round-trip: upsert child chunks and retrieve via semantic search."""
        parent = self._make_parent(0)
        child = self._make_child("parent-0", 0)
        embedding = [0.1, 0.2, 0.3, 0.4]

        store.upsert_chunks([parent])
        store.upsert_chunks([child], embeddings=[embedding])

        results = store.semantic_search(embedding, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "child-0"

    def test_parent_fetch(self, store: Any) -> None:
        """Upsert parents then fetch by ID."""
        parent = self._make_parent(1)
        store.upsert_chunks([parent])

        results = store.fetch_parents(["parent-1"])
        assert len(results) == 1
        assert results[0]["id"] == "parent-1"
        assert results[0]["text"] == "parent text 1"

    def test_bm25_corpus_persistence(self, store: Any) -> None:
        """store_bm25_corpus → load_bm25_corpus round-trip."""
        corpus = [{"id": "1", "text": "hello world"}, {"id": "2", "text": "foo bar"}]
        store.store_bm25_corpus(corpus)
        loaded = store.load_bm25_corpus()
        assert loaded == corpus

    def test_load_bm25_corpus_empty(self, store: Any) -> None:
        """load_bm25_corpus returns [] when no corpus has been stored."""
        result = store.load_bm25_corpus()
        assert result == []

    def test_protocol_compliance(self, store: Any) -> None:
        """ChromaStore must satisfy DocumentStoreProtocol."""
        assert isinstance(store, DocumentStoreProtocol)

    def test_parents_not_in_search_results(self, store: Any) -> None:
        """Parent chunks must never appear in semantic_search results."""
        parent = self._make_parent(2)
        child = self._make_child("parent-2", 2)
        embedding = [0.9, 0.1, 0.0, 0.0]

        store.upsert_chunks([parent])
        store.upsert_chunks([child], embeddings=[embedding])

        results = store.semantic_search(embedding, top_k=10)
        result_ids = {r["id"] for r in results}
        assert "parent-2" not in result_ids, "Parent chunks must not appear in ANN results"


# ---------------------------------------------------------------------------
# Weaviate tests — skipped when Docker is not running
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(_no_weaviate_running(), reason="Weaviate not available on localhost:8080")
class TestWeaviateStore:
    """Integration tests for WeaviateStore. Requires live Weaviate instance."""

    @pytest.fixture
    def store(self) -> Any:
        from rag.store.weaviate import WeaviateStore

        return WeaviateStore(host="localhost", port=8080)

    def test_protocol_compliance(self, store: Any) -> None:
        assert isinstance(store, DocumentStoreProtocol)

    def test_upsert_and_search(self, store: Any) -> None:
        parent = Chunk(
            id="wp-1", text="weaviate parent", metadata={"source": "w.pdf", "page": 0},
            parent_id=None, token_count=50,
        )
        child = Chunk(
            id="wc-1", text="weaviate child", metadata={"source": "w.pdf", "page": 0},
            parent_id="wp-1", token_count=10,
        )
        embedding = [0.1] * 1536
        store.upsert_chunks([parent])
        store.upsert_chunks([child], embeddings=[embedding])

        results = store.semantic_search(embedding, top_k=1)
        assert any(r["id"] == "wc-1" for r in results)

    def test_bm25_corpus_persistence(self, store: Any) -> None:
        corpus = [{"id": "w1", "text": "weaviate test"}]
        store.store_bm25_corpus(corpus)
        loaded = store.load_bm25_corpus()
        assert loaded == corpus
