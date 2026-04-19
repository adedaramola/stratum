"""Unit tests for RAGPipeline query and pipeline_fn interface."""

from __future__ import annotations

from rag.interfaces.retriever import RetrievedChunk
from rag.pipeline import RAGPipeline


def _make_chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id=str(i), text=f"chunk {i}", source="doc.pdf", page=i, score=0.9)
        for i in range(1, n + 1)
    ]


def test_pipeline_query_returns_cited_answer(mock_pipeline: RAGPipeline) -> None:
    """query() returns a CitedAnswer with a non-empty answer."""
    from tests.conftest import MockStore

    store = mock_pipeline.retriever._store  # type: ignore[attr-defined]
    assert isinstance(store, MockStore)

    # Seed the store with chunks so retrieval returns results
    from rag.interfaces.store import Chunk

    embedder = mock_pipeline.retriever._embedder  # type: ignore[attr-defined]
    parent = Chunk(
        id="p1",
        text="parent passage",
        metadata={"source": "doc.pdf", "page": 0},
        parent_id=None,
        token_count=50,
    )
    child = Chunk(
        id="c1",
        text="child chunk text about RAG",
        metadata={"source": "doc.pdf", "page": 0},
        parent_id="p1",
        token_count=20,
    )
    vec = embedder.embed(child.text)
    store.upsert_chunks([parent])
    store.upsert_chunks([child], embeddings=[vec])
    store.store_bm25_corpus([{"id": "c1", "text": child.text, "source": "doc.pdf", "page": 0}])
    mock_pipeline.retriever.build_index(store.load_bm25_corpus())

    result = mock_pipeline.query("What is RAG?")
    assert result.answer
    assert len(result.citations) >= 1
    assert len(result.raw_context) >= 1


def test_pipeline_fn_returns_deepeval_compatible_dict(mock_pipeline: RAGPipeline) -> None:
    """pipeline_fn() returns a dict with 'actual_output' and 'retrieval_context' keys."""
    from rag.interfaces.store import Chunk

    store = mock_pipeline.retriever._store  # type: ignore[attr-defined]
    embedder = mock_pipeline.retriever._embedder  # type: ignore[attr-defined]

    parent = Chunk(
        id="p2",
        text="parent text",
        metadata={"source": "s.pdf", "page": 1},
        parent_id=None,
        token_count=50,
    )
    child = Chunk(
        id="c2",
        text="child text about retrieval",
        metadata={"source": "s.pdf", "page": 1},
        parent_id="p2",
        token_count=20,
    )
    vec = embedder.embed(child.text)
    store.upsert_chunks([parent])
    store.upsert_chunks([child], embeddings=[vec])
    store.store_bm25_corpus([{"id": "c2", "text": child.text, "source": "s.pdf", "page": 1}])
    mock_pipeline.retriever.build_index(store.load_bm25_corpus())

    out = mock_pipeline.pipeline_fn("What is retrieval?")
    assert "actual_output" in out
    assert "retrieval_context" in out
    assert isinstance(out["retrieval_context"], list)
