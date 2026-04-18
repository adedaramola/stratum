"""Unit tests for citation extraction in CitationGroundedGenerator."""

from __future__ import annotations

from rag.generation.generator import CitationGroundedGenerator
from rag.interfaces.retriever import RetrievedChunk


def _make_chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id=str(i), text=f"text {i}", source=f"doc{i}.pdf", page=i, score=1.0)
        for i in range(1, n + 1)
    ]


def test_single_citation() -> None:
    """[src 1] is parsed into a single CitationRef with index=1."""
    chunks = _make_chunks(3)
    answer = "The sky is blue [src 1]."
    refs = CitationGroundedGenerator._extract_citations(answer, chunks)
    assert len(refs) == 1
    assert refs[0].index == 1
    assert refs[0].source == "doc1.pdf"
    assert refs[0].page == 1


def test_multiple_citations() -> None:
    """[src 1] ... [src 3] produces two distinct CitationRefs."""
    chunks = _make_chunks(3)
    answer = "First claim [src 1]. Second claim [src 3]."
    refs = CitationGroundedGenerator._extract_citations(answer, chunks)
    assert len(refs) == 2
    indices = {r.index for r in refs}
    assert indices == {1, 3}


def test_duplicate_citations_deduplicated() -> None:
    """[src 1] appearing twice is deduplicated to a single CitationRef."""
    chunks = _make_chunks(3)
    answer = "Claim A [src 1]. Claim B [src 1]."
    refs = CitationGroundedGenerator._extract_citations(answer, chunks)
    assert len(refs) == 1
    assert refs[0].index == 1


def test_no_citations_raises() -> None:
    """An answer with zero [src N] markers raises CitationGroundingError."""
    chunks = _make_chunks(3)
    # CitationGroundedGenerator.generate() raises; test _extract_citations returns []
    answer = "The sky is blue with no citations."
    refs = CitationGroundedGenerator._extract_citations(answer, chunks)
    assert refs == []


def test_out_of_range_index_silently_dropped() -> None:
    """[src 99] with only 3 chunks is silently dropped — no CitationRef created."""
    chunks = _make_chunks(3)
    answer = "A claim [src 99]."
    refs = CitationGroundedGenerator._extract_citations(answer, chunks)
    assert refs == []


def test_web_content_no_page() -> None:
    """Web chunks with page=None produce CitationRef with page=None."""
    chunk = RetrievedChunk(
        id="1", text="web text", source="https://example.com", page=None, score=1.0
    )
    answer = "A web claim [src 1]."
    refs = CitationGroundedGenerator._extract_citations(answer, [chunk])
    assert len(refs) == 1
    assert refs[0].page is None
    assert refs[0].source == "https://example.com"
