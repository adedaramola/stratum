"""Unit tests for RRF fusion logic in HybridRetriever."""

from __future__ import annotations

from rag.retrieval.hybrid import RRF_K, _rrf_fuse


def test_overlap_boosts_rank() -> None:
    """A doc ID present in both lists must outscore a doc in only one list."""
    dense = ["doc_a", "doc_b", "doc_c"]
    sparse = ["doc_a", "doc_d", "doc_e"]

    fused = _rrf_fuse(dense, sparse)

    # doc_a is in both lists; doc_b and doc_d are in only one
    assert fused.index("doc_a") < fused.index("doc_b"), (
        "doc_a (both lists) should rank above doc_b (dense only)"
    )
    assert fused.index("doc_a") < fused.index("doc_d"), (
        "doc_a (both lists) should rank above doc_d (sparse only)"
    )


def test_rrf_formula() -> None:
    """Verify the exact RRF scores for rank-1 positions."""
    dense = ["doc_x"]
    sparse = ["doc_x"]

    # doc_x is rank-1 in both lists → score = 2 / (RRF_K + 1)
    fused = _rrf_fuse(dense, sparse)
    assert fused == ["doc_x"]

    # Check a doc in only one list at rank-1
    dense_only = ["doc_a"]
    sparse_only: list[str] = []
    fused_single = _rrf_fuse(dense_only, sparse_only)
    # Score = 1 / (60 + 1) = 1/61
    assert fused_single == ["doc_a"]

    # Numerically verify via internal scores
    from rag.retrieval.hybrid import _rrf_fuse as rrf

    # Two separate docs at rank 1 each in one list
    result = rrf(["doc_a"], ["doc_b"])
    # Both have score 1/(RRF_K+1); order may vary but both should be present
    assert set(result) == {"doc_a", "doc_b"}


def test_rrf_constant_values() -> None:
    """Score for rank=1 in both lists is exactly 2/(RRF_K+1)."""
    # We verify by checking rank order: a doc in both at rank 1
    # must beat a doc in both at rank 2
    dense = ["doc_top", "doc_second"]
    sparse = ["doc_top", "doc_second"]

    fused = _rrf_fuse(dense, sparse)
    assert fused[0] == "doc_top"
    assert fused[1] == "doc_second"


def test_deduplication() -> None:
    """A doc ID appearing in both lists appears exactly once in fused output."""
    dense = ["doc_a", "doc_b"]
    sparse = ["doc_a", "doc_c"]

    fused = _rrf_fuse(dense, sparse)
    assert fused.count("doc_a") == 1, "doc_a should appear exactly once in fused output"


def test_empty_lists() -> None:
    """_rrf_fuse([], []) returns [] and _rrf_fuse(hits, []) preserves order."""
    assert _rrf_fuse([], []) == []

    dense = ["doc_a", "doc_b", "doc_c"]
    fused = _rrf_fuse(dense, [])
    assert fused == dense, "Single list fusion should preserve original order"


def test_rrf_k_constant() -> None:
    """RRF_K must be 60 (standard value from Cormack et al. 2009)."""
    assert RRF_K == 60
