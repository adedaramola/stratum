# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Hierarchical ingestion pipeline with parent/child chunking strategy (ADR-001)
- Dual store backend: Chroma (default, zero-infra) and Weaviate (production) (ADR-002)
- Separate parent/child collections — no zero-vector pollution (ADR-003)
- BM25 corpus persisted via store interface — no split state on restart (ADR-004)
- Dropped `unstructured` dependency in favour of pypdf + BeautifulSoup (ADR-005)
- Hybrid retrieval: BM25 + semantic search with RRF fusion (K=60)
- Cross-encoder re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Citation-grounded generation enforced at prompt level (ADR-006)
- RAGAS evaluation harness with `warn_only` mode and empirical baseline guidance
- Weekly scheduled RAGAS CI gate — not triggered on every PR
- Protocol-based interfaces with full dependency injection throughout
- `structlog` structured logging throughout
- `mypy` strict mode, `ruff` linting, pre-commit hooks
