# Stratum

A production-grade domain-specific RAG engine — ask questions against your documents and receive precise, citation-grounded answers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAGPipeline                              │
│                                                                  │
│  Query ──► HybridRetriever ──────────────────────────────────►  │
│               │                                                  │
│               ├── Dense:  EmbedderProtocol → DocumentStore ANN  │
│               ├── Sparse: BM25Okapi (in-store corpus)           │
│               ├── Fuse:   RRF (K=60, Cormack et al. 2009)       │
│               ├── Expand: child → parent context resolution     │
│               └── Rerank: CrossEncoder ms-marco-MiniLM          │
│                                                                  │
│  Chunks ──► CitationGroundedGenerator ──────────────────────►   │
│               │                                                  │
│               ├── Context block: [src N] (source p.PAGE)        │
│               ├── Claude API call with citation-enforcing prompt │
│               └── Parse + validate [src N] markers              │
│                                                                  │
│  ◄──────────────────────────── CitedAnswer (answer + citations) │
└─────────────────────────────────────────────────────────────────┘

Document ingestion:
  Source (PDF / URL / dir) → Loader → HierarchicalChunker
  → BGEEmbedder / OpenAIEmbedder → DocumentStore (Chroma | Weaviate)
```

---

## Stack

| Layer | Default | Production | Rationale |
|---|---|---|---|
| Embedding | OpenAI text-embedding-3-small | BAAI/bge-large-en-v1.5 | Zero-infra default; BGE opt-in via `[local-embed]` |
| Vector store | Chroma (in-process) | Weaviate 1.25.3 | Zero Docker for dev; HNSW tuning + gRPC for prod |
| Retrieval fusion | BM25 + ANN → RRF (K=60) | Same | Parameter-free, robust to miscalibrated retrievers |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Same | Strong precision at low latency (~50ms CPU) |
| LLM | claude-sonnet-4-20250514 | Same | Citation-enforcing prompt, structured output |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-handle/stratum.git && cd stratum

# 2. Install (default: Chroma + OpenAI embeddings — zero Docker)
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env   # add STRATUM_ANTHROPIC_API_KEY and STRATUM_OPENAI_API_KEY

# 4. Ingest documents
stratum-ingest --source /path/to/your/docs/

# 5. Query
python -c "
from rag import build_pipeline, get_settings
pipeline = build_pipeline(get_settings())
result = pipeline.query('What is the data retention policy?')
print(result.answer)
for c in result.citations:
    print(f'  [{c.index}] {c.source} p.{c.page}')
"
```

### Switching to Weaviate

```bash
# .env
STRATUM_STORE_BACKEND=weaviate

make docker-up   # starts Weaviate on localhost:8080
stratum-ingest --source /path/to/docs/
```

---

## Development Workflow

| Command | Description |
|---|---|
| `make install` | Install all dev dependencies + pre-commit hooks |
| `make lint` | ruff check + format check |
| `make format` | ruff format + autofix |
| `make typecheck` | mypy strict mode |
| `make test-unit` | Unit tests (no network, no Docker) |
| `make test-integration` | Integration tests (Chroma: no Docker; Weaviate: skipped without Docker) |
| `make eval` | RAGAS evaluation gate |
| `make docker-up` | Start Weaviate service container |
| `make ci` | Full CI: lint + typecheck + unit tests |
| `make clean` | Remove build artefacts and caches |

---

## Project Layout

```
stratum/
├── src/rag/                  # All package code (PEP 517 src layout)
│   ├── interfaces/           # typing.Protocol definitions — the public contract
│   ├── ingestion/            # Loaders, chunker, embedders
│   ├── store/                # Chroma, Weaviate, factory
│   ├── retrieval/            # HybridRetriever (BM25 + dense + RRF + rerank)
│   ├── generation/           # CitationGroundedGenerator
│   ├── evaluation/           # RAGASEvaluator
│   ├── scripts/              # CLI entry points
│   ├── config.py             # Pydantic Settings — one object rules all config
│   ├── exceptions.py         # Domain exception hierarchy
│   └── pipeline.py           # RAGPipeline + build_pipeline() factory
├── tests/
│   ├── unit/                 # No network, no Docker — fast
│   ├── integration/          # Chroma: no Docker; Weaviate: skipped without Docker
│   └── e2e/                  # RAGAS gate (weekly CI schedule)
├── docs/                     # ADRs and evaluation guide
├── config/                   # eval_thresholds.yaml
└── data/golden/              # Golden QA pairs for RAGAS evaluation
```

---

## Implementation Plan

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Core pipeline: ingestion, hybrid retrieval, citation generation, unit tests | ✅ Complete |
| **Phase 2** | Golden dataset curation, RAGAS baseline establishment, eval gate activation | 🔄 In progress |
| **Phase 3** | Learned fusion weights, streaming responses, multi-tenancy, prompt caching | 📋 Planned |

---

## Evaluation

RAGAS evaluation runs **weekly** (Monday 06:00 UTC) and on manual trigger — not on every PR.
This avoids LLM judge costs (~$0.02–0.10 per run) and prevents flakiness from judge variance (±0.03–0.05).

```bash
# Run locally
make eval
```

See [docs/evaluation.md](docs/evaluation.md) for:
- What each metric measures and what a low score tells you to fix
- How to curate a golden QA dataset
- How to establish empirical baselines before enabling the hard gate
- How to interpret and tune thresholds

---

## ADR Index

| # | Decision | Status |
|---|---|---|
| [ADR-001](docs/architecture.md#adr-001-hierarchical-chunking-over-flat-chunking) | Hierarchical chunking (parent/child) | Accepted |
| [ADR-002](docs/architecture.md#adr-002-dual-store-backend-chroma-default-weaviate-production) | Dual store backend (Chroma + Weaviate) | Accepted |
| [ADR-003](docs/architecture.md#adr-003-separate-parent-and-child-collections) | Separate parent/child collections | Accepted |
| [ADR-004](docs/architecture.md#adr-004-bm25-corpus-co-located-with-vector-store) | BM25 corpus co-located with vector store | Accepted |
| [ADR-005](docs/architecture.md#adr-005-drop-unstructured-as-a-dependency) | Drop `unstructured` dependency | Accepted |
| [ADR-006](docs/architecture.md#adr-006-citation-grounding-enforced-at-generation-time) | Citation grounding at generation time | Accepted |

---

## License

MIT
