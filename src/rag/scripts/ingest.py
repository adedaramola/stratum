"""CLI entrypoint for document ingestion.

Zero business logic — orchestration only. All logic lives in the rag package.
Entry point registered in pyproject.toml as: stratum-ingest = "rag.scripts.ingest:main"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import structlog

from rag.config import get_settings
from rag.ingestion.chunker import HierarchicalChunker
from rag.ingestion.embedder import get_embedder
from rag.ingestion.loaders import Document, PDFLoader, WebLoader
from rag.store.factory import get_store

logger = structlog.get_logger(__name__)


def _resolve_sources(source: str) -> list[tuple[Any, str]]:
    """Return a list of (loader, path_or_url) pairs from the given source argument.

    Accepts:
      - A single .pdf file
      - A URL (http:// or https://)
      - A directory (globs *.pdf recursively)
    """
    path = Path(source)

    if source.startswith(("http://", "https://")):
        return [(WebLoader(), source)]

    if path.is_file() and path.suffix.lower() == ".pdf":
        return [(PDFLoader(), source)]

    if path.is_dir():
        pairs: list[tuple[Any, str]] = []
        for pdf in sorted(path.rglob("*.pdf")):
            pairs.append((PDFLoader(), str(pdf)))
        if not pairs:
            logger.warning("no_pdfs_found", directory=source)
        return pairs

    logger.error("unknown_source_type", source=source)
    sys.exit(1)


def main() -> None:
    """CLI entry point for stratum-ingest."""
    parser = argparse.ArgumentParser(
        description="Stratum document ingestion — loads, chunks, embeds, and indexes documents."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="PDF file path, URL (http/https), or directory containing PDFs",
    )
    parser.add_argument(
        "--env",
        default="dev",
        choices=["dev", "ci", "prod"],
        help="Environment label (logged for observability; does not alter Settings)",
    )
    args = parser.parse_args()

    settings = get_settings()
    log = logger.bind(source=args.source, env=args.env, store_backend=settings.store_backend)

    store = get_store(settings)
    embedder = get_embedder(settings)
    chunker = HierarchicalChunker(settings)

    sources = _resolve_sources(args.source)
    if not sources:
        log.error("no_sources_resolved")
        sys.exit(1)

    all_child_chunks: list[dict[str, Any]] = []
    parent_count = 0
    child_count = 0
    doc_count = 0

    for loader, path_or_url in sources:
        try:
            documents: list[Document] = loader.load(path_or_url)
        except Exception as exc:
            log.error("document_load_failed", path=path_or_url, error=str(exc))
            continue

        for doc in documents:
            chunks = list(chunker.chunk_document(doc.text, doc.metadata))
            parents = [c for c in chunks if c.is_parent()]
            children = [c for c in chunks if not c.is_parent()]

            if not children:
                continue

            child_embeddings = embedder.embed_batch([c.text for c in children])

            store.upsert_chunks(parents, embeddings=None)
            store.upsert_chunks(children, embeddings=child_embeddings)

            all_child_chunks.extend(
                {"id": c.id, "text": c.text, **c.metadata} for c in children
            )
            parent_count += len(parents)
            child_count += len(children)
            doc_count += 1

    store.store_bm25_corpus(all_child_chunks)

    log.info(
        "ingestion_complete",
        documents=doc_count,
        parents=parent_count,
        children=child_count,
        sources=len(sources),
    )
