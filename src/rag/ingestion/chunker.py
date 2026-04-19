"""Hierarchical chunking strategy for the RAG ingestion pipeline.

WHY hierarchical over flat chunking:
  Flat chunks force a precision/context tradeoff — small chunks retrieve with high
  precision but lose surrounding context; large chunks preserve context but dilute
  relevance scores. The parent/child model resolves this: child chunks (~300 tokens)
  are the retrieval units, ensuring high precision. On retrieval, each child's parent_id
  is resolved and the parent passage (~1500 tokens) is passed to the LLM, giving it
  full context without sacrificing retrieval precision.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Generator
from typing import Any

import structlog

from rag.config import Settings
from rag.exceptions import ChunkingError
from rag.interfaces.store import Chunk

logger = structlog.get_logger(__name__)

# Sentence boundary pattern: period/exclamation/question followed by whitespace
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


class HierarchicalChunker:
    """Splits documents into parent/child chunk pairs.

    Child chunks (~300 tokens) are the retrieval units.
    Parent chunks (~1500 tokens) are passed to the LLM as context.
    Each child carries a parent_id used for context expansion at query time.
    """

    def __init__(self, settings: Settings) -> None:
        self._parent_token_size = settings.parent_token_size
        self._child_token_size = settings.child_token_size
        self._overlap_sentences = settings.overlap_sentences

    def chunk_document(self, text: str, metadata: dict[str, Any]) -> Generator[Chunk, None, None]:
        """Yield chunks depth-first: parent → its children → next parent → its children.

        Yields nothing for empty input without raising.
        Raises ChunkingError on unexpected failures.
        """
        if not text.strip():
            return

        doc_id = metadata.get("source", "unknown")
        try:
            parent_chunks = self._build_parent_chunks(text, metadata)
            for parent in parent_chunks:
                yield parent
                yield from self._build_child_chunks(parent, metadata)
        except ChunkingError:
            raise
        except Exception as exc:
            raise ChunkingError(
                document_id=str(doc_id),
                message=f"Unexpected error chunking document '{doc_id}': {exc}",
            ) from exc

    def _build_parent_chunks(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split text into parent-sized chunks on sentence boundaries."""
        sentences = _SENTENCE_BOUNDARY.split(text)
        parents: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = _estimate_tokens(sentence)

            if buffer_tokens + sentence_tokens > self._parent_token_size and buffer:
                parents.append(self._make_chunk(buffer, metadata, parent_id=None))
                buffer = []
                buffer_tokens = 0

            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            parents.append(self._make_chunk(buffer, metadata, parent_id=None))

        return parents

    def _build_child_chunks(
        self, parent: Chunk, metadata: dict[str, Any]
    ) -> Generator[Chunk, None, None]:
        """Split a parent chunk into child-sized chunks with sentence overlap."""
        sentences = _SENTENCE_BOUNDARY.split(parent.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        buffer: list[str] = []
        buffer_tokens = 0

        for _i, sentence in enumerate(sentences):
            sentence_tokens = _estimate_tokens(sentence)

            if buffer_tokens + sentence_tokens > self._child_token_size and buffer:
                chunk = self._make_chunk(buffer, metadata, parent_id=parent.id)
                log = logger.bind(
                    chunk_id=chunk.id,
                    parent_id=parent.id,
                    token_count=chunk.token_count,
                )
                log.debug("child_chunk_created")
                yield chunk

                # carry over the last `overlap_sentences` sentences into next child
                overlap_start = max(0, len(buffer) - self._overlap_sentences)
                buffer = buffer[overlap_start:]
                buffer_tokens = sum(_estimate_tokens(s) for s in buffer)

            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            chunk = self._make_chunk(buffer, metadata, parent_id=parent.id)
            logger.bind(
                chunk_id=chunk.id,
                parent_id=parent.id,
                token_count=chunk.token_count,
            ).debug("child_chunk_created")
            yield chunk

    @staticmethod
    def _make_chunk(
        sentences: list[str],
        metadata: dict[str, Any],
        parent_id: str | None,
    ) -> Chunk:
        text = " ".join(sentences)
        return Chunk(
            id=str(uuid.uuid4()),
            text=text,
            metadata=dict(metadata),
            parent_id=parent_id,
            token_count=_estimate_tokens(text),
        )
