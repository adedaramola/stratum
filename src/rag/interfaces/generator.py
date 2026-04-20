"""Protocol defining the generation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag.interfaces.retriever import RetrievedChunk


@dataclass
class CitationRef:
    """A single citation reference extracted from a generated answer."""

    index: int
    source: str
    page: int | None  # None for web-sourced content


@dataclass
class CitedAnswer:
    """The output of the generation step: answer text plus grounded citations."""

    answer: str
    citations: list[CitationRef] = field(default_factory=list)
    raw_context: list[RetrievedChunk] = field(default_factory=list)
    input_tokens: int = 0  # LLM prompt tokens consumed
    output_tokens: int = 0  # LLM completion tokens produced


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Contract for answer generation backends."""

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> CitedAnswer:
        """Generate a citation-grounded answer from retrieved context chunks."""
        ...
