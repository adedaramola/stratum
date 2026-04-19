"""RAGPipeline and build_pipeline() factory.

build_pipeline() is the only place in the codebase where concrete classes are instantiated.
All other code works against Protocols — swapping implementations means changing Settings,
not editing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.config import Settings
from rag.interfaces.generator import CitedAnswer, GeneratorProtocol
from rag.interfaces.retriever import RetrieverProtocol


@dataclass
class RAGPipeline:
    """Assembled query pipeline. Holds a retriever and generator, both as Protocols."""

    retriever: RetrieverProtocol
    generator: GeneratorProtocol

    def query(self, question: str) -> CitedAnswer:
        """Run the full RAG pipeline: retrieve context then generate a cited answer."""
        chunks = self.retriever.retrieve(question)
        return self.generator.generate(question, chunks)

    def pipeline_fn(self, question: str) -> dict[str, Any]:
        """DeepEval-compatible interface: returns actual_output and retrieval_context keys."""
        result = self.query(question)
        return {
            "actual_output": result.answer,
            "retrieval_context": [c.text for c in result.raw_context],
        }


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Factory function. The only place concrete implementations are instantiated.

    All downstream code works against Protocols — swap implementations by
    changing Settings env vars, not by editing call sites.
    """
    from rag.generation.generator import CitationGroundedGenerator
    from rag.ingestion.embedder import get_embedder
    from rag.retrieval.hybrid import HybridRetriever
    from rag.store.factory import get_store

    store = get_store(settings)
    embedder = get_embedder(settings)
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        reranker_model=settings.reranker_model,
        top_k_dense=settings.top_k_dense,
        top_k_rerank=settings.top_k_rerank,
    )
    generator = CitationGroundedGenerator(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
    return RAGPipeline(retriever=retriever, generator=generator)
