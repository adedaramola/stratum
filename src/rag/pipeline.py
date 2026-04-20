"""RAGPipeline and build_pipeline() factory.

build_pipeline() is the only place in the codebase where concrete classes are instantiated.
All other code works against Protocols — swapping implementations means changing Settings,
not editing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from rag.config import Settings
from rag.interfaces.generator import CitedAnswer, GeneratorProtocol
from rag.interfaces.retriever import RetrieverProtocol

logger = structlog.get_logger(__name__)


def _default_tracer() -> Any:
    """Return a no-op tracer. Used as a safe default for RAGPipeline.tracer."""
    from rag.tracing import _NoOpTracer  # noqa: PLC0415

    return _NoOpTracer()


@dataclass
class RAGPipeline:
    """Assembled query pipeline. Holds a retriever and generator, both as Protocols."""

    retriever: RetrieverProtocol
    generator: GeneratorProtocol
    tracer: Any = field(default_factory=_default_tracer)

    def query(self, question: str) -> CitedAnswer:
        """Run the full RAG pipeline: retrieve context then generate a cited answer.

        Wraps both steps in Langfuse spans when tracing is enabled. Falls back
        transparently to no-op spans when Langfuse is not configured.
        """
        with self.tracer.trace("query", input=question) as span:
            with span.span("retrieval") as s:
                chunks = self.retriever.retrieve(question)
                s.update(output={"chunks": len(chunks)})
            with span.span("generation") as s:
                answer = self.generator.generate(question, chunks)
                s.update(output={"citations": len(answer.citations)})
        self.tracer.flush()
        return answer

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
    from rag.generation.generator import CitationGroundedGenerator  # noqa: PLC0415
    from rag.ingestion.embedder import get_embedder  # noqa: PLC0415
    from rag.retrieval.hybrid import HybridRetriever  # noqa: PLC0415
    from rag.store.factory import get_store  # noqa: PLC0415
    from rag.tracing import get_tracer  # noqa: PLC0415

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
    tracer = get_tracer(settings)
    if tracer.enabled:
        logger.info("tracing_enabled")
    return RAGPipeline(retriever=retriever, generator=generator, tracer=tracer)
