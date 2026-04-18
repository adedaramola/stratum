"""FastAPI application. Exposes POST /query and GET /health over the RAGPipeline.

The pipeline — including cross-encoder model load and store connection — is
initialised once at startup via the lifespan context manager, not on first request.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from rag.config import get_settings
from rag.exceptions import RAGError
from rag.pipeline import RAGPipeline, build_pipeline

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming question."""

    question: str


class CitationOut(BaseModel):
    """One source cited in the answer."""

    index: int
    source: str
    page: int | None = None


class QueryResponse(BaseModel):
    """Answer with inline citations and retrieval metadata."""

    answer: str
    citations: list[CitationOut]
    context_chunks: int


# ---------------------------------------------------------------------------
# Application lifespan — builds the pipeline once at startup
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Build the RAGPipeline on startup; release on shutdown."""
    global _pipeline
    settings = get_settings()
    logger.info("stratum_api_startup", store_backend=settings.store_backend)
    try:
        _pipeline = build_pipeline(settings)
        logger.info("stratum_api_ready")
    except Exception as exc:
        logger.error("stratum_api_startup_failed", error=str(exc))
        raise
    yield
    _pipeline = None
    logger.info("stratum_api_shutdown")


app = FastAPI(
    title="Stratum RAG API",
    description="Citation-grounded document Q&A powered by hybrid retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict[str, str]:
    """Liveness probe — used by the ALB health check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    """Run a question through the RAG pipeline and return a cited answer."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised — startup may still be in progress.",
        )
    log = logger.bind(question=body.question[:80])
    try:
        result = _pipeline.query(body.question)
        log.info("query_ok", citations=len(result.citations))
        return QueryResponse(
            answer=result.answer,
            citations=[
                CitationOut(index=c.index, source=c.source, page=c.page)
                for c in result.citations
            ],
            context_chunks=len(result.raw_context),
        )
    except RAGError as exc:
        log.warning("query_rag_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.error("query_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc
