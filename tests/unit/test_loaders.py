"""Unit tests for PDF and web document loaders."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.exceptions import DocumentLoadError
from rag.ingestion.loaders import PDFLoader, WebLoader

# ---------------------------------------------------------------------------
# WebLoader
# ---------------------------------------------------------------------------


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from httpx import HTTPStatusError, Request, Response

        resp.raise_for_status.side_effect = HTTPStatusError(
            "error", request=MagicMock(spec=Request), response=MagicMock(spec=Response)
        )
    return resp


_SIMPLE_HTML = """
<html>
<head><title>Test</title></head>
<body>
  <nav>Nav content to strip</nav>
  <p>Main content here.</p>
  <footer>Footer to strip</footer>
</body>
</html>
"""


def test_web_loader_returns_document() -> None:
    with patch("rag.ingestion.loaders.httpx.get", return_value=_mock_response(_SIMPLE_HTML)):
        docs = WebLoader().load("https://example.com")
    assert len(docs) == 1
    assert "Main content here." in docs[0].text
    assert docs[0].source == "https://example.com"


def test_web_loader_strips_nav_footer() -> None:
    with patch("rag.ingestion.loaders.httpx.get", return_value=_mock_response(_SIMPLE_HTML)):
        docs = WebLoader().load("https://example.com")
    assert "Nav content to strip" not in docs[0].text
    assert "Footer to strip" not in docs[0].text


def test_web_loader_includes_fetched_at_metadata() -> None:
    with patch("rag.ingestion.loaders.httpx.get", return_value=_mock_response(_SIMPLE_HTML)):
        docs = WebLoader().load("https://example.com")
    assert "fetched_at" in docs[0].metadata


def test_web_loader_network_error_raises_document_load_error() -> None:
    with (
        patch("rag.ingestion.loaders.httpx.get", side_effect=ConnectionError("timeout")),
        pytest.raises(DocumentLoadError) as exc_info,
    ):
        WebLoader().load("https://example.com")
    assert exc_info.value.source == "https://example.com"


def test_web_loader_http_error_raises_document_load_error() -> None:
    with (
        patch("rag.ingestion.loaders.httpx.get", return_value=_mock_response("", status_code=404)),
        pytest.raises(DocumentLoadError),
    ):
        WebLoader().load("https://example.com/missing")


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------


def _mock_pdf_reader(pages: list[str]) -> MagicMock:
    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock(extract_text=MagicMock(return_value=p)) for p in pages]
    return mock_reader


def test_pdf_loader_returns_one_doc_per_page() -> None:
    # pypdf is imported inside load() — patch the source module
    with patch("pypdf.PdfReader", return_value=_mock_pdf_reader(["page 1", "page 2"])):
        docs = PDFLoader().load(Path("test.pdf"))
    assert len(docs) == 2
    assert docs[0].text == "page 1"
    assert docs[1].text == "page 2"


def test_pdf_loader_skips_empty_pages() -> None:
    with patch("pypdf.PdfReader", return_value=_mock_pdf_reader(["content", "", "more"])):
        docs = PDFLoader().load(Path("test.pdf"))
    assert len(docs) == 2


def test_pdf_loader_metadata_contains_source_and_page() -> None:
    with patch("pypdf.PdfReader", return_value=_mock_pdf_reader(["text"])):
        docs = PDFLoader().load(Path("my_doc.pdf"))
    assert docs[0].metadata["source"] == "my_doc.pdf"
    assert docs[0].metadata["page"] == 0


def test_pdf_loader_error_raises_document_load_error() -> None:
    with (
        patch("pypdf.PdfReader", side_effect=OSError("file not found")),
        pytest.raises(DocumentLoadError),
    ):
        PDFLoader().load(Path("missing.pdf"))
