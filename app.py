"""Stratum RAG — Streamlit chat interface.

Connects to the FastAPI backend (rag.api.main) running on localhost:8000.
Start the backend first:
    uvicorn rag.api.main:app --reload

Then run this app:
    streamlit run app.py
"""

from __future__ import annotations

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stratum RAG",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Stratum RAG")
st.caption("Citation-grounded document Q&A powered by hybrid retrieval.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_ok" not in st.session_state:
    st.session_state.api_ok = False


# ---------------------------------------------------------------------------
# Backend health check (once per session)
# ---------------------------------------------------------------------------


def check_backend() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


if not st.session_state.api_ok:
    st.session_state.api_ok = check_backend()

if not st.session_state.api_ok:
    st.error(
        "**Backend not reachable.** Start it with:\n"
        "```\nuvicorn rag.api.main:app --reload\n```",
        icon="🔌",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — document info & controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Stratum")
    st.markdown(
        "Ask questions about your ingested documents. "
        "Every answer includes citations so you can verify the source."
    )
    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- Hybrid retrieval (BM25 + dense)\n"
        "- Cross-encoder re-ranking\n"
        "- Citation-grounded generation\n"
        "- ChromaDB vector store"
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s)"):
                for c in msg["citations"]:
                    page_label = f" · p.{c['page']}" if c.get("page") is not None else ""
                    st.markdown(f"**[{c['index']}]** `{c['source']}`{page_label}")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if question := st.chat_input("Ask a question about your documents…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call the backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                citations = data.get("citations", [])
                context_chunks = data.get("context_chunks", 0)

                st.markdown(answer)

                if citations:
                    with st.expander(f"📎 {len(citations)} source(s) · {context_chunks} chunks retrieved"):
                        for c in citations:
                            page_label = f" · p.{c['page']}" if c.get("page") is not None else ""
                            st.markdown(f"**[{c['index']}]** `{c['source']}`{page_label}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )

            except httpx.HTTPStatusError as exc:
                error_msg = f"API error {exc.response.status_code}: {exc.response.text}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ {error_msg}", "citations": []}
                )
            except Exception as exc:
                error_msg = f"Could not reach backend: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ {error_msg}", "citations": []}
                )
