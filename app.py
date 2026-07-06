"""
Document Question Answering System - Streamlit UI
----------------------------------------------------
100% local: local embedding model + local FAISS index + local
Flan-T5 generation model. There is no API key field anywhere in
this app because no external API is ever called.
"""

import sys
import time
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import RAGPipeline

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = None


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Loaded once per session; models stay resident in memory."""
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Sidebar - document upload & indexing (no API settings of any kind)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Runs fully offline on local models. No API keys required.")

    st.subheader("1. Upload your documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size (words)", 100, 400, 200, step=20)
    overlap = st.slider("Chunk overlap (words)", 0, 100, 40, step=10)
    top_k = st.slider("Chunks retrieved per question", 1, 8, 4)

    build_clicked = st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True)

    if build_clicked:
        if not uploaded_files:
            st.error("Please upload at least one document first.")
        else:
            with st.spinner("Loading local models and indexing documents..."):
                pipeline = load_pipeline()
                pipeline.chunk_size = chunk_size
                pipeline.overlap = overlap
                pipeline.top_k = top_k

                tmp_paths = []
                tmp_dir = tempfile.mkdtemp()
                for f in uploaded_files:
                    p = Path(tmp_dir) / f.name
                    p.write_bytes(f.getbuffer())
                    tmp_paths.append(str(p))

                metrics = pipeline.ingest(tmp_paths)

                st.session_state.pipeline = pipeline
                st.session_state.ingested_files = [f.name for f in uploaded_files]
                st.session_state.metrics = metrics
                st.session_state.chat_history = []
            st.success(f"Indexed {metrics['num_chunks']} chunks from {metrics['num_documents']} document(s).")

    if st.session_state.ingested_files:
        st.subheader("Indexed documents")
        for name in st.session_state.ingested_files:
            st.markdown(f"- {name}")

    if st.session_state.metrics:
        st.subheader("System info")
        m = st.session_state.metrics
        st.markdown(
            f"""
            - **Embedding model:** `{m['embedding_model']}` ({m['embedding_mode']})
            - **Embedding dim:** {m['embedding_dimension']}
            - **Generator model:** `{m['generator_model']}` ({m['generator_mode']})
            - **Vector store:** {m['vector_store']}
            - **Chunks indexed:** {m['num_chunks']}
            - **Indexing time:** {m['total_setup_time_sec']}s
            """
        )

# ---------------------------------------------------------------------------
# Main panel - chat interface
# ---------------------------------------------------------------------------
st.header("Ask questions about your documents")

if st.session_state.pipeline is None:
    st.info("👈 Upload documents and click **Build Knowledge Base** to get started.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander(f"📚 Sources ({len(turn['sources'])} chunks) · {turn['total_time_sec']}s"):
                for rank, src in enumerate(turn["sources"], 1):
                    st.markdown(
                        f"**[{rank}] {src['source']}** — chunk {src['chunk_index']} "
                        f"(relevance: {src['final_score']:.2f})"
                    )
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))

    question = st.chat_input("Ask a question about your uploaded documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                result = st.session_state.pipeline.ask(question)
            st.write(result["answer"])
            with st.expander(
                f"📚 Sources ({len(result['retrieved_chunks'])} chunks) · {result['total_time_sec']}s"
            ):
                for rank, src in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(
                        f"**[{rank}] {src['source']}** — chunk {src['chunk_index']} "
                        f"(relevance: {src['final_score']:.2f})"
                    )
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result["retrieved_chunks"],
                "total_time_sec": result["total_time_sec"],
            }
        )
