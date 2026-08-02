"""
Streamlit Web Application for RAG Production System
Connects to the FastAPI backend (api.py) via HTTP.

Run backend first:
    uvicorn api:app --reload --port 8000

Then run the UI:
    streamlit run app.py
"""

import os
from typing import Optional

import requests
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Backend Connection
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

# Allowed extensions as enforced by the backend (src/data_loader.py)
ALLOWED_EXTENSIONS = [
    "pdf", "doc", "docx", "txt", "rtf", "json", "pptx", "csv", "md"
]


def api_get_health() -> dict:
    """Call GET /health on the backend."""
    resp = requests.get(f"{API_BASE_URL}/health", timeout=15)
    resp.raise_for_status()
    return resp.json()


def api_upload(file_name: str, file_bytes) -> dict:
    """Upload a file to POST /upload on the backend."""
    resp = requests.post(
        f"{API_BASE_URL}/upload",
        files={"file": (file_name, file_bytes)},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def api_search(query: str, n_results: int) -> dict:
    """Run semantic search via POST /search on the backend."""
    resp = requests.post(
        f"{API_BASE_URL}/search",
        json={"query": query, "n_results": n_results},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Production Workspace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .score-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend Availability & Health
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def get_health_cached() -> Optional[dict]:
    """Fetch backend health, cached briefly to avoid hammering the API."""
    try:
        return api_get_health()
    except requests.RequestException:
        return None


backend_online = True
try:
    health = get_health_cached()
except Exception:
    health = None

if health is None:
    backend_online = False
    st.error(
        f"⚠️ Could not reach the RAG backend at `{API_BASE_URL}`. "
        "Make sure it is running (`uvicorn api:app --reload --port 8000`)."
    )


# ---------------------------------------------------------------------------
# Sidebar UI & Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=64)
    st.title("RAG Controls")
    st.markdown("---")

    # Index Stats (from backend)
    if health:
        st.metric(label="Total Chunks Indexed", value=health.get("total_documents_in_store", 0))
    else:
        st.metric(label="Total Chunks Indexed", value="—")

    st.markdown("### ⚙️ Retrieval Parameters")
    n_results = st.slider("Top Search Results (k)", min_value=1, max_value=20, value=5)

    st.markdown("---")
    st.markdown("### ℹ️ System Status")
    if health:
        st.text(f"API: {API_BASE_URL}")
        st.text(f"Status: {health.get('status', 'unknown')}")
        st.text(f"Model: {health.get('model', 'unknown')}")
    else:
        st.text(f"API: {API_BASE_URL}")
        st.text("Status: offline")
        st.text("Model: —")
    st.text(f"Allowed Ext: {', '.join(ALLOWED_EXTENSIONS)}")

    if st.button("🔄 Refresh System Stats", use_container_width=True):
        get_health_cached.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main Layout & Tabs
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🔍 RAG Production System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload documents via the FastAPI backend, build vector embeddings, '
    'and perform semantic similarity searches.</div>',
    unsafe_allow_html=True
)

tab_search, tab_upload, tab_inspector = st.tabs([
    "🔎 Semantic Search",
    "📤 Document Ingestion",
    "📊 Vector Store Inspector"
])

# ---------------------------------------------------------------------------
# Tab 1: Semantic Search
# ---------------------------------------------------------------------------
with tab_search:
    st.subheader("Query Your Document Index")

    query_text = st.text_input(
        "Enter your question or query topic:",
        placeholder="e.g. What are the key takeaways from the document?",
        key="search_query"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_submitted = st.button("🚀 Search", use_container_width=True, type="primary")

    if search_submitted and not backend_online:
        st.warning("Backend is offline. Start the FastAPI server and try again.")

    if search_submitted and backend_online:
        if not query_text.strip():
            st.warning("Please enter a valid search query.")
        else:
            with st.spinner("Querying backend vector database..."):
                try:
                    response = api_search(query_text.strip(), n_results=n_results)
                    results = response.get("results", [])

                    if not results:
                        st.warning("No relevant matching chunks found in vector store.")
                    else:
                        st.success(f"Found {len(results)} matching document chunk(s):")
                        for item in results:
                            rank = item.get("rank", 1)
                            score = item.get("score", 0.0)
                            content = item.get("content", "")
                            meta = item.get("metadata", {}) or {}
                            original_filename = (
                                meta.get("original_filename")
                                or meta.get("filename")
                                or meta.get("source_file")
                                or meta.get("source")
                                or "Unknown Document"
                            )
                            page_num = meta.get("page", "N/A")

                            st.markdown(f"""
                                <div class="result-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <span style="font-weight: 700; font-size: 1.05rem; color: #1E293B;">
                                            #{rank} | 📄 {original_filename} (Page {page_num})
                                        </span>
                                        <span class="score-badge">Similarity: {score:.4f}</span>
                                    </div>
                                    <div style="font-size: 0.95rem; color: #334155; line-height: 1.5; white-space: pre-wrap;">
                                        {content}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            with st.expander(f"View Chunk #{rank} Metadata"):
                                st.json(meta)
                except requests.RequestException as e:
                    st.error(f"Search failed: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Document Ingestion
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload & Index New Documents")
    st.write("Upload `.pdf`, `.txt`, `.md` (and other supported) files. The backend will parse, "
             "chunk, embed, and store them in ChromaDB.")

    uploaded_files = st.file_uploader(
        "Choose document(s)",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True
    )

    if uploaded_files and not backend_online:
        st.warning("Backend is offline. Start the FastAPI server to ingest documents.")

    if uploaded_files and backend_online:
        if st.button("⚡ Process & Index Documents", type="primary"):
            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")
                if ext not in ALLOWED_EXTENSIONS:
                    st.error(f"Skipping {uploaded_file.name}: Unsupported file extension .{ext}")
                    continue

                with st.spinner(f"Uploading & processing '{uploaded_file.name}'..."):
                    try:
                        response = api_upload(uploaded_file.name, uploaded_file.getbuffer())
                        st.success(
                            f"Successfully processed **{response.get('filename', uploaded_file.name)}**: "
                            f"Created & indexed **{response.get('chunks_created', 0)}** chunk(s)! "
                            f"Store now has {response.get('total_documents_in_store', '?')} chunk(s)."
                        )
                    except requests.HTTPError as e:
                        detail = ""
                        try:
                            detail = e.response.json().get("detail", "")
                        except Exception:
                            pass
                        st.error(f"Error processing '{uploaded_file.name}': {detail or e}")
                    except requests.RequestException as e:
                        st.error(f"Error processing '{uploaded_file.name}': {e}")

            st.balloons()
            get_health_cached.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Vector Store Inspector
# ---------------------------------------------------------------------------
with tab_inspector:
    st.subheader("Collection Overview")

    if backend_online:
        current_count = health.get("total_documents_in_store", 0)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Indexed Chunks", current_count)
        col_b.metric("Embedding Model", health.get("model", "—"))
        col_c.metric("API Status", health.get("status", "—"))

        if current_count > 0:
            st.info(
                "Chunk-level inspection is available through the backend's API "
                f"(Swagger UI at `{API_BASE_URL}/docs`). This UI only exposes aggregate stats."
            )
        else:
            st.info("No data currently indexed in ChromaDB. Upload documents in the "
                    "'Document Ingestion' tab first.")
    else:
        st.warning("Cannot inspect the vector store while the backend is offline.")
