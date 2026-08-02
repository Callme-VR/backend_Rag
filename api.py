"""
FastAPI Minimal Backend for RAG Production System

Run with:
    uvicorn api:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Import core RAG components
from src.data_loader import (
    load_documents,
    chunk_documents,
    ALLOWED_EXTENSION,
    MAX_FILE_SIZE,
    UPLOAD_DIRECTORY,
)
from src.embedding import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.search import SearchManager
from schemas import (
    SearchRequest,
    SearchResultItem,
    SearchResponse,
    UploadResponse,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Global Application State
# ---------------------------------------------------------------------------
rag_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy models and connections once at application startup."""
    print("=" * 60)
    print("Starting RAG FastAPI Service — Initializing Components...")
    print("=" * 60)

    # Load SentenceTransformer model into RAM
    embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # Connect to persistent ChromaDB vector store
    vector_store_mgr = VectorStoreManager(
        collection_name="rag_documents",
        persist_directory="./chroma_db"
    )

    # Initialize search manager
    search_mgr = SearchManager(
        embedding_manager=embedding_mgr,
        vector_store_manager=vector_store_mgr
    )

    # Store references in global state
    rag_state["embedding"] = embedding_mgr
    rag_state["vector_store"] = vector_store_mgr
    rag_state["search"] = search_mgr

    print("RAG Subsystems Loaded Successfully.")
    yield
    print("Shutting down RAG FastAPI Service.")
    rag_state.clear()


# ---------------------------------------------------------------------------
# FastAPI App Initialization & CORS Config
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Production Core API",
    description="Minimal FastAPI backend for PDF/document ingestion and vector retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """Return health status and count of documents in the vector index."""
    vector_store: VectorStoreManager = rag_state["vector_store"]
    embedding: EmbeddingManager = rag_state["embedding"]

    return HealthResponse(
        status="healthy",
        model=embedding.model_name,
        total_documents_in_store=vector_store.get_collection_count(),
    )


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD), chunk it, generate embeddings,
    and persist it to ChromaDB.
    """
    file_path = Path(file.filename)
    ext = file_path.suffix.lower()

    if ext not in ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {ALLOWED_EXTENSION}",
        )

    # Ensure Uploads directory exists
    UPLOAD_DIRECTORY.mkdir(exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = UPLOAD_DIRECTORY / unique_name

    # Save uploaded file
    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    # Verify file size cap
    file_size = dest_path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024):.1f} MB",
        )

    # Process document through RAG pipeline
    try:
        documents = load_documents(str(dest_path))
        if not documents:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text content from the uploaded document.",
            )

        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=250)
        if not chunks:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to produce text chunks from document.",
            )

        # Embedding & Vector Storage
        embedding_mgr: EmbeddingManager = rag_state["embedding"]
        vector_store_mgr: VectorStoreManager = rag_state["vector_store"]

        texts = [c["content"] for c in chunks]
        embeddings = embedding_mgr.generate_embedding(texts)
        metadatas = [c["metadata"] for c in chunks]

        # Annotate original filename in metadata
        for meta in metadatas:
            meta["original_filename"] = file.filename

        vector_store_mgr.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return UploadResponse(
            status="success",
            filename=file.filename,
            saved_path=str(dest_path),
            chunks_created=len(chunks),
            total_documents_in_store=vector_store_mgr.get_collection_count(),
        )

    except HTTPException:
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}",
        )


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_documents(payload: SearchRequest):
    """
    Search vector index using semantic similarity.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )

    search_mgr: SearchManager = rag_state["search"]

    try:
        results = search_mgr.search_with_scores(
            query=payload.query,
            n_results=payload.n_results
        )

        formatted_results = [
            SearchResultItem(
                rank=r["rank"],
                score=r["score"],
                content=r["content"],
                metadata=r["metadata"],
                id=r["id"],
            )
            for r in results
        ]

        return SearchResponse(
            query=payload.query,
            total_results=len(formatted_results),
            results=formatted_results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}",
        )