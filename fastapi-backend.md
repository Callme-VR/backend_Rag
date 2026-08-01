# Simplified FastAPI Backend Architecture

## 1. Overview

This document specifies the minimal, streamlined FastAPI backend for the RAG Production system. It replaces complex microservice architectures, task queues, and layered ORM abstractions with a clean, direct API layer that connects directly to the core library (`src/data_loader.py`, `src/embedding.py`, `src/vector_store.py`, `src/search.py`).

### Key Design Principles
- **Zero Over-Engineering**: Direct delegation from FastAPI endpoint handlers to the core `src/` subsystems.
- **Single-Load Model Lifecycle**: The SentenceTransformer model (~90MB) and ChromaDB client are loaded **once** at application startup via FastAPI's `lifespan` manager and reused across requests.
- **Focused Core Endpoints**: Only three essential endpoints — Health check, Document Upload & Processing, and Semantic Search.

---

## 2. API Endpoints Specification

### 2.1 Health Check & Stats
- **Endpoint**: `GET /health`
- **Description**: Returns server status, loaded embedding model name, and the current document count in ChromaDB.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "model": "all-MiniLM-L6-v2",
  "total_documents_in_store": 42
}
```

---

### 2.2 Upload & Process Document
- **Endpoint**: `POST /upload`
- **Content-Type**: `multipart/form-data`
- **Description**: Receives a document file (PDF, TXT, MD), saves it into `Uploads/`, extracts text, splits text into chunks, generates vector embeddings, and stores them in ChromaDB.
- **Form Parameters**:
  - `file`: `UploadFile` (required)
- **Response `200 OK`**:
```json
{
  "status": "success",
  "filename": "sample_report.pdf",
  "saved_path": "Uploads/a1b2c3d4.pdf",
  "chunks_created": 15,
  "total_documents_in_store": 57
}
```
- **Error Responses**:
  - `400 Bad Request`: File type not supported or file exceeds size cap (200MB).
  - `500 Internal Server Error`: Processing or embedding generation failure.

---

### 2.3 Semantic Search & Retrieval
- **Endpoint**: `POST /search`
- **Content-Type**: `application/json`
- **Description**: Converts the query text into a vector embedding, queries ChromaDB for the top-k nearest matching chunks, and returns the formatted results with similarity scores.
- **Request Body**:
```json
{
  "query": "What is the summary of project budget?",
  "n_results": 3
}
```
- **Response `200 OK`**:
```json
{
  "query": "What is the summary of project budget?",
  "total_results": 3,
  "results": [
    {
      "rank": 1,
      "score": 0.8942,
      "content": "The overall project budget for Q3 is estimated at $150,000...",
      "metadata": {
        "source_file": "sample_report.pdf",
        "file_type": "pdf",
        "page": 2,
        "chunk_index": 4
      },
      "id": "e7f8g9h0..."
    }
  ]
}
```

---

## 3. Complete Implementation Code

Below is the complete, self-contained `api.py` implementation designed to run inside the `backend/` directory.

```python
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
from pydantic import BaseModel, Field

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
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., example="What is this document about?")
    n_results: int = Field(default=5, ge=1, le=20, example=5)


class SearchResultItem(BaseModel):
    rank: int
    score: float
    content: str
    metadata: Dict[str, Any]
    id: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]


class UploadResponse(BaseModel):
    status: str
    filename: str
    saved_path: str
    chunks_created: int
    total_documents_in_store: int


class HealthResponse(BaseModel):
    status: str
    model: str
    total_documents_in_store: int


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

    # Process document through RAG pipeline
    try:
        documents = load_documents(str(dest_path))
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text content from the uploaded document.",
            )

        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=250)
        if not chunks:
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
            metadatas=metadatas
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
```

---

## 4. Execution & Quickstart

### Prerequisites
Make sure dependencies are installed in your virtual environment:
```bash
pip install fastapi uvicorn python-multipart
```

### Running the API Server
Start the FastAPI server from the `backend/` directory:
```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Interactive API Documentation
Open your browser and navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Testing Endpoints via cURL

**1. Upload a Document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/document.pdf"
```

**2. Query the System:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are key findings?", "n_results": 3}'
```
