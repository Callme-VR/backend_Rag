# FastAPI Backend Architecture — RAG Production

This document outlines the architecture design and API specification for building a **FastAPI** service on top of the existing RAG core library (`src/data_loader.py`, `src/embedding.py`, `src/vector_store.py`, `src/search.py`).

---

## 1. Overview & Architecture

The FastAPI application acts as an asynchronous API layer exposing the RAG pipeline to frontends (e.g., Next.js frontend in `frentend/`) or third-party consumers.

```
┌─────────────────┐       HTTP / REST       ┌───────────────────────────────┐
│ Next.js Client  │ ──────────────────────> │         FastAPI App           │
│  (frentend/)    │ <────────────────────── │      (app/main.py & API)      │
└─────────────────┘                         └──────────────┬────────────────┘
                                                           │
                                            ┌──────────────┴────────────────┐
                                            │      RAG Core Subsystems      │
                                            ├───────────────────────────────┤
                                            │ • data_loader.py (Upload/Chunk)│
                                            │ • embedding.py (MiniLM-L6-v2) │
                                            │ • vector_store.py (ChromaDB)  │
                                            │ • search.py (SearchManager)   │
                                            └───────────────────────────────┘
```

### Key Architectural Objectives
1. **Asynchronous Non-blocking Uploads**: Handle multi-part file uploads efficiently without blocking the main event loop.
2. **Background Processing**: Support both immediate indexing for small documents and background task execution (`BackgroundTasks` or Celery) for large PDFs/documents.
3. **Modularity & Reusability**: Leverage existing `src/` modules directly through service wrappers or dependency injection.
4. **Structured Request/Response Schemas**: Strict Pydantic models for validation, serialization, and automatic OpenAPI doc generation (`/docs`).

---

## 2. Proposed Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point, CORS, lifespan setup
│   ├── api/                     # API routers
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # Top-level v1 router
│   │   │   ├── documents.py     # Upload, list, delete document endpoints
│   │   │   ├── search.py        # Semantic query & search endpoints
│   │   │   └── health.py        # System health & stats endpoints
│   ├── core/                    # App configuration, security, dependencies
│   │   ├── config.py            # Pydantic Settings (.env configuration)
│   │   └── dependencies.py      # FastAPI Dependency Injection (Managers)
│   ├── schemas/                 # Pydantic models for Requests/Responses
│   │   ├── document.py
│   │   ├── search.py
│   │   └── common.py
│   └── services/                # Business logic interfacing with src/
│       ├── document_service.py
│       └── search_service.py
├── src/                         # Existing core RAG library
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── search.py
└── fastapi-backend.md           # Architecture & Endpoint Documentation
```

---

## 3. OpenAPI Endpoint Specification

### 3.1 Document Ingestion & Management (`/api/v1/documents`)

#### `POST /api/v1/documents/upload`
Upload a single document (PDF, TXT, MD, DOCX), save to `Uploads/`, parse, chunk, embed, and store in vector database.

- **Request Format**: `multipart/form-data`
- **Form Parameters**:
  - `file`: `UploadFile` (Required) — Document file object.
  - `chunk_size`: `int` (Optional, default: 1000) — Maximum characters per chunk.
  - `chunk_overlap`: `int` (Optional, default: 250) — Character overlap between chunks.
  - `process_async`: `bool` (Optional, default: `false`) — Whether to process chunking and vector indexing in background.
- **Response Schemas**:
  - `200 OK` (Sync):
    ```json
    {
      "status": "success",
      "message": "File processed and indexed successfully",
      "file_info": {
        "original_filename": "sample.pdf",
        "saved_name": "a1b2c3d4.pdf",
        "saved_path": "Uploads/a1b2c3d4.pdf",
        "size_bytes": 1048576,
        "extension": ".pdf"
      },
      "processing_summary": {
        "total_pages_or_docs": 5,
        "total_chunks": 18,
        "indexed_vector_count": 18
      }
    }
    ```
  - `202 Accepted` (Async):
    ```json
    {
      "status": "processing",
      "message": "File upload accepted for background indexing",
      "task_id": "task_9f8e7d6c",
      "file_info": {
        "original_filename": "large_report.pdf",
        "saved_name": "e5f6g7h8.pdf"
      }
    }
    ```
- **Error Responses**:
  - `400 Bad Request`: Unsupported file type or file size exceeds maximum (200MB).
  - `422 Unprocessable Entity`: Validation failure on parameters.
  - `500 Internal Server Error`: Exception during parsing or embedding.

---

#### `POST /api/v1/documents/upload-batch`
Upload multiple documents in a single request.

- **Request Format**: `multipart/form-data`
- **Form Parameters**:
  - `files`: `List[UploadFile]` (Required) — List of files.
  - `chunk_size`: `int` (Optional, default: 1000)
  - `chunk_overlap`: `int` (Optional, default: 250)
- **Response**: `200 OK` with list of processed document summaries.

---

#### `GET /api/v1/documents/`
List all uploaded documents stored in the system and current vector store index counts.

- **Query Parameters**:
  - `skip`: `int` (default: 0)
  - `limit`: `int` (default: 50)
- **Response (`200 OK`)**:
  ```json
  {
    "total_files": 3,
    "total_vectors": 54,
    "documents": [
      {
        "filename": "sample.pdf",
        "saved_name": "a1b2c3d4.pdf",
        "extension": ".pdf",
        "size_bytes": 1048576,
        "uploaded_at": "2026-08-01T12:00:00Z"
      }
    ]
  }
  ```

---

#### `DELETE /api/v1/documents/{saved_name}`
Remove an uploaded document from disk and clear associated vectors from ChromaDB vector store.

- **Path Parameter**: `saved_name` (e.g. `a1b2c3d4.pdf`)
- **Response (`200 OK`)**:
  ```json
  {
    "status": "deleted",
    "saved_name": "a1b2c3d4.pdf",
    "message": "File and corresponding vector entries removed successfully"
  }
  ```
- **Error Responses**: `404 Not Found` if file does not exist.

---

### 3.2 Retrieval & Querying (`/api/v1/search`)

#### `POST /api/v1/search/query`
Perform semantic vector similarity search against the document knowledge base.

- **Request Body (`application/json`)**:
  ```json
  {
    "query": "What are the main findings in the report?",
    "n_results": 5,
    "filter_metadata": {
      "file_type": "pdf"
    }
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "query": "What are the main findings in the report?",
    "total_results": 3,
    "results": [
      {
        "rank": 1,
        "score": 0.8924,
        "content": "The study demonstrates a 25% increase in efficiency...",
        "metadata": {
          "source_file": "a1b2c3d4.pdf",
          "file_type": "pdf",
          "page": 2,
          "chunk_index": 4
        },
        "id": "e4d3c2b1-1234-5678-90ab-cdef12345678"
      }
    ]
  }
  ```
- **Error Responses**:
  - `400 Bad Request`: Empty query string.
  - `500 Internal Server Error`: Model or vector store query error.

---

### 3.3 Health & Collection Management (`/api/v1/health` & `/api/v1/vector-store`)

#### `GET /api/v1/health`
Check backend server status, embedding model status, and ChromaDB connectivity.

- **Response (`200 OK`)**:
  ```json
  {
    "status": "healthy",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_model_loaded": true,
    "vector_store_collection": "rag_documents",
    "total_indexed_documents": 54
  }
  ```

#### `POST /api/v1/vector-store/reset`
Delete and re-initialize the ChromaDB collection.

- **Response (`200 OK`)**:
  ```json
  {
    "status": "success",
    "message": "Vector store collection cleared successfully"
  }
  ```

---

## 4. Pydantic Data Schemas (`app/schemas/`)

### Document Schemas (`app/schemas/document.py`)
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class FileInfoSchema(BaseModel):
    original_filename: str
    saved_name: str
    saved_path: str
    size_bytes: int
    extension: str

class ProcessingSummarySchema(BaseModel):
    total_pages_or_docs: int
    total_chunks: int
    indexed_vector_count: int

class DocumentUploadResponse(BaseModel):
    status: str = "success"
    message: str
    file_info: FileInfoSchema
    processing_summary: Optional[ProcessingSummarySchema] = None
```

### Search Schemas (`app/schemas/search.py`)
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of matches to return")
    filter_metadata: Optional[Dict[str, Any]] = None

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
```

---

## 5. Integration Mapping to `src/` Subsystems

| FastAPI Endpoint | Core `src/` Function / Class | Execution Flow |
| :--- | :--- | :--- |
| `POST /api/v1/documents/upload` | `data_loader.upload_file`<br>`data_loader.load_documents`<br>`data_loader.chunk_documents`<br>`embedding.EmbeddingManager`<br>`vector_store.VectorStoreManager` | 1. Stream file payload to `Uploads/`<br>2. Extract text/pages<br>3. Split into overlapping chunks<br>4. Generate 384-dim embeddings<br>5. Insert into ChromaDB collection |
| `GET /api/v1/documents/` | `vector_store.VectorStoreManager` | Retrieve collection document counts and stored file entries |
| `DELETE /api/v1/documents/{name}`| `vector_store.VectorStoreManager` & `pathlib.Path.unlink` | Delete file from disk and filter/purge entries in ChromaDB |
| `POST /api/v1/search/query` | `search.SearchManager` | 1. `EmbeddingManager.generate_embedding([query])`<br>2. `VectorStoreManager.query()`<br>3. Compute similarity score ($1 - d/2$) |
| `GET /api/v1/health` | `EmbeddingManager` & `VectorStoreManager` | Inspect initialization state and total count |

---

## 6. Implementation Notes & Best Practices

1. **Singleton Dependencies**: `EmbeddingManager` and `VectorStoreManager` should be instantiated as application lifespan singletons to prevent expensive model reloading on each HTTP request.
2. **CORS Configuration**: Configure `CORSMiddleware` in `app/main.py` allowing origins like `http://localhost:3000` (Next.js frontend).
3. **File Handling Security**: Retain the existing `uuid` filename generation logic in `data_loader.py` to prevent arbitrary path traversal.
4. **Streaming & Memory**: For large files, stream `UploadFile.file` directly to disk before running `load_documents`.
