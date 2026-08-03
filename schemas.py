from typing import Dict, List, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(...)
    n_results: int = Field(...)


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
