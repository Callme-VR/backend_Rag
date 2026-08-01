import shutil
from google.api.http_pb2 import Http
from search import SearchManager
from vector_store import VectorStoreManager
import os
import uuid 
from dotenv import load_dotenv
load_dotenv()


from contextlib import asynccontextmanager

from pathlib import Path

from typing import List,Dict,Any,Optional

import shutil


from schemas import UploadResponse
from fastapi import FastAPI,UploadFile,File,HTTPException,status
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel,Field

# import cors rag componenets

from data_loader import (
    load_documents,
    chunk_documents,
    ALLOWED_EXTENSION,
    MAX_FILE_SIZE,
    UPLOAD_DIRECTORY,
)


from embedding import EmbeddingManager



# global applications states

rag_states:Dict[Any,str]={}

@asynccontextmanager
async def lifespan(app:FastAPI):
  """lifespan of the application when the server starts"""
  print("=" * 60)
  print("Starting RAG FastAPI Service — Initializing Components...")
  print("=" * 60)

  # here use the lodaing sentences transformer model into ram
  embedding_mgr=EmbeddingManager(model_name="all-MiniLM-L6-v2")


  # connnect to persistant direcctory chroma db vectore stor 
  vector_store_mgr=VectorStoreManager(
    collection_name=embedding_mgr,
    persist_directory="./chroma_db"
  ) 

  # intiliaze the search manager

  search_mgr=SearchManager(
    embedding_manager=embedding_mgr,
    vector_store_manager=vector_store_mgr
  )

  # store the references in global states

  rag_states["embedding"]=embedding_mgr
  rag_states["vector_store"]=vector_store_mgr
  rag_states["search"]=search_mgr


  print("RAG Subsystems Loaded Successfully.")
  yield
  print("Shutting down RAG FastAPI Service.")
  rag_states.clear()



# ---------------------------------------------------------------------------
# FastAPI App Initialization & CORS Config
# ---------------------------------------------------------------------------

app=FastAPI(
  title="Prduction_rag_for_all works",
  description="RAG System for all works",
  version="1.0.0",
  lifespan=lifespan,
)

# cors middleware configurations

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:3000",
    "*"
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)






@app.post("/upload",response_model= UploadResponse,tags=["ingestionofDocs"])
async def upload_documents(file:UploadFile=File(...)):
  """
  upload the a documents(pdf,txt,md),chunksit,genearte embedding,and persist it to chromadb
  """
  file_path=Path(file.filename)
  exi=file_path.suffix.lower()

  if exi not in ALLOWED_EXTENSION:
    raise HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail=f"Unsupported file format '{exi}'. Allowed: {ALLOWED_EXTENSION}",
    )

    # ensure the upload path dirrectory exits

    UPLOAD_DIRECTORY.mkdir(exist_ok=True)
    unique_name=f"{uuid.uuid4.hex}{exi}"
    des_path=UPLOAD_DIRECTORY/unique_name

    # save the uploaded file

    try:
      with des_path.open("wb")as buffer:
        shutil.copyfileobj(
          file.file,buffer
        )
    except Exception as e:
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to save uploaded file: {str(e)}",
      )