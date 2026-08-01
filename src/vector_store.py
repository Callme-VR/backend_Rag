import uuid
import numpy as np
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings


class VectorStoreManager:
    """Manages a ChromaDB vector store for document embeddings."""

    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store manager.

     args:
          collection_name: name of the ChromaDB collection
          persist_directory: directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Vector store initialized: collection='{self.collection_name}', "
                  f"persist='{self.persist_directory}'")
            print(f"Current document count: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]] = None):
        """
        Add documents with their embeddings to the vector store.

        args:
          texts: list of document text content
          embeddings: numpy array of embeddings (shape: [n_docs, embedding_dim])
          metadatas: optional list of metadata dicts for each document
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized")

        ids = [uuid.uuid4().hex for _ in range(len(texts))]

        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(texts))]

        # ChromaDB expects list of lists for embeddings
        embeddings_list = embeddings.tolist()

        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings_list[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )

        print(f"Added {len(texts)} documents to vector store")
        print(f"Total documents in collection: {self.collection.count()}")

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector store with an embedding.

        args:
          query_embedding: embedding vector for the query (shape: [embedding_dim])
          n_results: number of results to return

        returns:
          Dict with 'documents', 'metadatas', 'distances', and 'ids'
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized")

        if self.collection.count() == 0:
            print("Warning: vector store is empty, no results to return")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        # Ensure query embedding is a list
        if isinstance(query_embedding, np.ndarray):
            query_list = query_embedding.tolist()
        else:
            query_list = query_embedding

        # If 1D, wrap in a list
        if isinstance(query_list[0], (int, float)):
            query_list = [query_list]

        results = self.collection.query(
            query_embeddings=query_list,
            n_results=min(n_results, self.collection.count())
        )

        return results

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        if self.collection is None:
            return 0
        return self.collection.count()

    def delete_collection(self):
        """Delete the entire collection."""
        if self.client is not None:
            self.client.delete_collection(self.collection_name)
            self.collection = None
            print(f"Deleted collection: {self.collection_name}")
