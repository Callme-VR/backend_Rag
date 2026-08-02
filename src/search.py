import numpy as np
from typing import List, Dict, Any, Optional

try:
    from src.embedding import EmbeddingManager
    from src.vector_store import VectorStoreManager
except ImportError:
    from .embedding import EmbeddingManager
    from .vector_store import VectorStoreManager


class SearchManager:
    """Handles search queries against the vector store using embeddings."""

    def __init__(self, embedding_manager: EmbeddingManager, vector_store_manager: VectorStoreManager):
        """
        Initialize the search manager.

        args:
          embedding_manager: instance of EmbeddingManager for generating query embeddings
          vector_store_manager: instance of VectorStoreManager for querying stored documents
        """
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.

        args:
          query: search query string
          n_results: number of results to return

        returns:
          List of result dicts with 'content', 'metadata', and 'score'
        """
        print(f"\nSearching for: '{query}'")

        # Generate embedding for query
        query_embedding = self.embedding_manager.generate_embedding([query])

        # Query the vector store
        raw_results = self.vector_store_manager.query(
            query_embedding=query_embedding[0],
            n_results=n_results
        )

        # Format results
        results = self._format_results(raw_results)

        print(f"Found {len(results)} result(s)")
        return results

    def search_with_scores(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents and include similarity scores.

        args:
          query: search query string
          n_results: number of results to return

        returns:
          List of result dicts with 'content', 'metadata', 'score', and 'rank'
        """
        results = self.search(query, n_results)

        # Add rank to each result
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results

    def _format_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format raw ChromaDB results into a clean list of dicts."""
        results = []

        if not raw_results or not raw_results.get("documents"):
            return results

        documents = raw_results["documents"][0] if raw_results["documents"] else []
        metadatas = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
        distances = raw_results["distances"][0] if raw_results.get("distances") else []
        ids = raw_results["ids"][0] if raw_results.get("ids") else []

        for i in range(len(documents)):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            distance = distances[i] if i < len(distances) else 0
            similarity = 1 - (distance / 2)

            results.append({
                "content": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": round(similarity, 4),
                "id": ids[i] if i < len(ids) else "",
            })

        return results
