"""
RAG Pipeline — Main Entry Point

Pipeline flow: Upload → Load → Chunk → Embed → Store → Search
"""

from pathlib import Path

from data_loader import upload_file, load_directory, chunk_documents
from embedding import EmbeddingManager
from vector_store import VectorStoreManager
from search import SearchManager


def main():
    print("=" * 70)
    print("  RAG Pipeline — Starting")
    print("=" * 70)

    # -----------------------------------------------------------
    # Step 1: Load documents from the Uploads directory
    # -----------------------------------------------------------
    uploads_dir = Path("Uploads")
    uploads_dir.mkdir(exist_ok=True)

    print("\n[Step 1] Loading documents from Uploads/")
    documents = load_directory(str(uploads_dir))

    if not documents:
        print("\nNo documents found in Uploads/. Place PDF, TXT, or MD files there and run again.")
        print("Tip: Use data_loader.upload_file('path/to/file') to copy a file into Uploads/.")
        return

    # -----------------------------------------------------------
    # Step 2: Chunk the documents
    # -----------------------------------------------------------
    print("\n[Step 2] Chunking documents")
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=250)

    if not chunks:
        print("No chunks created. Check your documents.")
        return

    # -----------------------------------------------------------
    # Step 3: Generate embeddings
    # -----------------------------------------------------------
    print("\n[Step 3] Generating embeddings")
    embedding_manager = EmbeddingManager()

    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedding_manager.generate_embedding(texts)

    # -----------------------------------------------------------
    # Step 4: Store in vector store
    # -----------------------------------------------------------
    print("\n[Step 4] Storing in vector store")
    vector_store = VectorStoreManager(
        collection_name="rag_documents",
        persist_directory="./chroma_db"
    )

    metadatas = [chunk["metadata"] for chunk in chunks]
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # -----------------------------------------------------------
    # Step 5: Run a sample search
    # -----------------------------------------------------------
    print("\n[Step 5] Running sample search")
    search_manager = SearchManager(
        embedding_manager=embedding_manager,
        vector_store_manager=vector_store
    )

    sample_query = "What is this document about?"
    results = search_manager.search_with_scores(sample_query, n_results=3)

    print("\n" + "=" * 70)
    print(f"  Search Results for: '{sample_query}'")
    print("=" * 70)

    for result in results:
        print(f"\n  Rank #{result['rank']} | Score: {result['score']}")
        print(f"  Source: {result['metadata'].get('source_file', 'unknown')}")
        print(f"  Content: {result['content'][:200]}...")
        print("-" * 70)

    print(f"\nPipeline complete. {vector_store.get_collection_count()} documents in vector store.")


if __name__ == "__main__":
    main()
