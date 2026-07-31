# for loading the pdf from client and user ,let say when the user want to upload the docs in our rag pipeline system,then it processs it for rag

import shutil
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any

UPLOAD_DIRECTORY = Path("Uploads")
UPLOAD_DIRECTORY.mkdir(exist_ok=True)


ALLOWED_EXTENSION = {".pdf", ".doc", ".docx", ".txt", ".rtf", ".json", ".pptx", ".csv", ".md"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 mb


def upload_file(source_path) -> dict:
    """
    Copies a file into the uploads directory with validation.
    source_path: path to the file you want to upload (e.g. from a file picker or request).
    Returns dict with saved path, filename, and size.
    """
    src_path_name = Path(source_path)

    if not src_path_name.exists():
        raise ValueError(
            "No path provided"
        )
    ext = src_path_name.suffix.lower()
    if ext not in ALLOWED_EXTENSION:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSION}")

    file_size = src_path_name.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            f"File is too large, maximum allowed {MAX_FILE_SIZE / 1000 / 1000} MB"
        )
    # generates the unique number filename to avoid collision/overwrite
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = UPLOAD_DIRECTORY / unique_name

    shutil.copy2(src_path_name, dest_path)
    return {
        "original_filename": src_path_name.name,
        "saved_name": unique_name,
        "saved_path": str(dest_path),
        "size": file_size,
        "extension": ext
    }


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a document from the given file path and return a list of
    document dicts with 'content' and 'metadata' keys.

    Supports: .pdf, .txt, .md
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    documents = []

    if ext == ".pdf":
        documents = _load_pdf(path)
    elif ext in (".txt", ".md"):
        documents = _load_text(path)
    else:
        print(f"Unsupported file type for loading: {ext}, attempting as text")
        documents = _load_text(path)

    print(f"Loaded {len(documents)} document(s) from {path.name}")
    return documents


def _load_pdf(path: Path) -> List[Dict[str, Any]]:
    """Load a PDF file page by page using PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        documents = []
        for page in pages:
            documents.append({
                "content": page.page_content,
                "metadata": {
                    "source_file": path.name,
                    "file_type": "pdf",
                    "page": page.metadata.get("page", 0),
                }
            })
        return documents
    except ImportError:
        print("PyPDFLoader not available, install langchain-community and pypdf")
        return []
    except Exception as e:
        print(f"Error loading PDF {path.name}: {e}")
        return []


def _load_text(path: Path) -> List[Dict[str, Any]]:
    """Load a plain text or markdown file."""
    try:
        content = path.read_text(encoding="utf-8")
        return [{
            "content": content,
            "metadata": {
                "source_file": path.name,
                "file_type": path.suffix.lower().lstrip("."),
            }
        }]
    except Exception as e:
        print(f"Error loading text file {path.name}: {e}")
        return []


def load_directory(directory_path: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Load all supported documents from a directory.

    args:
      directory_path: path to directory containing documents
      extensions: list of extensions to filter (e.g. ['.pdf', '.txt']). None = all supported.

    returns:
      List of document dicts with 'content' and 'metadata'.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if extensions is None:
        extensions = [".pdf", ".txt", ".md"]

    all_documents = []
    files = []
    for ext in extensions:
        files.extend(dir_path.glob(f"**/*{ext}"))

    print(f"Found {len(files)} file(s) to process in {directory_path}")

    for file_path in sorted(files):
        try:
            docs = load_documents(str(file_path))
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 250) -> List[Dict[str, Any]]:
    """
    Split documents into smaller chunks for embedding.

    args:
      documents: list of document dicts with 'content' and 'metadata'
      chunk_size: maximum characters per chunk
      chunk_overlap: overlap between consecutive chunks

    returns:
      List of chunked document dicts with 'content' and 'metadata'.
    """
    separators = ["\n\n", "\n", ". ", " "]
    chunked = []

    for doc in documents:
        text = doc["content"]
        metadata = doc["metadata"]

        if len(text) <= chunk_size:
            chunked.append({
                "content": text,
                "metadata": {**metadata, "chunk_index": 0}
            })
            continue

        chunks = _split_text(text, chunk_size, chunk_overlap, separators)
        for i, chunk in enumerate(chunks):
            chunked.append({
                "content": chunk,
                "metadata": {**metadata, "chunk_index": i}
            })

    print(f"Split {len(documents)} document(s) into {len(chunked)} chunk(s)")

    if chunked:
        print(f"\nExample chunk preview:")
        print(chunked[0]["content"][:350])
        print(f"\nMetadata: {chunked[0]['metadata']}")

    return chunked


def _split_text(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    """Recursively split text using separators, similar to LangChain's RecursiveCharacterTextSplitter."""
    chunks = []
    # Try each separator in order
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            current_chunk = ""
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # Handle overlap
                    if chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + sep + part
                    else:
                        current_chunk = part
            if current_chunk:
                chunks.append(current_chunk.strip())
            return [c for c in chunks if c]

    # Fallback: split by chunk_size
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks
