"""
This file is mainly used for RAG testing purposes.

Topics Covered:
1. Create a LangChain Document manually.
2. Create sample text files.
3. Load multiple text files using DirectoryLoader.
4. Display the loaded Document objects.
"""

import os

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ==========================================================
# 1. Create a Sample LangChain Document
# ==========================================================

doc = Document(
    page_content="vishalRajput_30jobs_plan",
    metadata={
        "source": "vishal.txt",
        "author": "Vishal Rajput",
        "date_created": "28-07-2026",
    },
)

print("=" * 70)
print("Manual LangChain Document")
print("=" * 70)
print(doc)

# ==========================================================
# 2. Create Sample Text Files
# ==========================================================

DATA_DIR = "./data/text"

os.makedirs(DATA_DIR, exist_ok=True)

sample_text = {
    os.path.join(DATA_DIR, "python.txt"): """
Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in:
- Web Development
- Data Science
- Artificial Intelligence
- Automation
- Machine Learning
""",
    os.path.join(DATA_DIR, "machine_learning.txt"): """
Machine Learning Basics

Machine Learning is a subset of Artificial Intelligence (AI) that enables systems to learn from
experience without being explicitly programmed.

Types of Machine Learning

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Applications:
- Image Recognition
- Speech Processing
- Recommendation Systems
- Fraud Detection
- Self Driving Cars
""",
    os.path.join(DATA_DIR, "langchain.txt"): """
LangChain Introduction

LangChain is a framework for building applications powered by Large Language Models (LLMs).

Major Components:
- Document Loaders
- Text Splitters
- Embeddings
- Vector Stores
- Retrievers
- Chains
- Agents

LangChain is commonly used for RAG (Retrieval-Augmented Generation) applications.
""",
}

for path, text in sample_text.items():
    with open(path, "w", encoding="utf-8") as file:
        file.write(text.strip())

print("\n✅ Sample text files created successfully.")
print(f"📂 Folder: {DATA_DIR}")

# ==========================================================
# 3. Load All Text Files
# ==========================================================

loader = DirectoryLoader(
    path=DATA_DIR,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)

documents = loader.load()

# ==========================================================
# 4. Display Loaded Documents
# ==========================================================

print("\n")
print("=" * 70)
print(f"Total Documents Loaded: {len(documents)}")
print("=" * 70)

for index, document in enumerate(documents, start=1):
    print(f"\n📄 Document {index}")
    print("-" * 70)

    print(f"Source File : {document.metadata.get('source')}")
    print(f"Metadata    : {document.metadata}")

    print("\nContent:")
    print(document.page_content)

    print("-" * 70)
