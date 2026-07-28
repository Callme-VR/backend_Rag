# This file is mainly used for RAG testing purposes.

import os
from langchain_core.documents import Document

# --------------------------------------------------
# Create a sample LangChain Document
# --------------------------------------------------

doc = Document(
    page_content="vishalRajput_30jobs_plan",
    metadata={
        "source": "vishal.txt",
        "author": "Vishal Rajput",
        "date_created": "28-07-2026",
    },
)

print("Document Object:")
print(doc)

# --------------------------------------------------
# Create the data/text directory
# --------------------------------------------------

os.makedirs("./data/text", exist_ok=True)

# --------------------------------------------------
# Sample text documents
# --------------------------------------------------

sample_text = {
    "./data/text/python.txt": """
Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.
""",
    "./data/text/machine_learning.txt": """
Machine Learning Basics

Machine Learning is a subset of Artificial Intelligence (AI) that enables systems to learn and improve
from experience without being explicitly programmed.

Types of Machine Learning:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Applications include image recognition, speech processing, recommendation systems,
fraud detection, and autonomous vehicles.
""",
}

# --------------------------------------------------
# Write the files
# --------------------------------------------------

for path, text in sample_text.items():
    with open(path, "w", encoding="utf-8") as file:
        file.write(text.strip())

print(f"\n✅ {len(sample_text)} text files created successfully.")
print("📂 Location: ./data/text/")