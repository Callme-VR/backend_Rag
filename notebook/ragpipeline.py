
from operator import length_hint
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from pathlib import Path


# first thing for the rag to find all docs from storage ,when user uploaded in user client then server will process it and then store it somewhere else
# step-1
# Readll the Pdfs inside the directory


def process_all_pdf(pdf_directory):
  """ Process all pdf files in directory"""

  # find all the pdf recusively
  all_documents=[]
  pdf_dir=Path(pdf_directory)
  pdf_files=list(pdf_dir.glob("**/*.pdf"))

  print(f"Found {len(pdf_files)} PDF files to process")
  if not pdf_files:
    print("No pdf files found")
    return []
  
  for pdf_file in pdf_files:
    print(f"\n processing the {pdf_file.name}")
    try:
      loader=PyPDFLoader(str(pdf_file))
      documents=loader.load()

      for doc in documents:
        doc.metadata['sources_files']=pdf_file.name
        doc.metadata["files_type"]='pdf'

      all_documents.extend(
        documents
      )
      print(f"  ✓ Loaded {len(documents)} pages")
    except Exception as e:
      print(f"Error processing {pdf_file.name}: {e}")


  print("\n=============================================")
  print(f"Total documents loaded..:{len(all_documents)}")
  print("=============================================")

  return all_documents

  # step-2



# ------------------------------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# ------------------------------------------------

def chunks_splitting(documents,chunks_size=1000,chunks_overlap=250):


  text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=chunks_size,
    chunk_overlap=chunks_overlap,
    length_function=len,
    separators=["\n\n","\n","."," "],
  )
  split_docs=text_splitter.split_documents(documents)
  print(f"\n split {len(documents)} documents into {len(split_docs)} chunks")

  # show the example chunks

  if split_docs:
    print("\n example chunks")
    print(split_docs[0].page_content[:350])
    print("\n metadata:",split_docs[0].metadata)

  return split_docs


# Resolve backend root directory (parent of the notebook directory)
backend_dir = Path(__file__).resolve().parent.parent

# Point to 'Uploads' folder where data_loader.py saves files uploaded by the user client
pdf_directory = backend_dir / "Uploads"

# Ensure the uploads directory exists
pdf_directory.mkdir(exist_ok=True)

# step-1
# load the pdfs
documents=process_all_pdf(pdf_directory)
chunked_data=chunks_splitting(documents)
print(f"Total chunks created: {len(chunked_data)}")

# step:3
# Initialize embeddings
# 






# step:3

# Initialize embeddings