import numpy as np
import uuid
from sentence_transformers import SentenceTransformer

from chromadb.config import Settings

from typing import List,Dict,Any,Tuple

from sklearn.metrics.pairwise import cosine_similarity




#llm get the llm embedding setup

class EmbeddingManager:


  def __init__(self,model_name="all-MiniLM-L6-v2"):
    """
    Intialize the embedding manager 

    args:
    model_name:huggingface model name for sentences embedding
    """
    self.model_name=model_name
    self.model=None
    self._load_model()


  # load the embedding models here in this place

  def _load_model(self):
    """Load the sentences tranformer generation using SentencesTransformer"""
    try:
      print(f"Loading embedding model: {self.model_name}")
      self.model=SentenceTransformer(self.model_name)
      dim=self.model.get_sentence_embedding_dimension()
      print(f"embedding model loaded successfully with dimension {dim}")
    except Exception as e:
      print(f"error in loading the model {self.model_name}: {e}")


  # generate the embeddings

  def generate_embedding(self,texts:List[str])->np.ndarray:
    """Generate the embedding for list texts
    args:
      texts: list of strings to embed
    returns:
      np.ndarray: array of embeddings
      
    """

    if self.model is None:
      raise ValueError(
        "model not found"
      )
    print(f"Generating embedding for {len(texts)} documents")

    embeddings=self.model.encode(
      texts,
      show_progress_bar=True
    )
    print(f"generating embedding with shape:{embeddings.shape}")

    return embeddings


if __name__ == "__main__":
  embedding_manager=EmbeddingManager()
  print(f"Embedding manager created successfully:",embedding_manager)
  