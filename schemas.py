from importlib import metadata
from pydantic import Field
from typing import Dict,List,Any
from pydantic import BaseModel
class SearchRequest(BaseModel):
  query:str=Field(...,example="What is the capital of France?")
  n_result:int=Field(...,default=5,ge=1,le=20,example=5)

class SearchResultItems(BaseModel):
  rank:int
  id:str
  metadata:Dict[str,Any]
  score:int


class SearhResponse(BaseModel):
  query:str
  total_result:int
  results:List[SearchResultItems]

class UploadResponse(BaseModel):
  status:str
  filename:str
  saved_path:str
  chunks_created:int
  total_documents_in_store: int



class HealthResponse(BaseModel):
    status: str
    model: str
    total_documents_in_store: int