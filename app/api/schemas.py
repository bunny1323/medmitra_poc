from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="I have severe chest pain")
    top_k: int = Field(default=5, ge=1, le=20)

class SourceItem(BaseModel):
    page: str
    content: str

class QueryResponse(BaseModel):
    query: str
    severity_index: str
    confidence_score: float
    answer: str
    possible_diseases: List[str]
    home_cautions: List[str]
    sources: List[SourceItem]
    error: Optional[str] = None
