from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The medical query/symptoms to search for")
    top_k: int = Field(default=5, ge=1, le=20)

class SourceItem(BaseModel):
    page: str
    content: str
    source_name: Optional[str] = None
    original_filename: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer_mode: str
    severity_index: str
    severity_reasons: List[str] = Field(default_factory=list)
    retrieval_relevance_score: float
    retrieval_relevance_level: str
    confidence_note: str
    answer: str
    home_cautions: List[str] = Field(default_factory=list)
    sources: List[SourceItem] = Field(default_factory=list)
    error: Optional[str] = None
    emergency_detected: Optional[bool] = None
    emergency_matches: Optional[List[str]] = None

class ReindexRequest(BaseModel):
    mode: str = Field("append", description="Reindex mode: append, replace, delete, rebuild")
    book_filename: Optional[str] = Field(None, description="Filename of the book to process")
    source_id: Optional[str] = Field(None, description="Source ID of the book to delete or replace")
