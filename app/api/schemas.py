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

class MedicineDetail(BaseModel):
    name: str = Field(..., description="The name of the medicine")
    dosage: Optional[str] = Field(None, description="The dosage strength (e.g. 500mg)")
    frequency: Optional[str] = Field(None, description="How often to take it (e.g. Twice a day)")
    duration: Optional[str] = Field(None, description="How long to take it (e.g. 5 days)")
    confidence: str = Field(..., description="Confidence level of the extraction: High, Medium, or Low")

class PrescriptionResponse(BaseModel):
    medicines: List[MedicineDetail] = Field(default_factory=list, description="List of medicines parsed from the prescription")
    doctor_notes: Optional[str] = Field(None, description="Any additional notes or instructions from the doctor")
    unreadable_text_present: bool = Field(False, description="Whether some parts of the prescription were completely unreadable")
    error: Optional[str] = Field(None, description="Error message if parsing failed")
