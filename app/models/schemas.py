from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from app.models.enums import AgeGroup, RelevanceLabel, EmergencySeverity

class IngestRebuildRequest(BaseModel):
    force: bool = False

class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    processed_chunks: int
    total_chunks: int
    percent: float

class EmergencyMatchDetail(BaseModel):
    category: str
    matched_phrase: str
    action: str

class EmergencyCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class EmergencyCheckResponse(BaseModel):
    is_emergency: bool
    severity: Optional[EmergencySeverity] = None
    matches: List[EmergencyMatchDetail] = []
    message: str

class DiseaseSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    age_group: AgeGroup
    age_years: Optional[int] = Field(default=None, ge=0, le=120)
    age_months: Optional[int] = Field(default=None, ge=0, le=1440)
    duration_days: Optional[int] = Field(default=None, ge=0, le=3650)
    top_k: int = Field(default=4, ge=1, le=10)
    include_full_text: bool = False

class SourceDetail(BaseModel):
    source_title: str
    source_file: str
    page_number: int
    section: Optional[str] = None
    age_group: str

class DiseaseSearchResultDetail(BaseModel):
    snippet: str = Field(..., max_length=950)
    score: float
    source: SourceDetail

class DiseaseSearchResponse(BaseModel):
    query: str
    age_group: AgeGroup
    retrieval_relevance: RelevanceLabel
    disclaimer: str = "Retrieval relevance is not a diagnosis probability."
    results: List[DiseaseSearchResultDetail] = []

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)

class ChatRagRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    age_group: AgeGroup
    age_years: Optional[int] = Field(default=None, ge=0, le=120)
    age_months: Optional[int] = Field(default=None, ge=0, le=1440)
    duration_days: Optional[int] = Field(default=None, ge=0, le=3650)
    top_k: int = Field(default=4, ge=1, le=6)
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, max_length=6)

class RagAnswerCause(BaseModel):
    name: str
    reason: str
    certainty: str = "possible"

class RagAnswerStructured(BaseModel):
    summary: str
    possible_causes: List[RagAnswerCause] = []
    follow_up_questions: List[str] = []
    warning_signs: List[str] = []
    next_steps: List[str] = []
    disclaimer: str = "This is educational information only and not a diagnosis."

class RagCitation(BaseModel):
    source_title: str
    page_number: int
    section: Optional[str] = None

class ChatRagResponse(BaseModel):
    request_id: str
    is_emergency: bool
    retrieval_relevance: RelevanceLabel
    answer: RagAnswerStructured
    sources: List[RagCitation] = []

class PrescriptionMatchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    manual_review_confirmed: bool = False

class PrescriptionMatchResponse(BaseModel):
    status: str
    normalized_text: str
    candidate_medicines: List[str] = []
    manual_review_required: bool = True
    message: str = "Medicine matching requires a verified structured medicine catalogue."

class HybridRagRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class HybridRagResult(BaseModel):
    rank: int
    chunk_id: str
    text: str
    source_name: str
    source_type: str
    page_number: int
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: float
    retrieval_method: str = "hybrid"

class HybridRagResponse(BaseModel):
    status: str = "success"
    query: str
    emergency_detected: bool
    emergency_message: Optional[str] = None
    retrieval_method: str = "medcpt_bm25_rrf"
    results: List[HybridRagResult] = []
    disclaimer: str = "This information is retrieved from verified medical sources and is not a medical diagnosis. Consult a qualified healthcare professional for medical advice."
