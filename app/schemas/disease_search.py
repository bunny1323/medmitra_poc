from pydantic import BaseModel, Field
from typing import List, Optional
from app.models.enums import AgeGroup, RelevanceLabel


class DiseaseSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Symptom description or disease query")
    age_group: AgeGroup = AgeGroup.ADULT
    duration_days: Optional[int] = Field(default=None, ge=0, le=3650)
    top_k: int = Field(default=5, ge=1, le=10)
    include_full_text: bool = False


class DiseaseSearchResult(BaseModel):
    rank: int
    condition_name: str
    matched_symptoms: List[str] = []
    description: str = ""
    precautions: List[str] = []
    source_name: str = ""
    source_type: str = ""
    dataset_slug: str = ""
    review_status: str = "prototype_unverified"
    rrf_score: float = 0.0
    dense_score: Optional[float] = None


class DiseaseSearchResponse(BaseModel):
    status: str = "success"
    query: str
    normalized_query: Optional[str] = None
    age_group: AgeGroup = AgeGroup.ADULT
    emergency_detected: bool = False
    emergency_message: Optional[str] = None
    retrieval_relevance: RelevanceLabel = RelevanceLabel.LOW
    results: List[DiseaseSearchResult] = []
    message: Optional[str] = None
    disclaimer: str = "These are informational retrieval results and not a diagnosis. Consult a qualified healthcare professional."

