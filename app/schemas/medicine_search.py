from pydantic import BaseModel, Field
from typing import List, Optional
from app.models.enums import RelevanceLabel


class MedicineSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Medicine name or description of use")
    top_k: int = Field(default=5, ge=1, le=10)
    allow_typo_correction: bool = True
    include_full_text: bool = False


class MedicineSearchResult(BaseModel):
    rank: int
    medicine_name: str
    generic_name: str = ""
    category: str = ""
    uses: List[str] = []
    side_effects: List[str] = []
    warnings: List[str] = []
    mechanism_of_action: str = ""
    salt_composition: str = ""
    source_name: str = ""
    source_type: str = ""
    dataset_slug: str = ""
    review_status: str = "prototype_unverified"
    match_type: str = "semantic_match"
    rrf_score: float = 0.0
    dense_score: Optional[float] = None


class MedicineSearchResponse(BaseModel):
    status: str = "success"
    query: str
    corrected_query: Optional[str] = None
    retrieval_relevance: RelevanceLabel = RelevanceLabel.LOW
    results: List[MedicineSearchResult] = []
    disclaimer: str = "This is general medicine information. Do not start, stop or change medication without advice from a qualified healthcare professional."

