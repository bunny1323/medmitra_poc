from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The medical query, symptom description, or first-aid question to search for.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of retrieved chunks to use for grounding the answer.",
    )


class SourceItem(BaseModel):
    page: str = Field(
        ...,
        description="Page number or page label from the source document.",
    )
    content: str = Field(
        ...,
        description="Retrieved content snippet used for answer grounding.",
    )
    source_name: Optional[str] = Field(
        default=None,
        description="Human-readable source/book name.",
    )
    original_filename: Optional[str] = Field(
        default=None,
        description="Original uploaded filename.",
    )


class QueryResponse(BaseModel):
    query: str = Field(..., description="Original user query.")
    answer_mode: str = Field(
        ...,
        description="Mode used to answer: emergency_escalation, retrieval_grounded, medical_safety_block, general_information_fallback.",
    )
    severity_index: str = Field(
        ...,
        description="Deterministic severity label such as NORMAL, URGENT, or CRITICAL.",
    )
    severity_reasons: List[str] = Field(
        default_factory=list,
        description="Matched severity triggers or reasons for the severity label.",
    )
    retrieval_relevance_score: float = Field(
        ...,
        description="Numeric retrieval relevance score after hybrid search/rerank.",
    )
    retrieval_relevance_level: str = Field(
        ...,
        description="Bucketed retrieval relevance level such as VERY_LOW, LOW, MEDIUM, HIGH.",
    )
    confidence_note: str = Field(
        ...,
        description="Clarifies that retrieval score is not a diagnosis probability.",
    )
    answer: str = Field(
        ...,
        description="Final grounded answer shown to the user.",
    )
    home_cautions: List[str] = Field(
        default_factory=list,
        description="Safety cautions / home-care guardrails.",
    )
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="Retrieved source snippets used for grounding.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if something failed, else null.",
    )
    emergency_detected: bool = Field(
        ...,
        description="Whether emergency detector triggered.",
    )
    emergency_matches: List[str] = Field(
        default_factory=list,
        description="Emergency phrases matched from the query.",
    )
    safety_blocked: bool = Field(
        ...,
        description="Whether the answer was blocked by medical safety policy.",
    )
    safety_reason: Optional[str] = Field(
        default=None,
        description="Reason why the answer was safety-blocked, if applicable.",
    )


class ReindexRequest(BaseModel):
    mode: str = Field(
        ...,
        description="append | replace | delete | rebuild",
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Required for replace/delete modes.",
    )


class PrescriptionMedicine(BaseModel):
    name: str
    dosage: str = "Not specified"
    frequency: str = "Not specified"
    duration: str = "Not specified"
    confidence: str = "Medium"


class PrescriptionResponse(BaseModel):
    medicines: List[PrescriptionMedicine] = Field(default_factory=list)
    doctor_notes: str = ""
    unreadable_text_present: bool = False
    raw_extracted_text: Optional[str] = None
    error: Optional[str] = None