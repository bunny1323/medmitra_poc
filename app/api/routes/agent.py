from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.security import verify_internal_api_key
from app.services.container import container

router = APIRouter()


class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000, description="User's question or symptom description")
    session_id: Optional[str] = Field(None, description="Optional session identifier for context tracking")


class AgentSourceDetail(BaseModel):
    source_name: str
    source_type: str
    dataset_slug: str = ""
    review_status: str = "prototype_unverified"


class AgentChatResponse(BaseModel):
    status: str = "success"
    query: str
    corrected_query: Optional[str] = None
    tool_used: List[str] = []
    emergency_detected: bool = False
    emergency_message: Optional[str] = None
    intent: Optional[str] = None
    answer: str
    sources: List[AgentSourceDetail] = []
    disclaimer: str = (
        "This information is general and informational only. It is not a medical diagnosis, "
        "prescription, or clinical advice. Always consult a qualified healthcare professional."
    )
    retrieval_relevance: str = "LOW"


@router.post(
    "/agent/chat",
    response_model=AgentChatResponse,
    tags=["Agent"],
    summary="MedMitra restricted medical information agent",
    description=(
        "Deterministic tool-routing agent for medical information queries. "
        "Always runs emergency check first. Routes to medicine_search or disease_search "
        "based on query intent, then generates a grounded LLM answer from retrieved records. "
        "Does NOT diagnose, prescribe, or recommend antibiotics without prescription warning."
    )
)
async def agent_chat(
    payload: AgentChatRequest,
    internal_key: None = Depends(verify_internal_api_key)
):
    result = container.agent_service.run(
        query=payload.query,
        session_id=payload.session_id,
    )

    sources = [
        AgentSourceDetail(
            source_name=s.get("source_name", ""),
            source_type=s.get("source_type", ""),
            dataset_slug=s.get("dataset_slug", ""),
            review_status=s.get("review_status", "prototype_unverified"),
        )
        for s in result.get("sources", [])
    ]

    return AgentChatResponse(
        status=result.get("status", "success"),
        query=result["query"],
        corrected_query=result.get("corrected_query"),
        tool_used=result.get("tool_used", []),
        emergency_detected=result.get("emergency_detected", False),
        emergency_message=result.get("emergency_message"),
        intent=result.get("intent"),
        answer=result.get("answer", ""),
        sources=sources,
        disclaimer=result.get("disclaimer", ""),
        retrieval_relevance=result.get("retrieval_relevance", "LOW"),
    )
