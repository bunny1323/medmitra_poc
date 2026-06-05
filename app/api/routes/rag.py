from fastapi import APIRouter, Depends, Request
from app.models.schemas import HybridRagRequest, HybridRagResponse
from app.core.security import verify_internal_api_key
from app.services.container import container

router = APIRouter()

@router.post("/chat/rag", response_model=HybridRagResponse, tags=["RAG Chat"])
async def chat_rag(payload: HybridRagRequest, request: Request, internal_key: None = Depends(verify_internal_api_key)):
    res = container.hybrid_retriever.retrieve_hybrid(payload.query, payload.top_k)
    return HybridRagResponse(
        status="success",
        query=payload.query,
        emergency_detected=res["emergency_detected"],
        emergency_message=res["emergency_message"],
        retrieval_method="medcpt_bm25_rrf",
        results=res["results"]
    )

