from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.ollama_service import check_ollama_health, generate_response
from services.rag_service import (
    build_context,
    compute_confidence,
    initialize_vector_store,
    search_medicines,
)
from utils.emergency_detector import detect_emergency

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class RetrievedSource(BaseModel):
    name: str
    category: str
    similarity: float


class ChatResponse(BaseModel):
    response: str
    confidence: float
    is_emergency: bool
    emergency_message: str | None = None
    sources: list[RetrievedSource] = Field(default_factory=list)


@router.get("/health")
async def health_check():
    ollama = await check_ollama_health()
    return {
        "status": "ok",
        "service": "MedMitra AI",
        "ollama": ollama,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    is_emergency, emergency_msg = detect_emergency(request.message)

    if is_emergency:
        return ChatResponse(
            response=(
                "I've detected language that may indicate a medical emergency.\n\n"
                "⚠️ Please contact emergency services immediately.\n"
                "MedMitra AI cannot provide emergency medical care."
            ),
            confidence=0.95,
            is_emergency=True,
            emergency_message=emergency_msg,
            sources=[],
        )

    retrieved = search_medicines(request.message)

    # If nothing relevant found in medicine database
    if not retrieved:
        return ChatResponse(
            response=(
                "MedMitra AI is designed only for medicine and healthcare information.\n\n"
                "Please ask questions about:\n"
                "• Medicines\n"
                "• Drug interactions\n"
                "• Side effects\n"
                "• Dosage information\n"
                "• Medical conditions\n"
                "• Healthcare guidance"
            ),
            confidence=0.20,
            is_emergency=False,
            sources=[],
        )

    context = build_context(retrieved)
    confidence = compute_confidence(retrieved, is_emergency)

    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        response_text = await generate_response(
            user_message=request.message,
            context=context,
            conversation_history=history,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI service unavailable. Ensure Ollama is running. {exc}",
        ) from exc

    sources = [
        RetrievedSource(
            name=item["metadata"].get("name", "Unknown"),
            category=item["metadata"].get("category", ""),
            similarity=item["similarity"],
        )
        for item in retrieved
    ]

    return ChatResponse(
        response=response_text,
        confidence=confidence,
        is_emergency=False,
        sources=sources,
    )


@router.post("/index")
async def reindex_medicines():
    from services import rag_service
    from services.rag_service import get_chroma_client

    client = get_chroma_client()

    try:
        client.delete_collection("medicines")
    except Exception:
        pass

    rag_service._collection = None

    count = initialize_vector_store()

    return {
        "status": "indexed",
        "documents": count,
    }