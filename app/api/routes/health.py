from fastapi import APIRouter
from app.core.config import settings
from app.services.container import container

router = APIRouter()

@router.get("/live", tags=["Health"])
async def get_health_live():
    return {"status": "alive", "service": "medmitra-ml-search"}

@router.get("/ready", tags=["Health"])
async def get_health_ready():
    qdrant_ok = container.qdrant_service.is_connected()
    
    collection_alias = settings.QDRANT_COLLECTION_ALIAS
    collection_ok = False
    indexed_points = 0
    if qdrant_ok:
        collection_ok = container.qdrant_service.check_collection_exists(collection_alias)
        if collection_ok:
            indexed_points = container.qdrant_service.get_collection_points_count(collection_alias)
            
    llm_ok = settings.GROQ_API_KEY is not None and len(settings.GROQ_API_KEY) > 0
    status_label = "ready" if (qdrant_ok and collection_ok and llm_ok) else "degraded"
    if not qdrant_ok: status_label = "not_ready"
    
    return {
        "status": status_label,
        "qdrant_connected": qdrant_ok,
        "collection_exists": collection_ok,
        "indexed_points": indexed_points,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_configured": llm_ok
    }

