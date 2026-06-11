from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.services.retrieval_service import get_qdrant_client, collection_exists
from app.services.llm_service import is_groq_configured
from app.core import config

app = FastAPI(
    title="MedMitra API",
    description="FastAPI backend for MedMitra semantic search and RAG.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health/live", tags=["health"])
def health_live():
    return {"status": "live", "message": "MedMitra API is running"}

@app.get("/health/ready", tags=["health"])
def health_ready():
    # 1. Verify Groq API key is configured
    if not is_groq_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq API key not configured"
        )
        
    # 2. Verify Qdrant connection and collection existence
    try:
        client = get_qdrant_client()
        coll_name = config.QDRANT_COLLECTION
        
        # Check collection existence
        if not collection_exists(coll_name):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Qdrant collection '{coll_name}' does not exist"
            )
            
        # Count points
        count_res = client.count(collection_name=coll_name)
        if count_res.count == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Qdrant collection '{coll_name}' has 0 points"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant readiness check failed: {str(e)}"
        )
        
    return {
        "status": "ready",
        "qdrant": "connected",
        "collection": config.QDRANT_COLLECTION,
        "groq": "configured"
    }

# Keep original fallback health check
@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "message": "MedMitra API is running"}
