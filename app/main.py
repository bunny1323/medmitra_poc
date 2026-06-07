from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

app = FastAPI(
    title="MedMitra API",
    description="FastAPI backend for MedMitra semantic search and RAG.",
    version="1.0.0"
)

# Enable CORS for the Express backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update this to your Express backend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MedMitra API is running"}
