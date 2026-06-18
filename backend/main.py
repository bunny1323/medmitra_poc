from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings
from services.rag_service import initialize_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        count = initialize_vector_store()
        print(f"✓ Vector store ready with {count} medicine documents")
    except Exception as exc:
        print(f"⚠ Vector store initialization deferred: {exc}")
    yield


app = FastAPI(
    title="MedMitra AI",
    description="Healthcare RAG Chatbot API powered by Ollama & Llama 3.2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "MedMitra AI",
        "docs": "/docs",
        "health": "/api/health",
    }
