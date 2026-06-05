import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Validate that INTERNAL_API_KEY is set in the environment
if not os.getenv("INTERNAL_API_KEY"):
    raise RuntimeError("Configuration Error: INTERNAL_API_KEY is not configured on the server.")

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import register_exception_handlers
from app.core.middleware import process_request_middleware, setup_cors
from app.api.routes import health, admin, emergency, search, rag, prescriptions, inventory, metrics, agent
from app.services.container import container


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing MedMitra ML Search Service...")

    # --- Startup status check ---
    import json

    # Qdrant status
    qdrant_ok = container.qdrant_service.is_connected()
    collection_name = settings.QDRANT_COLLECTION
    collection_exists = container.qdrant_service.check_collection_exists(collection_name) if qdrant_ok else False

    disease_count = 0
    medicine_count = 0
    if qdrant_ok and collection_exists:
        disease_count = container.qdrant_service.get_record_type_count(collection_name, "disease")
        medicine_count = container.qdrant_service.get_record_type_count(collection_name, "medicine")
    else:
        # Fallback: count from processed JSON if available
        for path, attr in [
            (settings.processed_diseases_path, None),
            (settings.disease_demo_path, None),
        ]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        disease_count = len(json.load(f))
                    break
                except Exception:
                    pass

        for path in [settings.processed_medicines_path, settings.medicine_demo_path]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        medicine_count = len(json.load(f))
                    break
                except Exception:
                    pass

    # Print startup summary
    print(f"[MedMitra] Qdrant connected    : {qdrant_ok}")
    print(f"[MedMitra] Collection '{collection_name}' exists: {collection_exists}")
    print(f"[MedMitra] Disease records indexed   : {disease_count}")
    print(f"[MedMitra] Medicine records indexed  : {medicine_count}")
    print(f"[MedMitra] LLM model                 : {settings.LLM_MODEL} ({settings.LLM_PROVIDER})")
    print(f"[MedMitra] Agent enabled             : {settings.AGENT_ENABLED}")
    print(f"[MedMitra] Emergency rules           : {settings.emergency_rules_path}")

    if not qdrant_ok:
        logger.warning(
            "Qdrant is not connected. Search will fall back to local numpy/BM25 indexes. "
            "Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
        )
    if disease_count == 0 and medicine_count == 0:
        logger.warning(
            "No records indexed. Run the following to populate the search index:\n"
            "  1. Place Kaggle CSVs in app/data/raw/\n"
            "  2. python -m app.scripts.clean_datasets\n"
            "  3. python -m app.scripts.build_qdrant_index"
        )

    yield
    logger.info("Shutting down MedMitra ML Search Service...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedMitra ML Search Service",
        description=(
            "Medical information retrieval service using MedCPT + BM25 hybrid search with RRF fusion. "
            "Data sourced from Kaggle datasets (prototype_unverified). "
            "This service provides general informational content only — it does NOT diagnose or prescribe."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs"
    )
    setup_cors(app)
    app.middleware("http")(process_request_middleware)
    register_exception_handlers(app)

    app.include_router(health.router, prefix="/health")
    app.include_router(metrics.router)
    app.include_router(emergency.router, prefix=settings.API_V1_PREFIX)
    app.include_router(search.router, prefix=settings.API_V1_PREFIX)
    app.include_router(rag.router, prefix=settings.API_V1_PREFIX)
    app.include_router(agent.router, prefix=settings.API_V1_PREFIX)
    app.include_router(prescriptions.router, prefix=settings.API_V1_PREFIX)
    app.include_router(inventory.router, prefix=settings.API_V1_PREFIX)
    app.include_router(admin.router, prefix=f"{settings.API_V1_PREFIX}/admin")
    return app


app = create_app()
