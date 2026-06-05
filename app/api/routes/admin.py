import os
import sys
import json
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import IngestRebuildRequest
from app.core.security import verify_internal_api_key
from app.core.config import settings
from app.services.container import container

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /api/v1/admin/index-status
# ---------------------------------------------------------------------------

@router.get("/index-status", tags=["Admin Index Management"])
async def get_index_status(internal_key: None = Depends(verify_internal_api_key)):
    """
    Returns the current status of the search indexes and infrastructure.
    Includes Qdrant connection, collection record counts, LLM status and agent status.
    """
    # Qdrant status
    qdrant_svc = container.qdrant_service
    qdrant_connected = qdrant_svc.is_connected()
    collection_name = settings.QDRANT_COLLECTION
    collection_exists = qdrant_svc.check_collection_exists(collection_name) if qdrant_connected else False

    disease_records_indexed = 0
    medicine_records_indexed = 0

    if qdrant_connected and collection_exists:
        disease_records_indexed = qdrant_svc.get_record_type_count(collection_name, "disease")
        medicine_records_indexed = qdrant_svc.get_record_type_count(collection_name, "medicine")
    else:
        # Fallback: count from local index metadata files
        d_meta = os.path.join(settings.indexes_dir, "disease_metadata.json")
        m_meta = os.path.join(settings.indexes_dir, "medicine_metadata.json")
        if os.path.exists(d_meta):
            try:
                with open(d_meta, "r", encoding="utf-8") as f:
                    disease_records_indexed = len(json.load(f))
            except Exception:
                pass
        if os.path.exists(m_meta):
            try:
                with open(m_meta, "r", encoding="utf-8") as f:
                    medicine_records_indexed = len(json.load(f))
            except Exception:
                pass

    # Emergency rules count
    emergency_rules_loaded = 0
    curated_rules_path = settings.curated_emergency_rules_path
    rules_path = settings.emergency_rules_path
    for path in [curated_rules_path, rules_path]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    emergency_rules_loaded = len(data.get("rules", []))
                break
            except Exception:
                pass

    # LLM status
    try:
        llm_enabled = container.llm_service.is_enabled()
    except Exception:
        llm_enabled = False

    return {
        "status": "success",
        "qdrant_connected": qdrant_connected,
        "collection_name": collection_name,
        "collection_exists": collection_exists,
        "disease_records_indexed": disease_records_indexed,
        "medicine_records_indexed": medicine_records_indexed,
        "emergency_rules_loaded": emergency_rules_loaded,
        "dense_vectors_ready": qdrant_connected and collection_exists,
        "sparse_vectors_ready": qdrant_connected and collection_exists,
        "llm_enabled": llm_enabled,
        "llm_model": settings.LLM_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "agent_enabled": settings.AGENT_ENABLED,
        "medcpt_query_model": settings.MEDCPT_QUERY_MODEL,
        "medcpt_article_model": settings.MEDCPT_ARTICLE_MODEL,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/admin/rebuild-index
# ---------------------------------------------------------------------------

@router.post("/rebuild-index", tags=["Admin Index Management"])
async def rebuild_index(
    payload: IngestRebuildRequest,
    internal_key: None = Depends(verify_internal_api_key)
):
    """
    Rebuild Qdrant index from processed JSON files.
    Runs build_qdrant_index.py logic synchronously.
    Requires processed JSON files to exist (run clean-data first).
    """
    try:
        # Validate that processed files exist
        if not os.path.exists(settings.processed_diseases_path):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Processed disease file not found: {settings.processed_diseases_path}. "
                    "Run POST /api/v1/admin/clean-data first."
                )
            )
        if not os.path.exists(settings.processed_medicines_path):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Processed medicine file not found: {settings.processed_medicines_path}. "
                    "Run POST /api/v1/admin/clean-data first."
                )
            )

        # Import and run the index builder
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_qdrant_index",
            os.path.join(settings.BASE_DIR, "app", "scripts", "build_qdrant_index.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

        # Return updated counts
        qdrant_svc = container.qdrant_service
        collection = settings.QDRANT_COLLECTION
        d_count = qdrant_svc.get_record_type_count(collection, "disease") if qdrant_svc.is_connected() else 0
        m_count = qdrant_svc.get_record_type_count(collection, "medicine") if qdrant_svc.is_connected() else 0

        return {
            "status": "success",
            "message": "Qdrant index rebuilt successfully.",
            "collection_name": collection,
            "disease_records_indexed": d_count,
            "medicine_records_indexed": m_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {e}")


# ---------------------------------------------------------------------------
# POST /api/v1/admin/clean-data
# ---------------------------------------------------------------------------

@router.post("/clean-data", tags=["Admin Index Management"])
async def clean_data(internal_key: None = Depends(verify_internal_api_key)):
    """
    Run the data cleaning pipeline (clean_datasets.py).
    Processes raw Kaggle CSV files from app/data/raw/ into
    app/data/processed/diseases.json and app/data/processed/medicines.json.
    Raw CSV files must be present before calling this endpoint.
    """
    raw_dir = settings.raw_data_dir
    if not os.path.exists(raw_dir):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Raw data directory not found: {raw_dir}. "
                "Create the directory and place Kaggle CSV files inside it. "
                "Download commands:\n"
                "  kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p app/data/raw --unzip\n"
                "  kaggle datasets download -d palakjain9/1000-drugs-and-side-effects -p app/data/raw --unzip"
            )
        )

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "clean_datasets",
            os.path.join(settings.BASE_DIR, "app", "scripts", "clean_datasets.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

        # Report output counts
        d_count, m_count, q_count = 0, 0, 0
        if os.path.exists(settings.processed_diseases_path):
            with open(settings.processed_diseases_path, "r", encoding="utf-8") as f:
                d_count = len(json.load(f))
        if os.path.exists(settings.processed_medicines_path):
            with open(settings.processed_medicines_path, "r", encoding="utf-8") as f:
                m_count = len(json.load(f))
        if os.path.exists(settings.symptom_queries_path):
            with open(settings.symptom_queries_path, "r", encoding="utf-8") as f:
                q_count = len(json.load(f))

        return {
            "status": "success",
            "message": "Data cleaning completed.",
            "disease_records": d_count,
            "medicine_records": m_count,
            "symptom_query_records": q_count,
            "review_status": "prototype_unverified",
            "note": "Kaggle-derived data is NOT clinically verified.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {e}")


# ---------------------------------------------------------------------------
# POST /api/v1/admin/seed-demo (kept for backward compat)
# ---------------------------------------------------------------------------

@router.post("/seed-demo", tags=["Admin Index Management"])
async def seed_demo(internal_key: None = Depends(verify_internal_api_key)):
    """Legacy endpoint — kept for backward compatibility. Returns success."""
    return {"status": "success", "message": "Demo seeding is now handled by clean-data and rebuild-index endpoints."}
