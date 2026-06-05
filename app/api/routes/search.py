from fastapi import APIRouter, Depends, status
from app.schemas.disease_search import DiseaseSearchRequest, DiseaseSearchResponse, DiseaseSearchResult
from app.schemas.medicine_search import MedicineSearchRequest, MedicineSearchResponse, MedicineSearchResult
from app.core.security import verify_internal_api_key
from app.services.container import container

router = APIRouter()


@router.post(
    "/search/disease",
    response_model=DiseaseSearchResponse,
    tags=["Search"],
    summary="Disease / symptom information search",
    description=(
        "Retrieves general informational content about diseases and conditions matching the query. "
        "Always runs emergency detection first. Uses MedCPT + BM25 hybrid retrieval with RRF fusion. "
        "Does NOT diagnose. Returns prototype_unverified Kaggle data."
    )
)
async def search_disease(
    payload: DiseaseSearchRequest,
    internal_key: None = Depends(verify_internal_api_key)
):
    res = container.disease_search_service.search(
        query=payload.query,
        age_group=payload.age_group.value,
        top_k=payload.top_k,
        include_full_text=payload.include_full_text,
    )

    # Build typed result objects
    result_objs = []
    for r in res.get("results", []):
        result_objs.append(DiseaseSearchResult(
            rank=r.get("rank", 0),
            condition_name=r.get("condition_name", ""),
            matched_symptoms=r.get("matched_symptoms", []),
            description=r.get("description", ""),
            precautions=r.get("precautions", []),
            source_name=r.get("source_name", ""),
            source_type=r.get("source_type", ""),
            dataset_slug=r.get("dataset_slug", ""),
            review_status=r.get("review_status", "prototype_unverified"),
            rrf_score=r.get("rrf_score", 0.0),
            dense_score=r.get("dense_score"),
        ))

    message = res.get("emergency_message")
    if not message and not result_objs:
        message = (
            "No sufficiently relevant informational matches were found in the available sources. "
            "Please consult a qualified healthcare professional."
        )

    return DiseaseSearchResponse(
        status="success",
        query=payload.query,
        normalized_query=res.get("normalized_query"),
        age_group=payload.age_group,
        emergency_detected=res.get("emergency_detected", False),
        emergency_message=res.get("emergency_message"),
        retrieval_relevance=res["retrieval_relevance"],
        results=result_objs,
        message=message,
        disclaimer=res.get("disclaimer", DiseaseSearchResponse.model_fields["disclaimer"].default),
    )


@router.post(
    "/search/medicine",
    response_model=MedicineSearchResponse,
    tags=["Search"],
    summary="Medicine / drug information search",
    description=(
        "Retrieves general informational content about medicines matching the query. "
        "Supports typo correction (e.g. paracetmol → paracetamol). "
        "Antibiotic results include mandatory prescription warnings. "
        "Does NOT prescribe. Returns prototype_unverified Kaggle data."
    )
)
async def search_medicine(
    payload: MedicineSearchRequest,
    internal_key: None = Depends(verify_internal_api_key)
):
    from fastapi import HTTPException
    
    # Run emergency detection first
    em_res = container.emergency_detector.check_emergency(payload.query)
    if em_res.is_emergency:
        raise HTTPException(
            status_code=400,
            detail=f"EMERGENCY DETECTED: {em_res.message}"
        )
        
    res = container.medicine_search_service.search(
        query=payload.query,
        top_k=payload.top_k,
        allow_typo_correction=payload.allow_typo_correction,
        include_full_text=payload.include_full_text,
    )


    result_objs = []
    for r in res.get("results", []):
        result_objs.append(MedicineSearchResult(
            rank=r.get("rank", 0),
            medicine_name=r.get("medicine_name", ""),
            generic_name=r.get("generic_name", ""),
            category=r.get("category", ""),
            uses=r.get("uses", []),
            side_effects=r.get("side_effects", []),
            warnings=r.get("warnings", []),
            mechanism_of_action=r.get("mechanism_of_action", ""),
            salt_composition=r.get("salt_composition", ""),
            source_name=r.get("source_name", ""),
            source_type=r.get("source_type", ""),
            dataset_slug=r.get("dataset_slug", ""),
            review_status=r.get("review_status", "prototype_unverified"),
            match_type=r.get("match_type", "semantic_match"),
            rrf_score=r.get("rrf_score", 0.0),
            dense_score=r.get("dense_score"),
        ))

    return MedicineSearchResponse(
        status="success",
        query=payload.query,
        corrected_query=res.get("corrected_query"),
        retrieval_relevance=res["retrieval_relevance"],
        results=result_objs,
        disclaimer=res.get("disclaimer", MedicineSearchResponse.model_fields["disclaimer"].default),
    )
