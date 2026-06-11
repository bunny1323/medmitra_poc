from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    ReindexRequest,
    SourceItem,
)
from app.core.security import verify_api_key
from app.services import ingestion_service
from app.services import llm_service
from app.services import retrieval_service
from app.services.emergency_service import check_emergency
from app.services.severity_service import check_severity


router = APIRouter()


# ---------------------------------------------------------------------
# Retrieval-relevance helpers
# ---------------------------------------------------------------------

def get_relevance_level(score: float) -> tuple[str, str]:
    """
    Convert the hybrid retrieval score into a descriptive relevance level.

    Important:
        This score represents retrieval relevance only.
        It is not a diagnosis probability or medical certainty score.
    """
    if score >= 0.80:
        level = "HIGH"
    elif score >= 0.50:
        level = "MEDIUM"
    elif score >= 0.20:
        level = "LOW"
    else:
        level = "VERY_LOW"

    note = (
        "This score represents retrieval relevance only. "
        "It is not a diagnosis probability."
    )

    return level, note


# ---------------------------------------------------------------------
# Main query endpoint
# ---------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
)
def handle_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """
    Handle a user medical-information query.

    Pipeline:
        1. Run deterministic emergency detection.
        2. Return immediately for emergency cases.
        3. Run hybrid dense + sparse retrieval.
        4. Calculate deterministic severity.
        5. Generate a readable answer through the Groq LLM layer.
        6. Return backend-friendly structured JSON.
    """

    # -----------------------------------------------------------------
    # 1. Emergency detection must run before retrieval or Groq
    # -----------------------------------------------------------------

    emergency_result = check_emergency(request.query)

    if emergency_result:
        relevance_level, confidence_note = get_relevance_level(0.0)

        return QueryResponse(
            query=request.query,
            answer_mode="emergency_escalation",
            severity_index=emergency_result["severity"],
            severity_reasons=emergency_result["matches"],
            retrieval_relevance_score=0.0,
            retrieval_relevance_level=relevance_level,
            confidence_note=confidence_note,
            answer=emergency_result["message"],
            home_cautions=[],
            sources=[],
            error=None,
            emergency_detected=True,
            emergency_matches=emergency_result["matches"],
        )

    # -----------------------------------------------------------------
    # 2. Hybrid retrieval
    # -----------------------------------------------------------------

    search_result = retrieval_service.search(
        query=request.query,
        top_k=request.top_k,
    )

    top_relevance = float(
        search_result.get("top_relevance", 0.0)
    )

    relevance_level, confidence_note = get_relevance_level(
        top_relevance
    )

    # -----------------------------------------------------------------
    # 3. Deterministic severity calculation
    # -----------------------------------------------------------------

    severity_info = check_severity(
        request.query
    )

    # -----------------------------------------------------------------
    # 4. Groq LLM response generation
    # -----------------------------------------------------------------

    llm_result = llm_service.generate_response(
        query=request.query,
        retrieval_result=search_result,
    )

    # -----------------------------------------------------------------
    # 5. Format source metadata
    # -----------------------------------------------------------------

    sources: list[SourceItem] = []

    for item in search_result.get("results", [])[: request.top_k]:
        metadata = item.get("metadata", {}) or {}

        sources.append(
            SourceItem(
                page=str(
                    metadata.get("page", "Unknown")
                ),
                content=item.get("content", ""),
                source_name=metadata.get(
                    "source_name",
                    "Unknown Book",
                ),
                original_filename=metadata.get(
                    "original_filename",
                    "Unknown File",
                ),
            )
        )

    # -----------------------------------------------------------------
    # 6. Return structured response
    # -----------------------------------------------------------------

    return QueryResponse(
        query=request.query,
        answer_mode=llm_result.get(
            "answer_mode",
            "general_information_fallback",
        ),
        severity_index=severity_info["severity_index"],
        severity_reasons=severity_info["severity_reasons"],
        retrieval_relevance_score=top_relevance,
        retrieval_relevance_level=relevance_level,
        confidence_note=confidence_note,
        answer=llm_result.get(
            "answer",
            "No answer could be generated.",
        ),
        home_cautions=llm_result.get(
            "home_cautions",
            [],
        ),
        sources=sources,
        error=search_result.get("error"),
        emergency_detected=False,
        emergency_matches=[],
    )


# ---------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------

@router.get(
    "/admin/books",
    dependencies=[Depends(verify_api_key)],
)
def list_books():
    """
    Return the registered medical books and ingestion metadata.
    """
    return ingestion_service.load_registry()


@router.post(
    "/admin/reindex",
    dependencies=[Depends(verify_api_key)],
)
def trigger_reindex(
    request: ReindexRequest,
):
    """
    Control the ingestion pipeline.

    Supported modes:
        append
        replace
        delete
        rebuild
    """
    mode = request.mode.lower().strip()

    try:
        if mode == "append":
            return ingestion_service.append_books()

        if mode == "replace":
            if not request.source_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="source_id is required for replace mode.",
                )

            return ingestion_service.replace_book(
                request.source_id
            )

        if mode == "delete":
            if not request.source_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="source_id is required for delete mode.",
                )

            return ingestion_service.delete_book(
                request.source_id
            )

        if mode == "rebuild":
            return ingestion_service.full_rebuild()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported reindex mode: "
                f"{request.mode}"
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reindexing failed: {exc}",
        ) from exc