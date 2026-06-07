from fastapi import APIRouter, HTTPException
from app.api.schemas import QueryRequest, QueryResponse, SourceItem
from app.services.retrieval_service import search
from app.services.llm_service import generate_response
from app.services.emergency_service import check_emergency

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    # Step 1: vector search (Hybrid)
    result = search(
        query=request.query,
        top_k=request.top_k
    )
    
    if "error" in result and result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
        
    top_relevance = result.get("top_relevance", 0.0)
    
    # Check JSON Rules for emergencies
    emergency_warning = None
    emergency_check = check_emergency(request.query)
    if emergency_check:
        emergency_warning = emergency_check.get("message")
    
    # Step 2: LLM generation for dynamic severity and empathetic answer
    llm_result = generate_response(
        query=request.query, 
        retrieval_result=result,
        emergency_warning=emergency_warning
    )
    
    # Step 3: Format sources
    sources = []
    for r in result.get("results", [])[:request.top_k]:
        page = str(r.get("metadata", {}).get("page", "Unknown"))
        content = r.get("content", "")
        sources.append(SourceItem(page=page, content=content))
    
    return QueryResponse(
        query=request.query,
        severity_index=llm_result.get("severity_index", "NORMAL"),
        confidence_score=top_relevance,
        answer=llm_result.get("answer", "No answer could be generated."),
        possible_diseases=llm_result.get("possible_diseases", []),
        home_cautions=llm_result.get("home_cautions", []),
        sources=sources
    )
