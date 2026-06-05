import uuid
from typing import Optional
from app.models.enums import RelevanceLabel
from app.models.schemas import ChatRagRequest, ChatRagResponse, RagAnswerStructured, RagCitation, RagAnswerCause

class RagService:
    def __init__(self, emergency_service, query_classifier, retrieval_service, llm_service):
        self.emergency_service = emergency_service
        self.query_classifier = query_classifier
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    def process_chat_rag(self, request: ChatRagRequest, request_id: Optional[str] = None) -> ChatRagResponse:
        req_id = request_id or str(uuid.uuid4())
        
        # Emergency check
        em_res = self.emergency_service.check_query(request.query)
        if em_res.is_emergency:
            action = em_res.matches[0].action if em_res.matches else "Seek urgent medical attention."
            ans = RagAnswerStructured(
                summary=em_res.message,
                possible_causes=[],
                follow_up_questions=[],
                warning_signs=["Urgent symptoms detected"],
                next_steps=[action],
                disclaimer="EMERGENCY ALERT: Educational assistant bypassed due to potential clinical urgency."
            )
            return ChatRagResponse(request_id=req_id, is_emergency=True, retrieval_relevance=RelevanceLabel.LOW, answer=ans, sources=[])
            
        # Classify
        cl = self.query_classifier.classify(request.query, request.age_group, request.age_years, request.age_months, request.duration_days)
        if cl["category"] == "unsupported":
            ans = RagAnswerStructured(
                summary="This query appears unrelated to medical symptoms or healthcare standard guidelines.",
                possible_causes=[],
                follow_up_questions=[],
                warning_signs=[],
                next_steps=["Rephrase symptoms or ask clinical questions."],
                disclaimer="Out of scope."
            )
            return ChatRagResponse(request_id=req_id, is_emergency=False, retrieval_relevance=RelevanceLabel.LOW, answer=ans, sources=[])
            
        # Retrieve
        items, relevance = self.retrieval_service.retrieve(request.query, request.age_group.value, cl["topic_group"].value if cl["topic_group"] else None, request.top_k)
        
        # LLM RAG
        context_payload = [{"snippet": it.snippet, "source": {"source_title": it.source.source_title, "page_number": it.source.page_number}} for it in items]
        hist = [{"role": m.role, "content": m.content} for m in request.conversation_history] if request.conversation_history else []
        
        llm_res = self.llm_service.generate_grounded_response(request.query, context_payload, hist)
        
        causes = [RagAnswerCause(name=c["name"], reason=c["reason"], certainty=c.get("certainty", "possible")) for c in llm_res.get("possible_causes", [])]
        ans = RagAnswerStructured(
            summary=llm_res["summary"],
            possible_causes=causes,
            follow_up_questions=llm_res.get("follow_up_questions", []),
            warning_signs=llm_res.get("warning_signs", []),
            next_steps=llm_res.get("next_steps", []),
            disclaimer=llm_res.get("disclaimer", "This is educational information only.")
        )
        cites = [RagCitation(source_title=c["source_title"], page_number=c["page_number"]) for c in llm_res.get("sources", [])]
        return ChatRagResponse(request_id=req_id, is_emergency=False, retrieval_relevance=relevance, answer=ans, sources=cites)
