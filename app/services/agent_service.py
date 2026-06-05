"""
MedMitra Restricted Agent Service
===================================
Deterministic tool-router agent for medical information queries.

Tool routing order:
  1. emergency_check  — ALWAYS runs first for any query
  2. medicine_search  — if query is medicine-related
  3. disease_search   — if query is symptom/condition-related
  4. rag_lookup       — LLM-grounded explanation from retrieved records
  5. inventory_lookup — for availability-related questions (stub)

The agent must NEVER:
  - Diagnose a disease
  - Prescribe medicines
  - Recommend antibiotics without prescription warning
  - Ignore emergency red flags
  - Answer from its own memory when retrieval is empty

When retrieval is empty, returns the standard "not found" message.
"""

from typing import Dict, Any, List, Optional
from app.core.logging import logger
from app.models.enums import RelevanceLabel

_EMPTY_RETRIEVAL_MESSAGE = (
    "I could not find sufficiently reliable information in the available sources. "
    "Please consult a qualified healthcare professional."
)

_DISCLAIMER = (
    "This information is general and informational only. It is not a medical diagnosis, "
    "prescription, or clinical advice. Data sourced from Kaggle datasets (prototype_unverified) "
    "has not been clinically validated. Always consult a qualified healthcare professional."
)

# Keywords indicating a medicine/drug query
_MEDICINE_KEYWORDS = {
    "drug", "medicine", "medication", "tablet", "capsule", "syrup", "dosage",
    "dose", "mg", "ml", "pill", "paracetamol", "ibuprofen", "amoxicillin",
    "azithromycin", "metformin", "atorvastatin", "uses", "side effect",
    "side effects", "antibiotic", "painkiller", "analgesic",
}

# Keywords indicating a symptom/disease query
_DISEASE_KEYWORDS = {
    "symptom", "symptoms", "fever", "cough", "cold", "pain", "headache",
    "nausea", "vomiting", "diarrhea", "rash", "infection", "disease",
    "condition", "illness", "sick", "feeling", "hurt", "ache",
}


_INVENTORY_KEYWORDS = {
    "available", "availability", "stock", "in stock", "buy", "purchase", "where to get", "where to buy",
    "order", "shop", "price", "cost", "pharmacy", "store", "get paracetamol", "get medicine"
}


def _classify_query(query: str) -> str:
    """
    Simple keyword-based query classifier.
    Returns: 'inventory', 'medicine', 'disease'
    """
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in _INVENTORY_KEYWORDS):
        return "inventory"
        
    med_score = sum(1 for kw in _MEDICINE_KEYWORDS if kw in query_lower)
    dis_score = sum(1 for kw in _DISEASE_KEYWORDS if kw in query_lower)

    if med_score > dis_score:
        return "medicine"
    else:
        return "disease"  # Default to disease/symptom search



class AgentService:
    """
    Deterministic tool-router agent for MedMitra.
    Tools: emergency_check, disease_search, medicine_search, rag_lookup
    """

    def __init__(
        self,
        emergency_service,
        disease_search_service,
        medicine_search_service,
        llm_service,
    ):
        self.emergency_service = emergency_service
        self.disease_search_service = disease_search_service
        self.medicine_search_service = medicine_search_service
        self.llm_service = llm_service

    def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the agent pipeline for a given query.

        Steps:
            1. emergency_check
            2. classify → medicine_search or disease_search
            3. rag_lookup (LLM grounding)
            4. Return structured response

        Returns dict with: status, query, tool_used, emergency_detected,
                           answer, sources, disclaimer
        """
        logger.info(f"Agent query: '{query[:80]}...' (session={session_id})")
        tool_used = []
        emergency_detected = False
        emergency_message = None

        # ----------------------------------------------------------------
        # Step 1: Emergency check — ALWAYS first
        # ----------------------------------------------------------------
        tool_used.append("emergency_check")
        try:
            em_result = self.emergency_service.check_query(query)
            if em_result.is_emergency:
                emergency_detected = True
                matches = em_result.matches if hasattr(em_result, "matches") else []
                action = matches[0].action if matches else "Seek urgent medical attention immediately."
                emergency_message = em_result.message

                return {
                    "status": "success",
                    "query": query,
                    "tool_used": tool_used,
                    "emergency_detected": True,
                    "emergency_message": emergency_message,
                    "answer": (
                        f"⚠️ EMERGENCY DETECTED: {emergency_message}\n\n"
                        f"Recommended action: {action}\n\n"
                        "Do not delay — seek immediate medical attention."
                    ),
                    "sources": [],
                    "disclaimer": _DISCLAIMER,
                    "retrieval_relevance": "LOW",
                }
        except Exception as e:
            logger.error(f"Emergency check error: {e}")

        # ----------------------------------------------------------------
        # Step 2: Classify query → choose search tool
        # ----------------------------------------------------------------
        intent = _classify_query(query)
        retrieved_records: List[Dict] = []
        corrected_query = query
        search_result: Dict[str, Any] = {}

        if intent == "inventory":
            tool_used.append("inventory_lookup")
        elif intent == "medicine":
            # ── medicine_search ──
            tool_used.append("medicine_search")

            try:
                search_result = self.medicine_search_service.search(
                    query=query,
                    top_k=5,
                    allow_typo_correction=True,
                )
                corrected_query = search_result.get("corrected_query", query)
                retrieved_records = search_result.get("results", [])
                # Add record_type for LLM context building
                for r in retrieved_records:
                    r.setdefault("record_type", "medicine")
            except Exception as e:
                logger.error(f"Medicine search error in agent: {e}")

        else:
            # ── disease_search ──
            tool_used.append("disease_search")
            try:
                search_result = self.disease_search_service.search(
                    query=query,
                    age_group="adult",
                    top_k=5,
                )
                corrected_query = search_result.get("normalized_query", query)
                retrieved_records = search_result.get("results", [])
                # Add record_type for LLM context building
                for r in retrieved_records:
                    r.setdefault("record_type", "disease")
                # If disease search itself detected an emergency, propagate it
                if search_result.get("emergency_detected"):
                    emergency_detected = True
                    return {
                        "status": "success",
                        "query": query,
                        "tool_used": tool_used,
                        "emergency_detected": True,
                        "emergency_message": search_result.get("emergency_message", ""),
                        "answer": search_result.get("emergency_message", "Seek immediate medical attention."),
                        "sources": [],
                        "disclaimer": _DISCLAIMER,
                        "retrieval_relevance": "LOW",
                    }
            except Exception as e:
                logger.error(f"Disease search error in agent: {e}")

        # ----------------------------------------------------------------
        # Step 3: RAG lookup — grounded LLM answer
        # ----------------------------------------------------------------
        if retrieved_records:
            tool_used.append("rag_lookup")
            try:
                answer = self.llm_service.generate_grounded_rag(
                    query=corrected_query,
                    retrieved_records=retrieved_records,
                )
            except Exception as e:
                logger.error(f"RAG generation error: {e}")
                answer = self._format_retrieval_answer(retrieved_records, intent)
        else:
            answer = _EMPTY_RETRIEVAL_MESSAGE

        # ----------------------------------------------------------------
        # Step 4: Build sources list
        # ----------------------------------------------------------------
        sources = []
        seen_sources = set()
        for r in retrieved_records:
            src_name = r.get("source_name", "")
            if src_name and src_name not in seen_sources:
                sources.append({
                    "source_name": src_name,
                    "source_type": r.get("source_type", ""),
                    "dataset_slug": r.get("dataset_slug", ""),
                    "review_status": r.get("review_status", "prototype_unverified"),
                })
                seen_sources.add(src_name)

        relevance = search_result.get("retrieval_relevance", RelevanceLabel.LOW)
        if hasattr(relevance, "value"):
            relevance = relevance.value

        return {
            "status": "success",
            "query": query,
            "corrected_query": corrected_query if corrected_query != query else None,
            "tool_used": tool_used,
            "emergency_detected": emergency_detected,
            "emergency_message": emergency_message,
            "intent": intent,
            "answer": answer,
            "sources": sources,
            "disclaimer": _DISCLAIMER,
            "retrieval_relevance": str(relevance),
        }

    def _format_retrieval_answer(self, records: List[Dict], intent: str) -> str:
        """Simple fallback text summary when LLM is unavailable."""
        if not records:
            return _EMPTY_RETRIEVAL_MESSAGE

        lines = ["Here is general information retrieved from the available sources:\n"]
        for r in records[:3]:
            if intent == "medicine":
                name = r.get("medicine_name", "")
                uses = ", ".join(r.get("uses", []))
                lines.append(f"• {name}: {uses}")
            else:
                name = r.get("condition_name", "")
                desc = r.get("description", "")
                lines.append(f"• {name}: {desc}")

        lines.append(
            "\nDisclaimer: This is general informational content only. "
            "Not a diagnosis or prescription. Consult a healthcare professional."
        )
        return "\n".join(lines)
