import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.core.config import settings
from app.core.logging import logger

_SYSTEM_PROMPT = """You are the MedMitra medical-information assistant.
Use only the supplied retrieved context.
Do not diagnose.
Do not prescribe medicines.
Do not invent facts.
Mention when information is unavailable.
Include the source names.
Escalate emergency warning signs immediately.

STRICT RULES:
1. When the retrieved context is empty or insufficient, say exactly: "I could not find sufficiently reliable information in the available sources. Please consult a qualified healthcare professional."
2. The data is sourced from Kaggle datasets (prototype_unverified) — state this clearly.
3. Keep responses concise and factual.
"""


_FALLBACK_EMPTY = (
    "I could not find sufficiently reliable information in the available sources. "
    "Please consult a qualified healthcare professional."
)


class LlmService:
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._init_client()

    def _init_client(self):
        if settings.GROQ_API_KEY:
            try:
                self.client = OpenAI(
                    base_url=settings.LLM_BASE_URL,
                    api_key=settings.GROQ_API_KEY
                )
                logger.info(f"LLM client initialised: {settings.LLM_MODEL} via {settings.LLM_PROVIDER}")
            except Exception as e:
                logger.warning(f"LLM client init failed: {e}")

    def is_enabled(self) -> bool:
        return self.client is not None and settings.LLM_ENABLED

    def generate_grounded_rag(
        self,
        query: str,
        retrieved_records: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a grounded informational response from retrieved records.
        Returns plain text answer with disclaimer.
        Never diagnoses or prescribes.
        """
        if not self.is_enabled():
            return self._fallback_text(retrieved_records)

        if not retrieved_records:
            return _FALLBACK_EMPTY

        # Build context string from retrieved records
        context_parts = []
        for i, rec in enumerate(retrieved_records[:5], 1):
            source = rec.get("source_name", "Unknown source")
            review = rec.get("review_status", "prototype_unverified")
            record_type = rec.get("record_type", "")

            if record_type == "disease":
                name = rec.get("condition_name", "")
                desc = rec.get("description", "")
                symptoms = ", ".join(rec.get("symptoms", []))
                precautions = ", ".join(rec.get("precautions", []))
                context_parts.append(
                    f"[EVIDENCE {i}] Source: {source} ({review})\n"
                    f"Condition: {name}\n"
                    f"Symptoms: {symptoms}\n"
                    f"Description: {desc}\n"
                    f"Precautions: {precautions}"
                )
            elif record_type == "medicine":
                name = rec.get("medicine_name", "")
                category = rec.get("category", "")
                uses = ", ".join(rec.get("uses", []))
                side_effects = ", ".join(rec.get("side_effects", []))
                warnings = " | ".join(rec.get("warnings", []))
                context_parts.append(
                    f"[EVIDENCE {i}] Source: {source} ({review})\n"
                    f"Medicine: {name} | Category: {category}\n"
                    f"Uses: {uses}\n"
                    f"Side Effects: {side_effects}\n"
                    f"Warnings: {warnings}"
                )
            else:
                text = rec.get("search_text") or rec.get("description") or rec.get("snippet", "")
                context_parts.append(f"[EVIDENCE {i}] Source: {source} ({review})\n{text}")

        context_str = "\n\n".join(context_parts)

        try:
            resp = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Retrieved context:\n{context_str}\n\nUser query: {query}"}
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_text(retrieved_records)

    def generate_grounded_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Legacy method — kept for backward compatibility with existing RAG service.
        Returns structured JSON dict.
        """
        if not self.is_enabled() or not retrieved_context:
            return self._fallback(retrieved_context or [])

        context_str = ""
        for i, item in enumerate(retrieved_context):
            src = item.get("source", {})
            context_str += (
                f"--- EVIDENCE {i+1} ---\n"
                f"Source: {src.get('source_title')}, Page: {src.get('page_number')}\n"
                f"{item.get('snippet')}\n\n"
            )

        system = (
            "You are an educational healthcare assistant for MedMitra. Use ONLY the context below. "
            "Do not diagnose, prescribe, or recommend self-medication.\n"
            "Return JSON: {\"summary\": str, \"possible_causes\": [{\"name\": str, \"reason\": str, \"certainty\": str}], "
            "\"follow_up_questions\": [str], \"warning_signs\": [str], \"next_steps\": [str], "
            "\"disclaimer\": str, \"sources\": [{\"source_title\": str, \"page_number\": int}]}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context_str}\nQuery: {query}"}
                ],
                temperature=settings.LLM_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content)
            parsed["sources"] = [
                {"source_title": c["source"]["source_title"], "page_number": c["source"]["page_number"]}
                for c in retrieved_context
            ]
            return parsed
        except Exception:
            return self._fallback(retrieved_context)

    def _fallback_text(self, records: List[Dict]) -> str:
        if not records:
            return _FALLBACK_EMPTY
        names = [r.get("condition_name") or r.get("medicine_name") or r.get("title", "") for r in records[:3]]
        listed = ", ".join(n for n in names if n)
        return (
            f"Retrieved general information about: {listed}. "
            "For detailed guidance, please consult a qualified healthcare professional. "
            "Disclaimer: This information is general and not a medical diagnosis or prescription."
        )

    def _fallback(self, context: list) -> dict:
        sources = [
            {"source_title": c["source"]["source_title"], "page_number": c["source"]["page_number"]}
            for c in context if "source" in c
        ]
        return {
            "summary": "Educational details are degraded — LLM is unconfigured or timed out. Retrieved sources are cited.",
            "possible_causes": [{"name": "Retrieved match", "reason": "Matched retrieved records", "certainty": "possible"}] if context else [],
            "follow_up_questions": ["How long have you felt these symptoms?"],
            "warning_signs": ["Difficulty breathing", "Chest pain"],
            "next_steps": ["Please seek prompt medical assessment."],
            "disclaimer": "This is educational information only and not a diagnosis.",
            "sources": sources
        }
