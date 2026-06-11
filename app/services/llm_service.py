"""
app/services/llm_service.py
────────────────────────────
Calls Groq / Llama to generate a JSON response following strict safety guidelines.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from app.core import config

_SYSTEM_PROMPT = """You are MedMitra, a healthcare-information assistant.

Use only the supplied retrieved context when context is available.

Rules:
1. Do not diagnose the user.
2. Do not prescribe medicines.
3. Do not provide personalized dosage instructions.
4. Do not recommend antibiotics.
5. Do not override deterministic emergency results.
6. Do not invent facts or citations.
7. If context is insufficient, say so clearly.
8. Keep responses simple and readable.
9. Return JSON only.

Your response must be a JSON object with the following schema:
{
  "answer_mode": "retrieval_grounded" | "general_information_fallback" | "safe_refusal",
  "answer": "Your simple, empathetic, and clear answer explanation summarizing retrieved context...",
  "home_cautions": ["List of 2-3 safe home cautions (e.g. rest, hydrate)..."]
}
"""

def is_groq_configured() -> bool:
    return bool(config.GROQ_API_KEY) and config.GROQ_API_KEY not in (
        "add_your_own_key_here", "YOUR_GROQ_API_KEY", "replace_with_groq_key"
    )

def check_unsafe_dosage_or_antibiotic(query: str) -> Optional[dict[str, Any]]:
    """
    Local pre-filter to detect queries asking for dosage/prescriptions/antibiotics.
    Intercepts and returns a safe refusal immediately.
    """
    query_lower = query.lower()
    
    dosage_keywords = [
        "how many tablets", "how many pills", "how much dosage", "dosage instructions", 
        "prescribe", "prescription", "how much dose", "how many doses", "milligram", "mg of"
    ]
    antibiotic_keywords = [
        "antibiotic", "amoxicillin", "penicillin", "azithromycin", "ciprofloxacin", 
        "doxycycline", "erythromycin", "metronidazole", "clarithromycin", "cephalexin"
    ]
    
    has_dosage_query = any(k in query_lower for k in dosage_keywords)
    has_antibiotic_query = any(k in query_lower for k in antibiotic_keywords)
    
    # Intercept requests asking for medicine recommendations or how to take them
    if (has_dosage_query and any(w in query_lower for w in ["take", "use", "need", "give", "eat", "consume"])) or \
       (has_antibiotic_query and any(w in query_lower for w in ["take", "use", "should i", "prescription", "recommend", "give", "write", "buy", "for"])):
        return {
            "answer_mode": "safe_refusal",
            "answer": "I cannot provide personalized dosage or antibiotic advice. Please consult a qualified healthcare professional.",
            "home_cautions": [
                "Seek guidance from a licensed healthcare practitioner.",
                "Do not self-prescribe or take unprescribed antibiotics."
            ]
        }
    return None

def generate_response(
    query: str,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    # 1. First, check local safety rules
    safety_check = check_unsafe_dosage_or_antibiotic(query)
    if safety_check:
        return safety_check

    results = retrieval_result.get("results", [])

    # If no results or retrieval failed
    if not results or retrieval_result.get("error"):
        return _fallback("No relevant medical information was found in the database. Please consult a doctor.", "general_information_fallback")

    if not is_groq_configured():
        # Fallback summary response using retrieved texts directly
        summary_text = _build_context(results)
        return {
            "answer_mode": "general_information_fallback",
            "answer": f"Groq API not configured. Here is direct text from medical guidelines: {summary_text[:300]}...",
            "home_cautions": ["Consult a medical practitioner.", "Monitor symptoms."]
        }

    context_text = _build_context(results)

    user_message = (
        f"USER QUERY: {query}\n\n"
        f"CONTEXT (From Medical Guidelines):\n{context_text}\n\n"
        f"Generate the JSON response matching the requested system prompt and schema."
    )

    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=config.LLAMA_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )
        
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        
        return {
            "answer_mode": parsed.get("answer_mode", "retrieval_grounded"),
            "answer": parsed.get("answer", "I couldn't generate a proper response."),
            "home_cautions": parsed.get("home_cautions", []),
            "model_used": config.LLAMA_MODEL,
            "source": "groq",
        }

    except Exception as e:
        return _fallback(f"Groq API call failed: {str(e)}. Please check your connection.", "general_information_fallback")


def _build_context(results: list[dict]) -> str:
    lines = []
    for r in results[:5]:  # use top 5 chunks
        page = r.get("metadata", {}).get("page", "Unknown")
        book_name = r.get("metadata", {}).get("source_name", "Unknown Book")
        content = r.get("content", "").strip()
        lines.append(f"[{book_name} - Page {page}] {content}")
    return "\n\n".join(lines)


def _fallback(message: str, answer_mode: str) -> dict:
    return {
        "answer_mode": answer_mode,
        "answer": message,
        "home_cautions": [
            "Monitor symptoms closely.",
            "Seek medical advice if symptoms worsen."
        ],
        "model_used": "local_fallback",
        "source": "local_fallback",
    }
