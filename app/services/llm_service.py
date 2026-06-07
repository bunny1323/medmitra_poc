"""
app/services/llm_service.py
────────────────────────────
Calls Groq / Llama to generate a JSON response with an empathetic answer and a severity index.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_LLAMA_MODEL  = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")

_SYSTEM_PROMPT = """You are MedMitra, an empathetic, caring, and highly knowledgeable medical assistant.
You are talking directly to a patient who is describing their symptoms or asking a medical question.

STRICT RULES:
1. CLEAR, ACCESSIBLE LANGUAGE: Do not use complex medical jargon that a layperson wouldn't understand. Maintain a professional, empathetic, and reassuring tone. Do not be overly casual, but avoid being overly technical.
2. DO NOT OVERHYPE SEVERITY: If a patient has a headache, a cold, or a mild fever, it is highly likely a common ailment. DO NOT jump to severe conditions unless there are extreme red flags. Most routine queries should be "NORMAL".
3. PATIENT EMPATHY & RED FLAGS: Make the patient feel heard. In your `answer`, you MUST explicitly call out "red flag" symptoms. Tell the patient exactly what to look out for (e.g., "If you begin to experience X, Y, or Z, please visit a doctor immediately.").
4. POSSIBLE DISEASES: Provide 2 to 3 possible conditions it could be, ALWAYS prioritizing the most common and least severe ones first. 
5. HOME CAUTIONS: Provide basic, safe things the patient can do at home to feel better or be cautious (e.g., drink water, rest, avoid bright lights).
6. SEVERITY INDEX:
   - "NORMAL": Routine questions, common symptoms (headaches, cold, mild pain). 
   - "URGENT": Symptoms needing a doctor soon but not instantly life-threatening (e.g., persistent high fever, moderate unexplained pain).
   - "CRITICAL": Life-threatening emergencies (e.g., severe chest pain, stroke signs, extreme bleeding).

OUTPUT FORMAT:
You must return a valid JSON object with EXACTLY four keys: "answer", "possible_diseases", "home_cautions", and "severity_index". 
Do NOT wrap the JSON in markdown code blocks. Just output the raw JSON object.
Example:
{
  "answer": "I'm sorry you're experiencing this headache. Based on the medical information, these symptoms often align with a tension headache or a common cold. However, please monitor your symptoms closely. If you develop a sudden, severe headache unlike any you've had before, a stiff neck, or visual changes, you should visit a doctor immediately.",
  "possible_diseases": ["Tension Headache", "Common Cold", "Dehydration"],
  "home_cautions": ["Drink plenty of water", "Rest in a quiet, dark room", "Apply a cool cloth to your forehead"],
  "severity_index": "NORMAL"
}
"""

def is_groq_configured() -> bool:
    return bool(_GROQ_API_KEY) and _GROQ_API_KEY not in (
        "add_your_own_key_here", "YOUR_GROQ_API_KEY"
    )

def generate_response(
    query: str,
    retrieval_result: dict[str, Any],
    emergency_warning: str = None
) -> dict[str, Any]:
    
    results = retrieval_result.get("results", [])

    if not results:
        return _fallback("No relevant medical information was found in the book.", "NORMAL")

    if not is_groq_configured():
        return _fallback("Groq API not configured. Cannot process LLM response.", "NORMAL")

    context_text = _build_context(results)

    user_message = (
        f"USER QUERY: {query}\n\n"
        f"CONTEXT (From Medical Book - Use this for accuracy but explain it VERY SIMPLY):\n{context_text}\n\n"
    )
    
    if emergency_warning:
        user_message += f"\nCRITICAL RULE MATCH: The system detected an emergency keyword. You MUST include this warning exactly in your answer: '{emergency_warning}' and you MUST set the severity_index to 'CRITICAL'.\n\n"
        
    user_message += "Generate the JSON response as instructed."

    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_API_KEY)
        response = client.chat.completions.create(
            model=_LLAMA_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        
        return {
            "answer": parsed.get("answer", "I couldn't generate a proper response."),
            "possible_diseases": parsed.get("possible_diseases", []),
            "home_cautions": parsed.get("home_cautions", []),
            "severity_index": parsed.get("severity_index", "NORMAL"),
            "model_used": _LLAMA_MODEL,
            "source": "groq",
        }

    except Exception as e:
        return _fallback(f"Groq API call failed: {str(e)}. Please try again.", "NORMAL")


def _build_context(results: list[dict]) -> str:
    lines = []
    for r in results[:5]:  # use top 5 chunks
        page = r.get("metadata", {}).get("page", "Unknown")
        content = r.get("content", "").strip()
        lines.append(f"[Page {page}] {content}")
    return "\n\n".join(lines)


def _fallback(message: str, severity: str) -> dict:
    return {
        "answer": message,
        "possible_diseases": [],
        "home_cautions": [],
        "severity_index": severity,
        "model_used": "local_fallback",
        "source": "local_fallback",
    }
