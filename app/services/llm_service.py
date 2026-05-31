"""
app/services/llm_service.py
────────────────────────────
Optional Groq / Llama response generation.

Safety rules enforced in the system prompt:
  1. Use ONLY the retrieved context — never invent information.
  2. Do NOT claim a confirmed diagnosis.
  3. Do NOT prescribe dosage or treatment plans.
  4. Do NOT replace a doctor.
  5. State uncertainty when context is weak.
  6. Always suggest professional consultation.
  7. Keep answers short and patient-friendly (≤ 150 words).

Llama is called ONLY when:
  - GROQ_API_KEY is present in .env
  - Retrieval returned at least one result
  - Top relevance score ≥ MIN_RELEVANCE_SCORE (default 0.55)
  - The query is NOT flagged as an emergency

If any condition fails, a safe local fallback is returned without crashing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Configuration from environment ────────────────────────────────────────────

_GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
_LLAMA_MODEL       = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
_MIN_RELEVANCE     = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))
_MAX_TOKENS        = 400
_TEMPERATURE       = 0.1   # low temperature → more factual, less creative

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are MedMitra, a medical information assistant.

STRICT RULES — follow every rule without exception:

1. Answer ONLY using the information provided in the CONTEXT block below.
2. NEVER claim a confirmed diagnosis. Always say "may suggest" or "could indicate".
3. NEVER recommend specific dosages or prescribe treatments.
4. NEVER invent information that is not in the context.
5. If the context is weak or unrelated, say so clearly and suggest seeing a doctor.
6. Always recommend consulting a qualified healthcare professional.
7. Keep the response under 120 words. Use simple, patient-friendly language.
8. Do NOT reproduce the context verbatim — summarise in your own words.
9. End every response with: "Please consult a qualified healthcare professional."
"""

# ── Public API ─────────────────────────────────────────────────────────────────

def is_groq_configured() -> bool:
    """Return True if a non-placeholder GROQ_API_KEY is set."""
    return bool(_GROQ_API_KEY) and _GROQ_API_KEY not in (
        "add_your_own_key_here", "YOUR_GROQ_API_KEY"
    )


def generate_response(
    query: str,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a Llama-based response using retrieved context.

    Parameters
    ----------
    query            : str  — original user query
    retrieval_result : dict — output from retrieval_service.search()

    Returns
    -------
    {
        "answer":      str,
        "model_used":  str,
        "source":      "groq" | "local_fallback",
        "skipped":     bool,
        "skip_reason": str | None,
    }
    """
    top_relevance = retrieval_result.get("top_relevance", 0.0)
    results       = retrieval_result.get("results", [])

    # ── Guard: no results ────────────────────────────────────────────────────
    if not results:
        return _fallback(
            "No relevant medical information was found for this query.",
            skip_reason="No retrieval results",
        )

    # ── Guard: relevance too low ──────────────────────────────────────────────
    if top_relevance < _MIN_RELEVANCE:
        return _fallback(
            f"I could not find sufficiently relevant medical information for this query "
            f"(relevance: {top_relevance:.2f}, threshold: {_MIN_RELEVANCE:.2f}). "
            f"Please consult a qualified healthcare professional.",
            skip_reason=f"Relevance {top_relevance:.2f} below threshold {_MIN_RELEVANCE:.2f}",
        )

    # ── Guard: API key not configured ────────────────────────────────────────
    if not is_groq_configured():
        context_text = _build_context(results)
        return _fallback(
            f"Groq API not configured. Here is what the database found:\n\n{context_text}\n\n"
            "Please consult a qualified healthcare professional.",
            skip_reason="GROQ_API_KEY not set",
        )

    # ── Build context block from top results ──────────────────────────────────
    context_text = _build_context(results)

    user_message = (
        f"USER QUERY: {query}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Answer the user's query using only the context above."
    )

    # ── Call Groq API ─────────────────────────────────────────────────────────
    try:
        from groq import Groq   # imported here to avoid crash if not installed
        client = Groq(api_key=_GROQ_API_KEY)
        response = client.chat.completions.create(
            model=_LLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        answer = response.choices[0].message.content.strip()
        return {
            "answer":      answer,
            "model_used":  _LLAMA_MODEL,
            "source":      "groq",
            "skipped":     False,
            "skip_reason": None,
        }

    except Exception as e:
        # API call failed — return fallback, never crash the app
        context_text = _build_context(results)
        return _fallback(
            f"Groq API call failed ({type(e).__name__}: {e}).\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Please consult a qualified healthcare professional.",
            skip_reason=f"Groq error: {e}",
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_context(results: list[dict]) -> str:
    """Format top retrieval results into a numbered context block."""
    lines = []
    for r in results[:3]:   # use at most top-3 to keep context concise
        score   = r.get("relevance_score", 0)
        content = r.get("content", "").strip()
        lines.append(f"[{r['rank']}] (relevance {score:.2f}) {content}")
    return "\n".join(lines)


def _fallback(message: str, skip_reason: str) -> dict:
    """Return a safe local fallback response."""
    return {
        "answer":      message,
        "model_used":  "local_fallback",
        "source":      "local_fallback",
        "skipped":     True,
        "skip_reason": skip_reason,
    }
