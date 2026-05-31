"""
app/services/emergency_service.py
──────────────────────────────────
Runs keyword-based emergency detection BEFORE any vector retrieval.

Why keyword-first?
  Vector search is great for semantic similarity but is too slow and
  uncertain for life-threatening situations. Simple substring matching
  on a curated list is faster, more reliable, and safer here.

If a match is found, Streamlit shows the emergency message immediately
and the LLM is NOT called.
"""

import json
from pathlib import Path
from typing import Optional


# ── Load rules once at import time ───────────────────────────────────────────

_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "emergency_rules.json"

def _load_rules() -> dict:
    """Load emergency rules from JSON. Raises FileNotFoundError if missing."""
    if not _RULES_PATH.exists():
        raise FileNotFoundError(
            f"Emergency rules file not found: {_RULES_PATH}\n"
            "Make sure config/emergency_rules.json exists."
        )
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_RULES: dict = _load_rules()


# ── Public API ────────────────────────────────────────────────────────────────

def check_emergency(query: str) -> Optional[dict]:
    """
    Check whether the query contains any emergency phrase.

    Parameters
    ----------
    query : str
        Raw user input from the Streamlit search box.

    Returns
    -------
    dict or None
        If an emergency is detected:
          {
            "is_emergency": True,
            "rule_id": "cardiac",
            "severity": "CRITICAL",
            "matched_phrases": ["chest pain"],
            "message": "⚠️ EMERGENCY: ...",
            "footer": "⚕️ MedMitra is not a substitute ..."
          }
        If no emergency is detected: None
    """
    if not query or not query.strip():
        return None

    query_lower = query.lower().strip()

    for rule in _RULES.get("emergency_rules", []):
        matched = [
            phrase for phrase in rule.get("phrases", [])
            if phrase.lower() in query_lower
        ]
        if matched:
            return {
                "is_emergency": True,
                "rule_id": rule["id"],
                "severity": rule["severity"],
                "matched_phrases": matched,
                "message": rule["message"],
                "footer": _RULES.get("emergency_footer", ""),
            }

    return None


def format_emergency_response(result: dict) -> str:
    """
    Format an emergency detection result into a clean string for display.
    Used by both Streamlit and command-line test scripts.
    """
    lines = [
        result["message"],
        "",
        result.get("footer", ""),
    ]
    return "\n".join(lines)
