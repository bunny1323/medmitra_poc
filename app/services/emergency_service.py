"""
app/services/emergency_service.py
──────────────────────────────────
Keyword-based emergency rule checker using config/emergency_rules.json.
"""

import json
from pathlib import Path
from typing import Optional

_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "emergency_rules.json"

def _load_rules() -> dict:
    if not _RULES_PATH.exists():
        return {}
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_RULES = _load_rules()

def check_emergency(query: str) -> Optional[dict]:
    if not query or not query.strip():
        return None

    query_lower = query.lower()

    for rule in _RULES.get("emergency_rules", []):
        for phrase in rule.get("phrases", []):
            if phrase.lower() in query_lower:
                return {
                    "is_emergency": True,
                    "severity": rule.get("severity", "CRITICAL"),
                    "message": rule.get("message", "⚠️ EMERGENCY DETECTED.")
                }
    return None
