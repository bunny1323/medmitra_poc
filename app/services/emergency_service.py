"""
app/services/emergency_service.py
──────────────────────────────────
Keyword-based emergency rule checker using config/emergency_rules.json with negation checking.
"""

import json
import re
from pathlib import Path
from typing import Optional

_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "emergency_rules.json"

def _load_rules() -> dict:
    if not _RULES_PATH.exists():
        return {}
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_RULES = _load_rules()

def is_negated(text: str, phrase: str) -> bool:
    """
    Checks if a phrase is negated in the text by looking for negation words
    preceding the phrase (within 30 characters).
    """
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    
    if phrase_lower not in text_lower:
        return False
        
    for match in re.finditer(re.escape(phrase_lower), text_lower):
        start_idx = match.start()
        # Look back up to 30 characters
        lookback = text_lower[max(0, start_idx - 30):start_idx]
        
        # Match negation terms on word boundaries
        negation_patterns = [
            r"\bno\b", r"\bnot\b", r"\bdon't\b", r"\bdont\b", r"\bdo\s+not\b",
            r"\bnever\b", r"\bwithout\b", r"\bdenies\b", r"\bdeny\b", r"\bfree\s+of\b",
            r"\bclear\s+of\b"
        ]
        if any(re.search(pat, lookback) for pat in negation_patterns):
            return True
    return False

def check_emergency(query: str) -> Optional[dict]:
    if not query or not query.strip():
        return None

    query_lower = query.lower()
    matched_phrases = []
    severities = []
    
    # Iterate through all rules and gather matches
    for rule in _RULES.get("emergency_rules", []):
        for phrase in rule.get("phrases", []):
            phrase_lower = phrase.lower()
            if phrase_lower in query_lower:
                # Check negation
                if not is_negated(query, phrase):
                    matched_phrases.append(phrase)
                    severities.append(rule.get("severity", "CRITICAL"))

    if not matched_phrases:
        return None

    # Deduplicate matching phrases while preserving order
    matched_phrases = list(dict.fromkeys(matched_phrases))
    
    # Severity is CRITICAL if any matched severity is CRITICAL, else URGENT
    final_severity = "CRITICAL" if "CRITICAL" in severities else "URGENT"
    
    default_msg = "This may require urgent medical attention. Please contact emergency medical services or visit the nearest hospital immediately."
    
    return {
        "is_emergency": True,
        "severity": final_severity,
        "matches": matched_phrases,
        "message": default_msg
    }
