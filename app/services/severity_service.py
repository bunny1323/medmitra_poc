"""
app/services/severity_service.py
──────────────────────────────────
Deterministic classification of patient query severity.
"""

from app.services.emergency_service import _RULES, is_negated

# URGENT symptom rules as defined in instructions
URGENT_PHRASES = [
    "persistent high fever",
    "severe dehydration",
    "continuous vomiting",
    "severe abdominal pain",
    "child unable to drink",
    "rapidly worsening symptoms"
]

def check_severity(query: str) -> dict:
    """
    Evaluates a query and returns its severity index (CRITICAL, URGENT, NORMAL)
    and list of triggering phrases, using deterministic checks and negation rules.
    """
    if not query or not query.strip():
        return {
            "severity_index": "NORMAL",
            "severity_reasons": []
        }

    query_lower = query.lower()
    
    # 1. Check CRITICAL phrases (emergency rules list)
    critical_matches = []
    for rule in _RULES.get("emergency_rules", []):
        for phrase in rule.get("phrases", []):
            phrase_lower = phrase.lower()
            if phrase_lower in query_lower:
                if not is_negated(query, phrase):
                    critical_matches.append(phrase)

    if critical_matches:
        return {
            "severity_index": "CRITICAL",
            "severity_reasons": list(dict.fromkeys(critical_matches))
        }

    # 2. Check URGENT phrases
    urgent_matches = []
    for phrase in URGENT_PHRASES:
        phrase_lower = phrase.lower()
        if phrase_lower in query_lower:
            if not is_negated(query, phrase):
                urgent_matches.append(phrase)

    if urgent_matches:
        return {
            "severity_index": "URGENT",
            "severity_reasons": list(dict.fromkeys(urgent_matches))
        }

    # 3. Default to NORMAL
    return {
        "severity_index": "NORMAL",
        "severity_reasons": []
    }
