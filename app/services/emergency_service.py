import json
import os
import re
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import EmergencyCheckResponse, EmergencyMatchDetail
from app.models.enums import EmergencySeverity

class EmergencyService:
    def __init__(self, rules_path: Optional[str] = None):
        self.rules_path = rules_path or settings.curated_emergency_rules_path
        self.rules = []
        self.negation_patterns = []
        self._load_rules()

    def _load_rules(self) -> None:
        path = self.rules_path
        # Fallback to legacy path if curated rules path does not exist
        if not os.path.exists(path):
            if os.path.exists(settings.emergency_rules_path):
                path = settings.emergency_rules_path
            else:
                # Safe hardcoded fallback if JSON not copied yet
                self.negation_patterns = [r"\bno\s+", r"\bwithout\s+", r"\bdenies\s+", r"\bnot\s+having\s+"]
                self.rules = [
                    {"category": "cardiac", "phrases": ["chest pain", "chest tightness"], "severity": "urgent", "action": "Seek urgent medical assessment."},
                    {"category": "respiratory", "phrases": ["cannot breathe", "difficulty breathing"], "severity": "urgent", "action": "Seek immediate emergency medical care."},
                    {"category": "pediatric_danger_sign", "phrases": ["child unable to drink", "child is unable to drink", "child unusually sleepy", "child convulsing"], "severity": "urgent", "action": "Take child to hospital immediately."},
                    {"category": "self_harm", "phrases": ["suicidal thoughts", "want to harm myself"], "severity": "urgent", "action": "Contact suicide helpline."}
                ]
                return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_rules = data.get("rules", [])
                self.negation_patterns = data.get("negation_patterns", [])
                
                self.rules = []
                for rule in raw_rules:
                    normalized_rule = {
                        "category": rule.get("category", ""),
                        "severity": rule.get("severity", "urgent"),
                        "phrases": rule.get("patterns") or rule.get("phrases") or [],
                        "action": rule.get("message") or rule.get("action") or rule.get("user_message") or "Seek urgent medical attention."
                    }
                    self.rules.append(normalized_rule)
        except Exception as e:
            logger.error(f"Failed to load rules from {path}: {e}")

    def check_query(self, query: str) -> EmergencyCheckResponse:
        query_clean = query.strip().lower()
        matched = []
        neg_regex = "|".join(self.negation_patterns) if self.negation_patterns else ""
        
        for rule in self.rules:
            for phrase in rule.get("phrases", []):
                phrase_clean = phrase.strip().lower()
                if phrase_clean in query_clean:
                    # Negation check within same clause
                    occurrences = [m.start() for m in re.finditer(re.escape(phrase_clean), query_clean)]
                    for start_idx in occurrences:
                        prefix = query_clean[max(0, start_idx - 35):start_idx]
                        clauses = re.split(r"[,;. ut and]\s+", prefix)
                        last_clause = clauses[-1] if clauses else ""
                        is_negated = False
                        if neg_regex and re.search(neg_regex, last_clause):
                            is_negated = True
                        if not is_negated:
                            matched.append(EmergencyMatchDetail(
                                category=rule["category"],
                                matched_phrase=phrase,
                                action=rule["action"]
                            ))
                            break
            # Allow matching multiple emergency categories
        if matched:
            return EmergencyCheckResponse(
                is_emergency=True,
                severity=EmergencySeverity.URGENT,
                matches=matched,
                message="Urgent warning signs were detected. Please seek prompt medical care."
            )
        return EmergencyCheckResponse(
            is_emergency=False,
            message="No configured emergency keyword was detected. This does not rule out an emergency."
        )
