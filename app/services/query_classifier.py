from typing import Dict, Any, Optional
from app.models.enums import AgeGroup, TopicGroup
from app.core.exceptions import MedicalSafetyException

class QueryClassifier:
    def classify(
        self,
        query: str,
        age_group: AgeGroup,
        age_years: Optional[int] = None,
        age_months: Optional[int] = None,
        duration_days: Optional[int] = None
    ) -> Dict[str, Any]:
        query_clean = query.strip().lower()
        if age_months is not None and age_months < 2:
            raise MedicalSafetyException(
                "For infants under 2 months of age, symptom search is unavailable. Consult pediatrician immediately.",
                code="INFANT_SAFETY_LIMITATION"
            )
            
        unsupported = {"laptop", "repair", "crypto", "stocks", "programming", "code"}
        if any(x in query_clean for x in unsupported):
            return {"category": "unsupported", "age_group": age_group, "topic_group": None}
            
        if age_group == AgeGroup.CHILD:
            return {
                "category": "child",
                "age_group": AgeGroup.CHILD,
                "topic_group": TopicGroup.PEDIATRIC
            }
            
        # Adult queries search both core and extended topic groups
        return {
            "category": "adult",
            "age_group": AgeGroup.ADULT,
            "topic_group": None # Searches both core and extended
        }
