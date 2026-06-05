import re

class TypoService:
    def normalize_text(self, text: str) -> str:
        text = text.replace("\t", " ").replace("\r", "")
        text = re.sub(r"[^\w\s\-\.\,\/\d]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def match_prescription_text(self, text: str, manual_review_confirmed: bool = False) -> dict:
        norm = self.normalize_text(text)
        return {
            "status": "catalogue_not_configured",
            "normalized_text": norm,
            "candidate_medicines": [],
            "manual_review_required": True,
            "message": "Medicine matching requires a verified structured medicine catalogue."
        }
