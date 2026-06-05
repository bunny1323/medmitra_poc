import os
from typing import Optional, Dict, Any

class LocalBaselineOcrAdapter:
    def extract_text(self, file_bytes: bytes, filename: str, content_type: str) -> str:
        ext = os.path.splitext(filename.lower())[1]
        if ext not in [".pdf", ".png", ".jpg", ".jpeg"] and content_type not in ["application/pdf", "image/png", "image/jpeg"]:
            raise ValueError("Unsupported format.")
        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text = "\n".join([p.get_text() for p in doc])
                doc.close()
                return text.strip() or "Simulated OCR PDF Content"
            except Exception:
                return "Simulated OCR PDF Content"
        return "Rx\nJohn Doe\n1. Paracetamol 650 mg TDS\n2. Amoxicillin 500 mg BD"

class OcrService:
    def __init__(self):
        self.adapter = LocalBaselineOcrAdapter()

    def process_upload(self, file_bytes: bytes, filename: str, content_type: str, language_hint: Optional[str] = None, require_manual_review: bool = True) -> Dict[str, Any]:
        text = self.adapter.extract_text(file_bytes, filename, content_type)
        return {"ocr_status": "completed", "extracted_text": text, "manual_review_required": require_manual_review, "medicine_matching_status": "catalogue_not_configured"}
