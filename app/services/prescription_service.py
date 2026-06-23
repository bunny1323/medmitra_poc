"""
app/services/prescription_service.py
Handles parsing of prescription images using Groq Vision Model and fuzzy matching.
"""

from __future__ import annotations

import os
import json
import difflib
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Simple medicine dictionary for fuzzy matching
COMMON_MEDICINES = [
    "Amoxicillin", "Paracetamol", "Ibuprofen", "Aspirin", "Azithromycin",
    "Cetirizine", "Metformin", "Omeprazole", "Pantoprazole", "Ciprofloxacin",
    "Doxycycline", "Erythromycin", "Atorvastatin", "Amlodipine", "Losartan",
    "Levothyroxine", "Albuterol", "Gabapentin", "Sertraline", "Fluoxetine",
    "Citalopram", "Escitalopram", "Trazodone", "Bupropion", "Venlafaxine",
    "Duloxetine", "Clonazepam", "Lorazepam", "Alprazolam", "Diazepam",
    "Hydrochlorothiazide", "Furosemide", "Spironolactone", "Clopidogrel",
    "Warfarin", "Rivaroxaban", "Apixaban", "Dabigatran", "Enoxaparin",
    "Insulin Glargine", "Insulin Lispro", "Insulin Aspart", "Insulin Detemir",
]

_SYSTEM_PROMPT = """You are an expert medical transcriptionist.
Your job is to read handwritten medical prescriptions and extract the text accurately.
Doctors' handwriting can be messy. Use your best judgment and context clues.

OUTPUT FORMAT:
Return ONLY a valid JSON object with EXACTLY these keys:
- "medicines": list of medicine objects
- "doctor_notes": string
- "unreadable_text_present": boolean

Each medicine object must contain:
- "name"
- "dosage"
- "frequency"
- "duration"
- "confidence"

If a field is missing, use "Not specified".
Do NOT wrap the JSON in markdown.
Output ONLY raw JSON.
"""


def _fuzzy_match_medicine(name: str) -> str:
    """Try to match a noisy medicine name to a known medicine list."""
    if not name or name.lower() == "not specified":
        return name

    matches = difflib.get_close_matches(name.title(), COMMON_MEDICINES, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return name


def _normalize_medicines(medicines: Any) -> List[Dict[str, str]]:
    """Ensure medicine list always has consistent keys."""
    if not isinstance(medicines, list):
        return []

    normalized: List[Dict[str, str]] = []

    for med in medicines:
        if not isinstance(med, dict):
            continue

        original_name = str(med.get("name", "Not specified")).strip()
        corrected_name = _fuzzy_match_medicine(original_name)

        normalized.append(
            {
                "name": corrected_name or "Not specified",
                "dosage": str(med.get("dosage", "Not specified")).strip() or "Not specified",
                "frequency": str(med.get("frequency", "Not specified")).strip() or "Not specified",
                "duration": str(med.get("duration", "Not specified")).strip() or "Not specified",
                "confidence": str(med.get("confidence", "Medium")).strip() or "Medium",
            }
        )

    return normalized


def parse_prescription_image(base64_image: str) -> Dict[str, Any]:
    """
    Parse a prescription image using Groq Vision.
    Returns a dict:
    {
        "medicines": [...],
        "doctor_notes": "...",
        "unreadable_text_present": bool,
        "error": None | str
    }
    """
    if not _GROQ_API_KEY:
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "error": "Groq API key not configured.",
        }

    try:
        from groq import Groq

        client = Groq(api_key=_GROQ_API_KEY)

        # Ensure proper data URL format
        if not base64_image.startswith("data:image"):
            image_url = f"data:image/jpeg;base64,{base64_image}"
        else:
            image_url = base64_image

        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        content = response.choices[0].message.content.strip()

        # Cleanup accidental markdown wrappers
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        parsed = json.loads(content.strip())

        medicines = _normalize_medicines(parsed.get("medicines", []))

        return {
            "medicines": medicines,
            "doctor_notes": parsed.get("doctor_notes", "") or "",
            "unreadable_text_present": bool(parsed.get("unreadable_text_present", False)),
            "error": None,
        }

    except json.JSONDecodeError:
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "error": "Failed to parse model response into JSON. Response might be malformed.",
        }
    except Exception as exc:
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "error": f"Error parsing prescription: {str(exc)}",
        }