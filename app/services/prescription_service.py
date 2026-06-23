"""
app/services/prescription_service.py
Handles parsing of prescription images using Groq Vision and post-processing medicine output.
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

# Expanded medicine list for fuzzy matching
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

    # Common Indian brands / common names
    "Dolo 650", "Crocin", "Augmentin", "Azee", "Pan 40", "Pantocid",
    "Montek LC", "Telma", "Ecosprin", "Shelcal", "Becosules", "Zerodol",
    "Taxim", "Cefixime", "ORS", "Sinarest", "Allegra", "Meftal",
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


def _clean_text(value: Any, default: str = "Not specified") -> str:
    """Convert value to clean string."""
    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    return text


def _clean_medicine_name(name: str) -> str:
    """Clean medicine name formatting before fuzzy match."""
    name = _clean_text(name)

    if name == "Not specified":
        return name

    # Collapse multiple spaces
    name = " ".join(name.split())

    # Title-case only if all caps / ugly casing
    # Example: PARACETAMOL -> Paracetamol
    # Example: cetirizine -> Cetirizine
    if name.isupper() or name.islower():
        name = name.title()

    return name


def _fuzzy_match_medicine(name: str) -> str:
    """Try to match a noisy medicine name to a known medicine list."""
    if not name or name.lower() == "not specified":
        return name

    cleaned = _clean_medicine_name(name)

    matches = difflib.get_close_matches(
        cleaned,
        COMMON_MEDICINES,
        n=1,
        cutoff=0.72,   # stronger than 0.6
    )
    if matches:
        return matches[0]

    return cleaned


def _normalize_confidence(value: Any) -> str:
    """
    Convert confidence into High / Medium / Low.
    Handles:
    - numeric string like "0.9"
    - float/int
    - existing labels
    """
    if value is None:
        return "Medium"

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return "Medium"

        lower = raw.lower()
        if lower in {"high", "medium", "low"}:
            return raw.title()

        # Try numeric parse
        try:
            numeric = float(raw)
        except ValueError:
            return "Medium"
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        return "Medium"

    if numeric >= 0.85:
        return "High"
    if numeric >= 0.65:
        return "Medium"
    return "Low"


def _normalize_medicines(medicines: Any) -> List[Dict[str, str]]:
    """Ensure medicine list always has consistent keys and cleaned values."""
    if not isinstance(medicines, list):
        return []

    normalized: List[Dict[str, str]] = []

    for med in medicines:
        if not isinstance(med, dict):
            continue

        original_name = med.get("name", "Not specified")
        corrected_name = _fuzzy_match_medicine(original_name)

        normalized.append(
            {
                "name": corrected_name or "Not specified",
                "dosage": _clean_text(med.get("dosage", "Not specified")),
                "frequency": _clean_text(med.get("frequency", "Not specified")),
                "duration": _clean_text(med.get("duration", "Not specified")),
                "confidence": _normalize_confidence(med.get("confidence", "Medium")),
            }
        )

    return normalized


def parse_prescription_image(base64_image: str) -> Dict[str, Any]:
    """
    Parse a prescription image using Groq Vision.

    Returns:
    {
        "medicines": [...],
        "doctor_notes": "...",
        "unreadable_text_present": bool,
        "raw_extracted_text": "...",
        "error": None | str
    }
    """
    if not _GROQ_API_KEY:
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "raw_extracted_text": None,
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

        content = content.strip()

        parsed = json.loads(content)

        medicines = _normalize_medicines(parsed.get("medicines", []))

        return {
            "medicines": medicines,
            "doctor_notes": _clean_text(parsed.get("doctor_notes", ""), default=""),
            "unreadable_text_present": bool(parsed.get("unreadable_text_present", False)),
            "raw_extracted_text": content,
            "error": None,
        }

    except json.JSONDecodeError:
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "raw_extracted_text": None,
            "error": "Failed to parse model response into JSON. Response might be malformed.",
        }
    except Exception as exc:
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "raw_extracted_text": None,
            "error": f"Error parsing prescription: {str(exc)}",
        }