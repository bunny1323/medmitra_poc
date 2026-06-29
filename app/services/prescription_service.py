"""
app/services/prescription_service.py
Handles parsing of prescription images using Groq Vision and post-processing medicine output.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

COMMON_MEDICINES = [
    "Aceclofenac",
    "Albuterol",
    "Amlodipine",
    "Amoxicillin",
    "Amoxycillin",
    "Aspirin",
    "Atorvastatin",
    "Azithromycin",
    "Becosules",
    "Betaloc",
    "Bupropion",
    "Cefixime",
    "Cefpodoxime",
    "Cefuroxime",
    "Cetirizine",
    "Cinnarizine",
    "Ciprofloxacin",
    "Citalopram",
    "Clopidogrel",
    "Clonazepam",
    "Crocin",
    "Dabigatran",
    "Diclofenac",
    "Dolo 650",
    "Doxycycline",
    "Duloxetine",
    "Ecosprin",
    "Enoxaparin",
    "Escitalopram",
    "Erythromycin",
    "Fluoxetine",
    "Furosemide",
    "Gabapentin",
    "Hydrochlorothiazide",
    "Ibuprofen",
    "Insulin Aspart",
    "Insulin Detemir",
    "Insulin Glargine",
    "Insulin Lispro",
    "Levofloxacin",
    "Levocetirizine",
    "Levothyroxine",
    "Levoflox",
    "Losartan",
    "Meftal",
    "Metformin",
    "Metronidazole",
    "Monocef",
    "Montek LC",
    "Nebilong",
    "Nicip",
    "Nise",
    "Norflox",
    "ORS",
    "Omeprazole",
    "Ofloxacin",
    "Pan 40",
    "Pantocid",
    "Papain",
    "Papaverine",
    "Paracetamol",
    "Rabeprazole",
    "Rantac",
    "Rivaroxaban",
    "Shelcal",
    "Sinarest",
    "Taxim",
    "Telma",
    "Trazodone",
    "Volini",
    "Warfarin",
    "Zerodol",
]

COMMON_MEDICINES = list(dict.fromkeys(COMMON_MEDICINES))
KNOWN_MEDICINE_SET = {medicine.lower() for medicine in COMMON_MEDICINES}
COMMON_MEDICINES_BY_LOWER = {medicine.lower(): medicine for medicine in COMMON_MEDICINES}

MEDICINE_ALIAS_MAP = {
    "amoxycillin": "Amoxicillin",
    "amoxillin": "Amoxicillin",
    "amoxcillin": "Amoxicillin",
    "amoxicillin": "Amoxicillin",
    "cetrizine": "Cetirizine",
    "cetrazine": "Cetirizine",
    "cetirazine": "Cetirizine",
    "cetirizine": "Cetirizine",
    "dolo": "Dolo 650",
    "dolo650": "Dolo 650",
    "dol650": "Dolo 650",
    "crocine": "Crocin",
    "crocinn": "Crocin",
    "crocin": "Crocin",
    "pantocid": "Pantocid",
    "pan40": "Pan 40",
    "pan 40": "Pan 40",
    "montek": "Montek LC",
    "monteklc": "Montek LC",
    "montek lc": "Montek LC",
    "levoflox": "Levofloxacin",
    "levofloxacin": "Levofloxacin",
    "levocetirizine": "Levocetirizine",
    "levocetirizine": "Levocetirizine",
    "metrogyl": "Metronidazole",
    "metro": "Metronidazole",
    "paracetmol": "Paracetamol",
    "paracatemol": "Paracetamol",
    "para": "Paracetamol",
    "pcm": "Paracetamol",
    "paracetamol": "Paracetamol",
    "dorzolamidum": "Dorzolamide",
    "dorzolanidum": "Dorzolamide",
    "dorzolamide": "Dorzolamide",
    "oflox": "Ofloxacin",
    "azithro": "Azithromycin",
    "cimetidin": "Cimetidine",
    "cimetidine": "Cimetidine",
    "lermin": "Cinnarizine",
    "lermina": "Cinnarizine",
}

OCR_CORRECTION_MAP = {key: value for key, value in MEDICINE_ALIAS_MAP.items()}

SUSPICIOUS_MEDICINE_TOKENS = {
    "pulang",
    "dextro",
    "volucan",
    "voluci",
    "lermina",
    "lerna",
    "precob",
    "precos",
    "preeob",
    "oxprelol",
    "leminate",
    "cahus",
    "gabun",
}

FREQUENCY_MAP = {
    "OD": "once daily",
    "QD": "once daily",
    "BD": "twice daily",
    "BID": "twice daily",
    "TDS": "three times daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "HS": "at bedtime",
    "SOS": "as needed",
    "STAT": "immediately",
    "1-0-1": "morning and night",
    "1-1-1": "three times daily",
    "0-1-0": "afternoon only",
    "0-0-1": "night only",
    "1x12": "twice daily",
    "1 x 12": "twice daily",
    "1x1": "once daily",
    "2x1": "twice daily",
}

_SYSTEM_PROMPT = """
You are an expert medical prescription reader.

Read the uploaded prescription image carefully and extract prescription information.

STRICT RULES:
1. Return ONLY valid JSON.
2. Do NOT add markdown fences.
3. Do NOT add explanations before or after JSON.
4. If medicine name is unclear but dosage/frequency line exists, still include the medicine object and set "name": "Unknown medicine".
5. Only include doctor_notes if clearly visible in prescription. Otherwise doctor_notes must be empty string.
6. Do not invent extra advice or notes.
7. If a field is not visible, use "Not specified".
8. Prefer common medicine spellings when handwriting is slightly unclear.
9. If frequency is abbreviated like TDS/BD/OD, preserve it in output if visible.
10. If medicine name is unreadable, do not hallucinate a random medicine.

Return JSON EXACTLY in this structure:

{
  "medicines": [
    {
      "name": "Medicine name or Unknown medicine",
      "dosage": "Dosage if visible, otherwise Not specified",
      "frequency": "Frequency if visible, otherwise Not specified",
      "duration": "Duration if visible, otherwise Not specified",
      "confidence": "High"
    }
  ],
  "doctor_notes": "",
  "unreadable_text_present": false
}
"""


def _clean_text(value: Any, default: str = "Not specified") -> str:
    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned.strip(".,;:-/\\")
    return cleaned or default


def _clean_medicine_name(name: str) -> str:
    cleaned = _clean_text(name, default="Unknown medicine")
    lower_name = cleaned.lower()
    if lower_name in {"unknown medicine", "not specified", "unknown", "unreadable", "na", "n/a"}:
        return "Unknown medicine"

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(".,;:/\\")
    if not cleaned:
        return "Unknown medicine"

    if cleaned.isupper() or cleaned.islower():
        return cleaned.title()

    return cleaned


def _normalize_frequency(freq: Any) -> str:
    cleaned = _clean_text(freq)
    if cleaned == "Not specified":
        return cleaned

    candidate = cleaned.upper()
    if candidate in FREQUENCY_MAP:
        return FREQUENCY_MAP[candidate]

    if "3-4-6" in cleaned:
        return "three times daily"

    if candidate == "TDS/BD":
        return "two to three times daily"

    normalized = re.sub(r"[^a-z0-9/\- ]+", " ", cleaned.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if "3x" in normalized or "3 times" in normalized:
        return "three times daily"
    if "2x" in normalized or "2 times" in normalized or "twice" in normalized:
        return "twice daily"
    if "1x" in normalized or "once" in normalized:
        return "once daily"
    if "noct" in normalized or "night" in normalized or "nacht" in normalized:
        return "at night"

    return cleaned


def _normalize_duration(duration: Any) -> str:
    cleaned = _clean_text(duration)
    if cleaned == "Not specified":
        return cleaned

    cleaned = re.sub(r"\s+", " ", cleaned).strip(".,;:")
    return cleaned or "Not specified"


def _normalize_dosage(dosage: Any) -> str:
    cleaned = _clean_text(dosage)
    if cleaned == "Not specified":
        return cleaned

    cleaned = re.sub(r"\b(tabs?|caps?|tablets?|capsules?)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=\d)\s*(mg|ml|mcg|g)\b", r" \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^0-9a-zA-Z.\-+/() ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(".,;:/")
    cleaned = cleaned.replace(" ule", " capsule")
    cleaned = cleaned.replace("cap ule", "capsule")
    return cleaned or "Not specified"


def _normalize_confidence(value: Any) -> str:
    if value is None:
        return "Medium"

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return "Medium"

        lower_value = raw.lower()
        if lower_value in {"high", "medium", "low"}:
            return lower_value.title()

        try:
            numeric = float(raw)
        except ValueError:
            return "Low"
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        return "Low"

    if numeric >= 0.85:
        return "High"
    if numeric >= 0.65:
        return "Medium"
    return "Low"


def _is_suspicious_medicine_name(name: str) -> bool:
    if not name:
        return True

    lower_name = name.strip().lower()
    if lower_name in {"unknown medicine", "not specified", "unreadable", "na", "n/a", ""}:
        return True
    if lower_name in SUSPICIOUS_MEDICINE_TOKENS:
        return True
    if len(lower_name) <= 2:
        return True
    if re.search(r"\d", lower_name) and any(
        token in lower_name for token in ["mg", "ml", "capsule", "tablet", "tab", "cap"]
    ):
        return True
    if re.fullmatch(r"[a-zA-Z]{1,3}", lower_name):
        return True
    return False


def _apply_alias_map(name: str) -> str:
    cleaned = _clean_medicine_name(name)
    alias = MEDICINE_ALIAS_MAP.get(cleaned.lower().strip())
    return alias if alias else cleaned


def _fuzzy_match_medicine(name: str) -> Tuple[str, bool]:
    """Return a corrected medicine name and whether it matched a known medicine."""
    cleaned = _clean_medicine_name(name)
    corrected = _apply_alias_map(cleaned)

    lowered = corrected.lower()
    if lowered in KNOWN_MEDICINE_SET:
        return COMMON_MEDICINES_BY_LOWER[lowered], True

    matches = difflib.get_close_matches(corrected, COMMON_MEDICINES, n=1, cutoff=0.82)
    if matches:
        return matches[0], True

    return corrected, False


def _should_keep_medicine(med: Dict[str, str]) -> bool:
    """Return True when the medicine object contains meaningful prescription information."""
    name = med.get("name", "").strip()
    dosage = med.get("dosage", "").strip()
    frequency = med.get("frequency", "").strip()
    duration = med.get("duration", "").strip()

    if name and name not in {"Unknown medicine", "Unreadable medicine name"}:
        return True
    return dosage != "Not specified" or frequency != "Not specified" or duration != "Not specified"


def _postprocess_unknown_name(
    corrected_name: str,
    original_name: str,
    dosage: str,
    frequency: str,
    duration: str,
    confidence: str,
) -> Tuple[str, str]:
    """Convert unmatched medicine candidates into a safe manual-verification or unknown result."""
    if _is_suspicious_medicine_name(original_name):
        if dosage != "Not specified" or frequency != "Not specified" or duration != "Not specified":
            return "Unknown medicine", "Low"
        return "Unreadable medicine name", "Low"

    if corrected_name.lower() not in KNOWN_MEDICINE_SET:
        if dosage != "Not specified" or frequency != "Not specified" or duration != "Not specified":
            if original_name.strip().lower() not in {"unknown medicine", "not specified", "", "unreadable"}:
                return f"Needs manual verification ({original_name})", "Low"
        return "Unknown medicine", "Low"

    return corrected_name, confidence


def _normalize_medicines(medicines: Any) -> List[Dict[str, str]]:
    """Normalize medicine output returned from Groq Vision."""
    if not isinstance(medicines, list):
        return []

    normalized: List[Dict[str, str]] = []
    seen_manual_verification_keys = set()

    for entry in medicines:
        if not isinstance(entry, dict):
            continue

        raw_name = _clean_text(entry.get("name"), default="Unknown medicine")
        raw_name_for_matching = OCR_CORRECTION_MAP.get(raw_name.lower().strip(), raw_name)

        dosage = _normalize_dosage(entry.get("dosage"))
        frequency = _normalize_frequency(entry.get("frequency"))
        duration = _normalize_duration(entry.get("duration"))
        confidence = _normalize_confidence(entry.get("confidence"))

        corrected_name, matched = _fuzzy_match_medicine(raw_name_for_matching)
        if corrected_name in {"", None, "Not specified"}:
            corrected_name = "Unknown medicine"

        if corrected_name.lower() in KNOWN_MEDICINE_SET:
            matched = True

        if corrected_name.lower() not in KNOWN_MEDICINE_SET and (
            not matched or _is_suspicious_medicine_name(raw_name_for_matching)
        ):
            corrected_name, confidence = _postprocess_unknown_name(
                corrected_name=corrected_name,
                original_name=raw_name,
                dosage=dosage,
                frequency=frequency,
                duration=duration,
                confidence=confidence,
            )

        has_details = (
            dosage != "Not specified"
            or frequency != "Not specified"
            or duration != "Not specified"
        )

        if corrected_name in {"Unknown medicine", "Unreadable medicine name"}:
            if (
                has_details
                and raw_name not in {"Unknown medicine", "Not specified", ""}
                and not _is_suspicious_medicine_name(raw_name)
            ):
                corrected_name = f"Needs manual verification ({raw_name})"
            elif corrected_name == "Unreadable medicine name":
                corrected_name = "Unknown medicine"
            confidence = "Low"

        if corrected_name.startswith("Needs manual verification") and confidence == "High":
            confidence = "Low"

        medicine = {
            "name": corrected_name,
            "raw_name": raw_name,
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "confidence": confidence,
        }

        if not _should_keep_medicine(medicine):
            continue

        if corrected_name.startswith("Needs manual verification"):
            manual_key = (
                raw_name.lower(),
                dosage.lower(),
                frequency.lower(),
                duration.lower(),
            )
            if manual_key in seen_manual_verification_keys:
                continue
            seen_manual_verification_keys.add(manual_key)

        normalized.append(medicine)

    deduped: List[Dict[str, str]] = []
    seen_keys = set()
    for medicine in normalized:
        key = (
            medicine["name"].lower(),
            medicine["dosage"].lower(),
            medicine["frequency"].lower(),
            medicine["duration"].lower(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(medicine)

    final: List[Dict[str, str]] = []
    seen_names = set()
    for medicine in deduped:
        name_key = medicine["name"].strip().lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        final.append(medicine)

    def sort_key(item: Dict[str, str]) -> Tuple[int, str]:
        name_key = item["name"].lower()
        if name_key.startswith("needs manual verification"):
            return 2, name_key
        if name_key == "unknown medicine":
            return 3, name_key
        if name_key == "unreadable medicine name":
            return 4, name_key
        return 1, name_key

    final.sort(key=sort_key)

    cleaned_output: List[Dict[str, str]] = []
    for medicine in final:
        cleaned_output.append(
            {
                "name": _clean_text(medicine.get("name"), default="Unknown medicine"),
                "raw_name": _clean_text(medicine.get("raw_name"), default="Unknown medicine"),
                "dosage": _clean_text(medicine.get("dosage"), default="Not specified"),
                "frequency": _clean_text(medicine.get("frequency"), default="Not specified"),
                "duration": _clean_text(medicine.get("duration"), default="Not specified"),
                "confidence": _normalize_confidence(medicine.get("confidence")),
            }
        )

    return cleaned_output


def _extract_json_block(content: str) -> str:
    """Extract a JSON object from raw model text safely."""
    if not content:
        return ""

    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return text[index : index + end]

    return text


def parse_prescription_image(base64_image: str) -> Dict[str, Any]:
    """Parse a prescription image using Groq Vision."""
    if not _GROQ_API_KEY:
        logger.warning("Groq API key is not configured.")
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "raw_extracted_text": None,
            "error": "Groq API key not configured.",
        }

    raw_content = None
    try:
        from groq import Groq

        client = Groq(api_key=_GROQ_API_KEY)
        image_url = (
            base64_image
            if base64_image.startswith("data:image")
            else f"data:image/jpeg;base64,{base64_image}"
        )

        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=1400,
        )

        raw_content = getattr(getattr(response.choices[0], "message", None), "content", None)
        if raw_content is None:
            raise ValueError("Groq response content is missing.")

        raw_content = str(raw_content).strip()
        logger.info("Received raw Groq prescription response.")

        cleaned_content = _extract_json_block(raw_content)
        if not cleaned_content:
            raise ValueError("No JSON object could be extracted from model response.")

        parsed = json.loads(cleaned_content)
        medicines = _normalize_medicines(parsed.get("medicines", []))

        doctor_notes = _clean_text(parsed.get("doctor_notes", ""), default="")
        if doctor_notes.lower() in {
            "not specified",
            "none",
            "nil",
            "na",
            "n/a",
            "drink 10 days water",
            "orally",
        }:
            doctor_notes = ""

        return {
            "medicines": medicines,
            "doctor_notes": doctor_notes,
            "unreadable_text_present": bool(parsed.get("unreadable_text_present", False)),
            "raw_extracted_text": raw_content,
            "error": None,
        }
    except json.JSONDecodeError as exc:
        logger.exception("Failed to decode JSON from Groq response.")
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "raw_extracted_text": raw_content,
            "error": f"Failed to parse model response into JSON. JSON error: {exc}",
        }
    except Exception as exc:
        logger.exception("Error parsing prescription image.")
        return {
            "medicines": [],
            "doctor_notes": "",
            "unreadable_text_present": False,
            "raw_extracted_text": raw_content,
            "error": f"Error parsing prescription: {exc}",
        }
