"""
app/services/prescription_service.py
────────────────────────────────────
Handles parsing of prescription images using Groq Vision Model and fuzzy matching.
"""

import os
import json
import difflib
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# A simple dictionary of common medicine names for fuzzy matching.
# In a real app, this could be loaded from a database or a comprehensive JSON file.
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
You must return ONLY a valid JSON object with EXACTLY three keys: "medicines", "doctor_notes", and "unreadable_text_present".
The "medicines" key must be a list of objects, each containing:
- "name": The name of the medicine.
- "dosage": The dosage (e.g., 500mg, 10ml). If not found, write "Not specified".
- "frequency": How often to take it (e.g., Twice a day). If not found, write "Not specified".
- "duration": How long to take it (e.g., 5 days). If not found, write "Not specified".
- "confidence": "High", "Medium", or "Low" indicating how confident you are in reading this specific medicine's details.

The "doctor_notes" key should be a string containing any other instructions found on the prescription.
The "unreadable_text_present" key should be a boolean (true if significant parts were completely illegible, false otherwise).

Do NOT wrap the JSON in markdown blocks like ```json. Output ONLY the raw JSON string.
"""

def _fuzzy_match_medicine(name: str) -> str:
    """Attempts to match a potentially misspelled medicine name to a known list."""
    if not name or name.lower() == "not specified":
        return name
        
    matches = difflib.get_close_matches(name.title(), COMMON_MEDICINES, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return name

def parse_prescription_image(base64_image: str) -> Dict[str, Any]:
    if not _GROQ_API_KEY:
        return {"error": "Groq API key not configured."}

    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_API_KEY)
        
        # Determine the mime type from the base64 prefix if needed, 
        # but Groq Vision expects data URL format: "data:image/jpeg;base64,..."
        # If the prefix isn't there, we assume it's jpeg for the API format, but it's better to ensure it.
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
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        parsed = json.loads(content.strip())
        
        # Apply fuzzy matching
        if "medicines" in parsed and isinstance(parsed["medicines"], list):
            for med in parsed["medicines"]:
                original_name = med.get("name", "")
                corrected_name = _fuzzy_match_medicine(original_name)
                med["name"] = corrected_name

        return {
            "medicines": parsed.get("medicines", []),
            "doctor_notes": parsed.get("doctor_notes", ""),
            "unreadable_text_present": parsed.get("unreadable_text_present", False),
            "error": None
        }

    except json.JSONDecodeError:
        return {"error": "Failed to parse model response into JSON. Response might be malformed."}
    except Exception as e:
        return {"error": f"Error parsing prescription: {str(e)}"}
