#!/usr/bin/env python3
"""
MedMitra Data Cleaning Pipeline
================================
Cleans and normalizes Kaggle datasets into structured JSON files for the
MedMitra prototype. All records carry review_status="prototype_unverified"
to make clear that Kaggle data is NOT clinically verified.

Usage:
    python -m app.scripts.clean_datasets

Expected raw CSV files in app/data/raw/:
    - disease_symptom_description_dataset.csv   (itachi9604/disease-symptom-description-dataset)
    - symptom2disease.csv                        (niyarrbarman/symptom2disease)
    - 1000_drugs_and_side_effects.csv            (palakjain9/1000-drugs-and-side-effects)
    - indian_medicine_data.csv                   (mohneesh7/indian-medicine-data)

Outputs:
    - app/data/processed/diseases.json
    - app/data/processed/medicines.json
    - app/data/processed/symptom_queries.json
"""

import os
import re
import sys
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap path so the script works both as module and as standalone script
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "app" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "app" / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("clean_datasets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Lowercase, remove punctuation, collapse spaces, strip leading/trailing."""
    if not text:
        return ""
    # Replace common punctuation with space
    text_clean = re.sub(r"[.,;!?()\[\]\"'_]", " ", text)
    # Lowercase and collapse spaces
    return re.sub(r"\s+", " ", text_clean.strip().lower())



def _clean_list_field(value: Any) -> List[str]:
    """Normalise a field that may be a string, list, or pipe-delimited value."""
    if not value or (isinstance(value, float)):
        return []
    if isinstance(value, list):
        return [_slug(str(v)) for v in value if str(v).strip()]
    # Pipe or comma separated string
    raw = str(value).strip()
    if "|" in raw:
        parts = raw.split("|")
    elif ";" in raw:
        parts = raw.split(";")
    else:
        parts = [raw]
    return [_slug(p) for p in parts if p.strip()]


def _safe_str(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float)):
        return default
    return str(value).strip()


def _build_disease_search_text(record: Dict) -> str:
    parts = [
        record.get("condition_name", ""),
        " ".join(record.get("symptoms", [])),
        record.get("description", ""),
        " ".join(record.get("precautions", [])),
    ]
    return _slug(" ".join(p for p in parts if p))


def _build_medicine_search_text(record: Dict) -> str:
    parts = [
        record.get("medicine_name", ""),
        record.get("generic_name", ""),
        " ".join(record.get("aliases", [])),
        record.get("category", ""),
        " ".join(record.get("uses", [])),
        record.get("mechanism_of_action", ""),
        record.get("salt_composition", ""),
        " ".join(record.get("side_effects", [])),
    ]
    return _slug(" ".join(p for p in parts if p))


def _report(label: str, orig: int, empty: int, dupes: int, malformed: int, final: int):
    skipped = orig - final
    log.info(
        f"[{label}] Original: {orig} | Empty: {empty} | Duplicates: {dupes} | "
        f"Malformed: {malformed} | Skipped: {skipped} | Final: {final}"
    )


# ---------------------------------------------------------------------------
# Dataset 1 â€” Disease Symptom Description (itachi9604)
# ---------------------------------------------------------------------------

def clean_disease_dataset(csv_path: Path) -> List[Dict]:
    """
    Reads itachi9604/disease-symptom-description-dataset.
    Expected columns (may vary): Disease, Symptom_1..Symptom_17, Description,
    Precaution_1..Precaution_4, Symptom_severity (separate CSV sometimes).
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required. Install it: pip install pandas")
        sys.exit(1)

    if not csv_path.exists():
        log.warning(f"Disease CSV not found: {csv_path}. Skipping disease cleaning.")
        return []

    df = pd.read_csv(csv_path)
    orig_count = len(df)
    log.info(f"Disease dataset loaded: {orig_count} rows, columns: {list(df.columns)}")

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Identify disease-name column
    name_col = next((c for c in df.columns if "disease" in c), None)
    if not name_col:
        log.error("No 'disease' column found in dataset.")
        return []

    # Collect symptom columns
    symptom_cols = sorted([c for c in df.columns if "symptom" in c and c != name_col])
    # Collect precaution columns
    precaution_cols = sorted([c for c in df.columns if "precaution" in c])
    # Description column
    desc_col = next((c for c in df.columns if "description" in c), None)

    # Remove completely empty rows
    df = df.dropna(how="all")
    empty_count = orig_count - len(df)

    # Remove rows with no disease name
    df = df[df[name_col].notna() & (df[name_col].str.strip() != "")]
    malformed_count = orig_count - empty_count - len(df)

    # Remove duplicates based on disease name
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[name_col])
    dupe_count = before_dedup - len(df)

    records = []
    seen_names = set()

    for _, row in df.iterrows():
        disease_name = _safe_str(row.get(name_col)).title()
        if not disease_name or disease_name.lower() in seen_names:
            continue
        seen_names.add(disease_name.lower())

        # Collect symptoms
        symptoms = []
        for sc in symptom_cols:
            val = _safe_str(row.get(sc, ""))
            if val and val.lower() not in ("", "nan", "none"):
                s = _slug(val.replace("_", " "))
                if s and s not in symptoms:
                    symptoms.append(s)

        # Collect precautions
        precautions = []
        for pc in precaution_cols:
            val = _safe_str(row.get(pc, ""))
            if val and val.lower() not in ("", "nan", "none"):
                p = _slug(val.replace("_", " "))
                if p and p not in precautions:
                    precautions.append(p)

        description = _safe_str(row.get(desc_col, "")) if desc_col else ""
        if not description:
            description = f"General information about {disease_name}."

        record = {
            "record_id": str(uuid.uuid4()),
            "record_type": "disease",
            "condition_name": disease_name,
            "symptoms": symptoms,
            "description": description,
            "precautions": precautions,
            "symptom_weights": {},
            "search_text": "",  # filled below
            "source_name": "Disease Symptom Prediction",
            "source_type": "kaggle_dataset",
            "dataset_slug": "itachi9604/disease-symptom-description-dataset",
            "review_status": "prototype_unverified",
        }
        record["search_text"] = _build_disease_search_text(record)

        # Skip records with empty search text
        if not record["search_text"].strip():
            malformed_count += 1
            continue

        records.append(record)

    _report("Diseases", orig_count, empty_count, dupe_count, malformed_count, len(records))
    return records


# ---------------------------------------------------------------------------
# Dataset 2 â€” Symptom2Disease (niyarrbarman) â€” for evaluation only
# ---------------------------------------------------------------------------

def clean_symptom_queries_dataset(csv_path: Path) -> List[Dict]:
    """
    Reads niyarrbarman/symptom2disease dataset.
    Expected columns: label (disease), text (symptom description).
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required.")
        sys.exit(1)

    if not csv_path.exists():
        log.warning(f"Symptom queries CSV not found: {csv_path}. Skipping.")
        return []

    df = pd.read_csv(csv_path)
    orig_count = len(df)
    log.info(f"Symptom queries dataset loaded: {orig_count} rows")

    df.columns = [c.strip().lower() for c in df.columns]
    label_col = next((c for c in df.columns if "label" in c or "disease" in c), None)
    text_col = next((c for c in df.columns if "text" in c or "symptom" in c), None)

    if not label_col or not text_col:
        log.warning("Could not identify label/text columns in symptom2disease dataset.")
        return []

    df = df.dropna(subset=[label_col, text_col])
    empty_count = orig_count - len(df)
    df = df[df[text_col].str.strip() != ""]
    malformed_count = orig_count - empty_count - len(df)
    before_dedup = len(df)
    df = df.drop_duplicates()
    dupe_count = before_dedup - len(df)

    records = []
    for _, row in df.iterrows():
        records.append({
            "record_id": str(uuid.uuid4()),
            "record_type": "symptom_query",
            "disease_label": _safe_str(row[label_col]).title(),
            "query_text": _safe_str(row[text_col]),
            "source_name": "Symptom2Disease",
            "source_type": "kaggle_dataset",
            "dataset_slug": "niyarrbarman/symptom2disease",
            "review_status": "prototype_unverified",
        })

    _report("SymptomQueries", orig_count, empty_count, dupe_count, malformed_count, len(records))
    return records


# ---------------------------------------------------------------------------
# Dataset 3 â€” 1000 Drugs and Side Effects (palakjain9)
# ---------------------------------------------------------------------------

def clean_drugs_dataset(csv_path: Path) -> Dict[str, Dict]:
    """
    Reads the uploaded drug dataset. Supports two formats:
      1. Patient-record style: Patient_ID, Age, Gender, Condition, Drug_Name,
         Dosage_mg, Treatment_Duration_days, Side_Effects, Improvement_Score
         â†’ groups by Drug_Name, aggregates Condition â†’ uses, Side_Effects â†’ side_effects
      2. Catalogue style: Drug Name, Medical Condition, Side Effects, Generic Name, Drug Class, Brand Names
    Returns dict keyed by normalised drug name for later merging.
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required.")
        sys.exit(1)

    if not csv_path.exists():
        log.warning(f"Drugs CSV not found: {csv_path}. Skipping.")
        return {}

    df = pd.read_csv(csv_path)
    orig_count = len(df)
    log.info(f"Drugs dataset loaded: {orig_count} rows, columns: {list(df.columns)}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def _find_col(df, keywords):
        for kw in keywords:
            for c in df.columns:
                if kw in c:
                    return c
        return None

    name_col = _find_col(df, ["drug_name", "drug"])
    condition_col = _find_col(df, ["condition", "medical_condition", "use", "indication"])
    side_effects_col = _find_col(df, ["side_effect"])
    generic_col = _find_col(df, ["generic_name", "generic"])
    class_col = _find_col(df, ["drug_class", "class", "category"])
    brand_col = _find_col(df, ["brand"])

    if not name_col:
        log.error("No drug name column identified.")
        return {}

    df = df.dropna(subset=[name_col])
    df = df[df[name_col].str.strip() != ""]
    empty_count = orig_count - len(df)

    records = {}
    malformed_count = 0
    dupe_count = 0

    # --- Patient-record style: group by Drug_Name ---
    is_patient_style = "patient_id" in df.columns
    if is_patient_style:
        log.info("Patient-record style detected â€” grouping by drug name and aggregating fields...")
        ANTIBIOTIC_DRUGS = {
            "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
            "erythromycin", "levofloxacin", "metronidazole", "clindamycin",
            "cephalexin", "trimethoprim", "sulfamethoxazole"
        }
        grouped = df.groupby(name_col)
        dupe_count = orig_count - len(grouped)

        for drug_name, group in grouped:
            drug_name = str(drug_name).strip()
            if not drug_name:
                malformed_count += 1
                continue

            conditions = sorted(set(
                _slug(str(v)) for v in group[condition_col].dropna()
                if str(v).strip().lower() not in ("", "nan")
            )) if condition_col else []

            side_effects = sorted(set(
                _slug(str(v)) for v in group[side_effects_col].dropna()
                if str(v).strip().lower() not in ("", "nan")
            )) if side_effects_col else []

            # Infer category
            dn_lower = drug_name.lower().split()[0]
            category = ""
            if dn_lower in ANTIBIOTIC_DRUGS:
                category = "antibiotic"
            elif any("diabetes" in c for c in conditions):
                category = "antidiabetic"
            elif any(kw in c for c in conditions for kw in ("hypertension", "blood pressure")):
                category = "antihypertensive"
            elif any("depression" in c for c in conditions):
                category = "antidepressant"
            elif any("pain" in c for c in conditions):
                category = "analgesic"
            elif any("infection" in c for c in conditions):
                category = "anti-infective"

            warnings = []
            if "antibiotic" in category or dn_lower in ANTIBIOTIC_DRUGS:
                warnings.append(
                    "This is a prescription antibiotic. Do not use without a doctor's prescription. "
                    "Misuse of antibiotics contributes to antimicrobial resistance."
                )

            key = drug_name.lower().strip()
            records[key] = {
                "record_id": str(uuid.uuid4()),
                "record_type": "medicine",
                "medicine_name": drug_name.title(),
                "generic_name": drug_name.title(),
                "aliases": [],
                "category": category,
                "dosage_form": "",
                "mechanism_of_action": "",
                "uses": conditions,
                "side_effects": side_effects,
                "warnings": warnings,
                "manufacturer": "",
                "salt_composition": "",
                "search_text": "",
                "source_name": "1000 Drugs and Side Effects",
                "source_type": "kaggle_dataset",
                "dataset_slug": "palakjain9/1000-drugs-and-side-effects",
                "review_status": "prototype_unverified",

            }

    else:
        # --- Catalogue style ---
        before_dedup = len(df)
        df = df.drop_duplicates(subset=[name_col])
        dupe_count = before_dedup - len(df)

        for _, row in df.iterrows():
            drug_name = _safe_str(row.get(name_col)).strip()
            if not drug_name:
                malformed_count += 1
                continue

            generic_name = _safe_str(row.get(generic_col, drug_name)) if generic_col else drug_name
            category = _safe_str(row.get(class_col, "")) if class_col else ""
            uses = _clean_list_field(_safe_str(row.get(condition_col, "")) if condition_col else "")
            side_effects = _clean_list_field(_safe_str(row.get(side_effects_col, "")) if side_effects_col else "")
            aliases = _clean_list_field(_safe_str(row.get(brand_col, "")) if brand_col else "")

            warnings = []
            if any(kw in category.lower() for kw in ["antibiotic", "antimicrobial", "antibacterial"]):
                warnings.append(
                    "This is a prescription antibiotic. Do not use without a doctor's prescription. "
                    "Misuse of antibiotics contributes to antimicrobial resistance."
                )

            key = drug_name.lower().strip()
            records[key] = {
                "record_id": str(uuid.uuid4()),
                "record_type": "medicine",
                "medicine_name": drug_name.title(),
                "generic_name": generic_name.title() if generic_name else drug_name.title(),
                "aliases": [a.title() for a in aliases if a],
                "category": _slug(category),
                "dosage_form": "",
                "mechanism_of_action": "",
                "uses": uses,
                "side_effects": side_effects,
                "warnings": warnings,
                "manufacturer": "",
                "salt_composition": "",
                "search_text": "",
                "source_name": "1000 Drugs and Side Effects",
                "source_type": "kaggle_dataset",
                "dataset_slug": "palakjain9/1000-drugs-and-side-effects",
                "review_status": "prototype_unverified",
            }

    _report("Drugs(primary)", orig_count, empty_count, dupe_count, malformed_count, len(records))
    return records


# ---------------------------------------------------------------------------
# Dataset 4 â€” Indian Medicine Data (mohneesh7) â€” secondary catalogue
# ---------------------------------------------------------------------------

def enrich_with_indian_catalogue(records: Dict[str, Dict], csv_path: Path) -> Dict[str, Dict]:
    """
    Reads mohneesh7/indian-medicine-data and merges Indian product names,
    salt composition and manufacturer into existing records where the
    medicine name or salt composition matches reliably.
    """
    try:
        import pandas as pd
    except ImportError:
        return records

    if not csv_path.exists():
        log.warning(f"Indian medicine CSV not found: {csv_path}. Skipping merge.")
        return records

    df = pd.read_csv(csv_path)
    log.info(f"Indian medicine catalogue loaded: {len(df)} rows")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def _fc(keywords):
        for kw in keywords:
            for c in df.columns:
                if kw in c:
                    return c
        return None

    name_col = _fc(["product_name", "medicine_name", "name"])
    salt_col = _fc(["salt_composition", "composition", "ingredient"])
    manufacturer_col = _fc(["manufacturer", "company"])
    category_col = _fc(["sub_category", "category"])

    if not name_col:
        log.warning("Could not identify name column in Indian medicine dataset.")
        return records

    enriched = 0
    for _, row in df.iterrows():
        product_name = _safe_str(row.get(name_col, "")).lower().strip()
        salt = _safe_str(row.get(salt_col, "")) if salt_col else ""
        manufacturer = _safe_str(row.get(manufacturer_col, "")) if manufacturer_col else ""
        category = _safe_str(row.get(category_col, "")) if category_col else ""

        # Try to match by name
        matched_key = None
        if product_name in records:
            matched_key = product_name
        else:
            # Try partial match on first word
            first_word = product_name.split()[0] if product_name.split() else ""
            if first_word and first_word in records:
                matched_key = first_word

        if matched_key:
            rec = records[matched_key]
            if salt and not rec["salt_composition"]:
                rec["salt_composition"] = _slug(salt)
            if manufacturer and not rec["manufacturer"]:
                rec["manufacturer"] = manufacturer.strip()
            if category and not rec["category"]:
                rec["category"] = _slug(category)
            enriched += 1
        else:
            # Create new record for Indian-only medicine
            new_key = product_name
            if new_key and new_key not in records:
                antibiotic_warn = []
                if "antibiotic" in category.lower():
                    antibiotic_warn = [
                        "This is a prescription antibiotic. Do not use without a doctor's prescription."
                    ]
                records[new_key] = {
                    "record_id": str(uuid.uuid4()),
                    "record_type": "medicine",
                    "medicine_name": product_name.title(),
                    "generic_name": _safe_str(row.get(salt_col, product_name)).title() if salt_col else product_name.title(),
                    "aliases": [],
                    "category": _slug(category),
                    "dosage_form": "",
                    "mechanism_of_action": "",
                    "uses": [],
                    "side_effects": [],
                    "warnings": antibiotic_warn,
                    "manufacturer": manufacturer.strip(),
                    "salt_composition": _slug(salt),
                    "search_text": "",
                    "source_name": "Indian Medicine Data",
                    "source_type": "kaggle_dataset",
                    "dataset_slug": "mohneesh7/indian-medicine-data",
                    "review_status": "prototype_unverified",
                }

    log.info(f"Indian catalogue: {enriched} existing records enriched, {len(records)} total after merge.")
    return records


# ---------------------------------------------------------------------------
# Finalise medicine records
# ---------------------------------------------------------------------------

def finalise_medicine_records(records: Dict[str, Dict]) -> List[Dict]:
    """Build search_text for each medicine record and return as a list."""
    result = []
    for rec in records.values():
        rec["search_text"] = _build_medicine_search_text(rec)
        if not rec["search_text"].strip():
            continue
        result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MedMitra Data Cleaning Pipeline â€” Starting")
    log.info("=" * 60)
    log.info(f"Raw directory  : {RAW_DIR}")
    log.info(f"Output directory: {PROCESSED_DIR}")

    # Check for CSV files
    disease_csv = RAW_DIR / "disease_symptom_description_dataset.csv"
    symptom_csv = RAW_DIR / "symptom2disease.csv"
    drugs_csv = RAW_DIR / "1000_drugs_and_side_effects.csv"
    indian_csv = RAW_DIR / "indian_medicine_data.csv"

    missing = [p for p in [disease_csv, symptom_csv, drugs_csv] if not p.exists()]
    if missing:
        log.warning(
            f"\nMissing raw CSV files:\n  " + "\n  ".join(str(m) for m in missing) +
            "\n\nDownload them using the Kaggle CLI:\n"
            "  kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p app/data/raw --unzip\n"
            "  kaggle datasets download -d niyarrbarman/symptom2disease -p app/data/raw --unzip\n"
            "  kaggle datasets download -d palakjain9/1000-drugs-and-side-effects -p app/data/raw --unzip\n"
            "  kaggle datasets download -d mohneesh7/indian-medicine-data -p app/data/raw --unzip\n"
            "\nRename files to match expected names if necessary."
        )

    # --- Clean Diseases ---
    disease_records = clean_disease_dataset(disease_csv)
    diseases_out = PROCESSED_DIR / "diseases.json"
    with open(diseases_out, "w", encoding="utf-8") as f:
        json.dump(disease_records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(disease_records)} disease records â†’ {diseases_out}")

    # --- Clean Symptom Queries ---
    query_records = clean_symptom_queries_dataset(symptom_csv)
    queries_out = PROCESSED_DIR / "symptom_queries.json"
    with open(queries_out, "w", encoding="utf-8") as f:
        json.dump(query_records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(query_records)} symptom query records â†’ {queries_out}")

    # --- Clean Medicines ---
    drug_records = clean_drugs_dataset(drugs_csv)
    drug_records = enrich_with_indian_catalogue(drug_records, indian_csv)
    medicine_records = finalise_medicine_records(drug_records)
    medicines_out = PROCESSED_DIR / "medicines.json"
    with open(medicines_out, "w", encoding="utf-8") as f:
        json.dump(medicine_records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(medicine_records)} medicine records â†’ {medicines_out}")

    log.info("=" * 60)
    log.info("Data cleaning complete.")
    log.info(
        f"Summary:\n"
        f"  Disease records    : {len(disease_records)}\n"
        f"  Medicine records   : {len(medicine_records)}\n"
        f"  Symptom queries    : {len(query_records)}\n"
        "  review_status      : prototype_unverified\n"
        "  âš  Kaggle data is NOT clinically verified."
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()

