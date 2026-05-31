"""
scripts/clean_disease_data.py
──────────────────────────────
Cleans the disease dataset. Run this AFTER inspect_datasets.py.

CONFIRMED FORMAT (from inspection):
  - File: Final_Augmented_dataset_Diseases_and_Symptoms.csv
  - 246,945 rows  |  378 columns
  - Column 1: 'diseases'  (string — 773 unique disease names)
  - Columns 2–378: 377 symptom columns (int64, values 0 or 1 only)
  - Duplicates: 57,298 rows
  - No missing values in 'diseases'

Strategy:
  1. Drop duplicate rows
  2. For each row, collect column names where value == 1
  3. Normalize symptom names (underscores → spaces, strip whitespace)
  4. Skip rows with fewer than 1 valid symptom
  5. Build a 'content' text field for embedding
  6. Deduplicate on (disease, symptom_set) level
  7. Save to data/clean_disease_data.csv

Usage:
    python scripts\\clean_disease_data.py
"""

import sys
import hashlib
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH   = Path("/mnt/user-data/uploads/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
OUTPUT_PATH  = PROJECT_ROOT / "data" / "clean_disease_data.csv"

SEP = "=" * 65

def normalize_symptom(s: str) -> str:
    """'body_pain' → 'body pain', strip extra whitespace."""
    return s.replace("_", " ").strip().lower()

def main():
    print(f"\n{SEP}")
    print("  MedMitra — Disease Dataset Cleaner")
    print(SEP)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        # Fallback: look in data/ folder
        fallback = PROJECT_ROOT / "data" / "disease_dataset.csv"
        if fallback.exists():
            input_path = fallback
        else:
            print(f"\n❌ Input file not found at:\n   {INPUT_PATH}")
            print("   Copy your disease CSV into the data/ folder as disease_dataset.csv")
            sys.exit(1)
    else:
        input_path = INPUT_PATH

    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Raw rows : {len(df):,}")
    print(f"   Columns  : {len(df.columns)}")

    # ── 2. Identify columns ───────────────────────────────────────────────────
    DISEASE_COL = "diseases"
    if DISEASE_COL not in df.columns:
        # Try to find it
        candidates = [c for c in df.columns
                      if c.lower() in ("disease","diseases","condition","diagnosis")]
        if not candidates:
            print("❌ Cannot find disease name column. Expected 'diseases'.")
            print(f"   Columns found: {list(df.columns[:10])}")
            sys.exit(1)
        DISEASE_COL = candidates[0]
        print(f"   Found disease column as: '{DISEASE_COL}'")

    symptom_cols = [c for c in df.columns if c != DISEASE_COL]
    print(f"   Symptom columns: {len(symptom_cols)}")

    # ── 3. Drop full duplicate rows ───────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"\n🗑  Removed {before_dedup - len(df):,} exact duplicate rows")
    print(f"   Remaining: {len(df):,}")

    # ── 4. Convert binary matrix → symptom lists ──────────────────────────────
    print("\n⚙️  Converting binary columns to symptom lists …")

    records = []
    skipped_no_symptoms = 0

    for idx, row in df.iterrows():
        disease = str(row[DISEASE_COL]).strip()
        if not disease or disease.lower() in ("nan", "none", ""):
            continue

        # Collect symptoms where value == 1
        symptoms = []
        for col in symptom_cols:
            val = row[col]
            # Accept int 1, float 1.0, string '1'
            if val == 1 or val == 1.0 or str(val).strip() == "1":
                sym = normalize_symptom(col)
                # Filter out any junk that slipped through
                if sym and sym not in ("0", "1", "nan", "none", "false", "true"):
                    symptoms.append(sym)

        if not symptoms:
            skipped_no_symptoms += 1
            continue

        # Build a fingerprint to deduplicate on (disease, same symptom set)
        symptom_set = tuple(sorted(symptoms))
        fp = hashlib.md5((disease.lower() + "|" + ",".join(symptom_set)).encode()).hexdigest()

        records.append({
            "_fp":     fp,
            "disease": disease,
            "symptoms": ", ".join(symptoms),
            "content": f"Disease: {disease}. Common symptoms: {', '.join(symptoms)}.",
        })

    print(f"   Skipped (no symptoms): {skipped_no_symptoms:,}")

    # ── 5. Deduplicate on fingerprint ─────────────────────────────────────────
    clean_df = pd.DataFrame(records)
    before = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["_fp"]).drop(columns=["_fp"])
    print(f"   Removed {before - len(clean_df):,} duplicate (disease, symptoms) combos")

    # ── 6. Add metadata columns ───────────────────────────────────────────────
    clean_df.insert(0, "id", [f"dis_{i:06d}" for i in range(len(clean_df))])
    clean_df["source_dataset"] = "Final_Augmented_dataset_Diseases_and_Symptoms"

    # Reorder
    clean_df = clean_df[["id", "disease", "symptoms", "content", "source_dataset"]]

    # ── 7. Validate ───────────────────────────────────────────────────────────
    print(f"\n✅ Validation:")
    print(f"   Final rows   : {len(clean_df):,}")
    print(f"   Unique diseases: {clean_df['disease'].nunique():,}")
    empty_content = (clean_df["content"].str.strip() == "").sum()
    print(f"   Empty content rows: {empty_content}")
    if empty_content > 0:
        print("   ⚠️  Warning: some content rows are empty — investigate before indexing")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved → {OUTPUT_PATH}")

    # ── 9. Preview ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Sample cleaned rows (10):")
    print("─"*65)
    pd.set_option("display.max_colwidth", 90)
    print(clean_df[["id","disease","symptoms","content"]].sample(
        min(10, len(clean_df)), random_state=42
    ).to_string(index=False))

    print(f"\n{SEP}")
    print("  DONE. Next step: python scripts\\clean_medicine_data.py")
    print(SEP)

if __name__ == "__main__":
    main()
