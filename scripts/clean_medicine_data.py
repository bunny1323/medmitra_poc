"""
scripts/clean_medicine_data.py
────────────────────────────────
Cleans the medicine dataset. Run AFTER clean_disease_data.py.

CONFIRMED FORMAT (from inspection):
  - File: medicine_dataset.csv
  - 50,000 rows  |  7 columns
  - Columns: Name, Category, Dosage Form, Strength, Manufacturer,
             Indication, Classification
  - 0 duplicates, 0 missing values
  - 64 unique medicine names (SYNTHETIC dataset — demo only)
  - 8 unique Indication values  (Pain, Fever, Infection, etc.)
  - 8 unique Category values    (Antibiotic, Antiviral, etc.)

⚠️  IMPORTANT NOTE FOR THE TEAM:
  This medicine dataset is synthetic. Names like 'Acetocillin' and
  'Ibuprocillin' do not exist. It is suitable for prototype demos only.
  For production, replace with a real pharmacopoeia dataset
  (e.g., OpenFDA, WHO Essential Medicines List).

Strategy:
  1. Rename columns to standard names
  2. Normalize text (strip, lowercase where appropriate)
  3. Build a 'content' field for embedding
  4. Deduplicate on (name, indication, category)
  5. Save to data/clean_medicine_data.csv

Usage:
    python scripts\\clean_medicine_data.py
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH   = Path("/mnt/user-data/uploads/medicine_dataset.csv")
OUTPUT_PATH  = PROJECT_ROOT / "data" / "clean_medicine_data.csv"

SEP = "=" * 65

def main():
    print(f"\n{SEP}")
    print("  MedMitra — Medicine Dataset Cleaner")
    print(SEP)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        fallback = PROJECT_ROOT / "data" / "medicine_dataset.csv"
        if fallback.exists():
            input_path = fallback
        else:
            print(f"\n❌ Input file not found: {INPUT_PATH}")
            sys.exit(1)
    else:
        input_path = INPUT_PATH

    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Raw rows : {len(df):,}")
    print(f"   Columns  : {list(df.columns)}")

    # ── 2. Validate expected columns exist ────────────────────────────────────
    expected = {"Name", "Category", "Dosage Form", "Strength",
                "Manufacturer", "Indication", "Classification"}
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        print(f"\n⚠️  Missing expected columns: {missing}")
        print("   Proceeding with available columns …")

    # ── 3. Rename to standard internal names ──────────────────────────────────
    col_map = {
        "Name":           "medicine",
        "Category":       "category",
        "Dosage Form":    "dosage_form",
        "Strength":       "strength",
        "Manufacturer":   "manufacturer",
        "Indication":     "indication",
        "Classification": "classification",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── 4. Drop rows with no medicine name ────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["medicine"])
    df = df[df["medicine"].str.strip().str.lower().notna()]
    df = df[~df["medicine"].str.strip().str.lower().isin(["nan","none","","0","1"])]
    print(f"\n🗑  Dropped {before - len(df):,} rows with no medicine name")

    # ── 5. Strip whitespace on all string columns ──────────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── 6. Drop duplicates on (medicine, indication, category) ───────────────
    before = len(df)
    df = df.drop_duplicates(subset=["medicine", "indication", "category"])
    print(f"   Removed {before - len(df):,} duplicate (name, indication, category) rows")
    print(f"   Remaining: {len(df):,}")

    # ── 7. Build content string for embedding ─────────────────────────────────
    def build_content(row) -> str:
        parts = [f"Medicine: {row['medicine']}."]
        if pd.notna(row.get("category", None)):
            parts.append(f"Category: {row['category']}.")
        if pd.notna(row.get("indication", None)):
            parts.append(f"Indication: {row['indication']}.")
        if pd.notna(row.get("dosage_form", None)):
            parts.append(f"Dosage form: {row['dosage_form']}.")
        if pd.notna(row.get("classification", None)):
            parts.append(f"Classification: {row['classification']}.")
        if pd.notna(row.get("strength", None)):
            parts.append(f"Strength: {row['strength']}.")
        return " ".join(parts)

    df["content"] = df.apply(build_content, axis=1)

    # ── 8. Add metadata columns ───────────────────────────────────────────────
    df.insert(0, "id", [f"med_{i:06d}" for i in range(len(df))])
    df["source_dataset"] = "medicine_dataset_synthetic"

    # Select and order final columns
    final_cols = ["id","medicine","category","indication","dosage_form",
                  "strength","classification","manufacturer","content","source_dataset"]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # ── 9. Validate ───────────────────────────────────────────────────────────
    print(f"\n✅ Validation:")
    print(f"   Final rows       : {len(df):,}")
    print(f"   Unique medicines : {df['medicine'].nunique():,}")
    if "indication" in df.columns:
        print(f"   Unique indications: {df['indication'].nunique()}")
        print(f"   Indication breakdown:\n{df['indication'].value_counts().to_string()}")
    empty_content = (df["content"].str.strip() == "").sum()
    print(f"   Empty content rows: {empty_content}")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved → {OUTPUT_PATH}")

    # ── 11. Preview ───────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Sample cleaned rows (8):")
    print("─"*65)
    pd.set_option("display.max_colwidth", 90)
    print(df[["id","medicine","category","indication","content"]].sample(
        min(8, len(df)), random_state=42
    ).to_string(index=False))

    print(f"\n⚠️  NOTE: This dataset is SYNTHETIC (demo only).")
    print("   Replace with real pharmacopoeia data before any clinical use.")

    print(f"\n{SEP}")
    print("  DONE. Next step: python scripts\\create_prototype_data.py")
    print(SEP)

if __name__ == "__main__":
    main()
