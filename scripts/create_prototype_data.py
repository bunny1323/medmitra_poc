"""
scripts/create_prototype_data.py
──────────────────────────────────
Creates small prototype subsets from cleaned CSVs.
Run AFTER both cleaning scripts.

Disease strategy:
  - 246,945 raw → 189,647 after dedup
  - After cleaning, many rows are same disease with slightly
    different symptom combos (augmented dataset)
  - We want: 1 representative row per disease (most common symptom combo)
    → caps at ~773 diseases max
  - Target: 500 diverse disease records

Medicine strategy:
  - 64 unique medicine names × ~8 indications = ~512 combos
  - Already small enough; we take all unique (medicine, indication) combos

Usage:
    python scripts\\create_prototype_data.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SEP = "=" * 65


def create_disease_prototype():
    src = DATA_DIR / "clean_disease_data.csv"
    dst = DATA_DIR / "prototype_disease_data.csv"

    if not src.exists():
        print(f"❌ {src} not found. Run clean_disease_data.py first.")
        return False

    print(f"\n📂 Loading cleaned disease data …")
    df = pd.read_csv(src)
    print(f"   Total clean rows: {len(df):,}")
    print(f"   Unique diseases : {df['disease'].nunique():,}")

    # Strategy: for each disease, keep the row with the most symptoms
    # (richest representation for embedding)
    df["_sym_count"] = df["symptoms"].str.count(",") + 1
    df_best = (
        df.sort_values("_sym_count", ascending=False)
          .drop_duplicates(subset=["disease"])
          .drop(columns=["_sym_count"])
          .reset_index(drop=True)
    )

    print(f"   After 1-per-disease selection: {len(df_best):,}")

    # Also add up to 3 additional augmented variants per disease
    # to give the vector index more diversity for retrieval
    df["_sym_count"] = df["symptoms"].str.count(",") + 1
    df_extras = (
        df.sort_values("_sym_count", ascending=False)
          .groupby("disease")
          .head(4)   # up to 4 rows per disease
          .reset_index(drop=True)
    )
    df_extras = df_extras.drop(columns=["_sym_count"])

    # Combine, dedup
    combined = pd.concat([df_best, df_extras]).drop_duplicates(subset=["id"])
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Reassign clean IDs
    combined["id"] = [f"dis_{i:06d}" for i in range(len(combined))]

    TARGET = 500
    if len(combined) > TARGET:
        # Stratified: keep equal-ish reps across diseases
        prototype = (
            combined.groupby("disease", group_keys=False)
                    .apply(lambda g: g.head(max(1, TARGET // combined["disease"].nunique())))
        )
        prototype = prototype.head(TARGET).reset_index(drop=True)
    else:
        prototype = combined

    print(f"   Prototype rows  : {len(prototype):,}")
    print(f"   Prototype diseases: {prototype['disease'].nunique():,}")

    prototype.to_csv(dst, index=False)
    print(f"💾 Saved → {dst}")
    print(f"\n   Sample:")
    print(prototype[["disease","symptoms"]].head(5).to_string(index=False))
    return True


def create_medicine_prototype():
    src = DATA_DIR / "clean_medicine_data.csv"
    dst = DATA_DIR / "prototype_medicine_data.csv"

    if not src.exists():
        print(f"❌ {src} not found. Run clean_medicine_data.py first.")
        return False

    print(f"\n📂 Loading cleaned medicine data …")
    df = pd.read_csv(src)
    print(f"   Total clean rows: {len(df):,}")
    print(f"   Unique medicines: {df['medicine'].nunique():,}")

    # For the synthetic dataset: 64 names × 8 indications ≈ 512 combos
    # All are already deduplicated — just use them all (or cap at 500)
    prototype = df.copy()
    if len(prototype) > 500:
        prototype = (
            prototype.groupby("medicine", group_keys=False)
                     .apply(lambda g: g.head(
                         max(1, 500 // prototype["medicine"].nunique())
                     ))
                     .reset_index(drop=True)
        )
        prototype = prototype.head(500).reset_index(drop=True)

    prototype["id"] = [f"med_{i:06d}" for i in range(len(prototype))]

    print(f"   Prototype rows  : {len(prototype):,}")
    print(f"   Prototype medicines: {prototype['medicine'].nunique():,}")

    prototype.to_csv(dst, index=False)
    print(f"💾 Saved → {dst}")
    print(f"\n   Sample:")
    cols = [c for c in ["medicine","category","indication","content"] if c in prototype.columns]
    print(prototype[cols].head(5).to_string(index=False))
    return True


def main():
    print(f"\n{SEP}")
    print("  MedMitra — Create Prototype Subsets")
    print(SEP)

    ok_d = create_disease_prototype()
    ok_m = create_medicine_prototype()

    print(f"\n{SEP}")
    if ok_d and ok_m:
        print("  ✅ Both prototype files created.")
        print("  Next steps:")
        print("    python scripts\\build_disease_index.py --reset")
        print("    python scripts\\build_medicine_index.py --reset")
    else:
        print("  ⚠️  One or more prototype files could not be created.")
        print("  Run the cleaning scripts first.")
    print(SEP)


if __name__ == "__main__":
    main()
