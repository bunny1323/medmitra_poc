"""
scripts/inspect_datasets.py
────────────────────────────
Step 1: Inspect all uploaded CSV files WITHOUT assuming column names.

Run this FIRST before any cleaning or indexing.

Usage:
    python scripts/inspect_datasets.py
    (Windows: python scripts\\inspect_datasets.py)

What it does:
  - Detects file size, row count, column names
  - Prints the first 5 rows
  - Reports missing values and duplicates
  - Detects disease_dataset format (binary / text / single-list)
  - Identifies likely medicine and PubMedQA columns
  - Prints a validation report so you can confirm before proceeding
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

FILES = {
    "disease":  DATA_DIR / "disease_dataset.csv",
    "medicine": DATA_DIR / "medicine_dataset.csv",
    "pubmedqa": DATA_DIR / "pubmedqa.csv",
}

SEPARATOR = "=" * 70


def file_size_str(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size/1024:.1f} KB"
    return f"{size/1024**2:.2f} MB"


def inspect_file(label: str, path: Path) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  FILE: {label.upper()}  →  {path.name}")
    print(SEPARATOR)

    if not path.exists():
        print(f"  ❌ File not found: {path}")
        print(f"  Please copy {path.name} into the data/ folder.")
        return

    print(f"  Size   : {file_size_str(path)}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  ❌ Could not read file: {e}")
        return

    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"\n  Column names:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        print(f"    {col!r:<35} dtype={dtype:<10} unique={n_unique:<8} missing={n_missing}")

    print(f"\n  First 5 rows:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(df.head(5).to_string(index=False))

    n_dupes = df.duplicated().sum()
    print(f"\n  Duplicate rows: {n_dupes:,}")

    total_missing = df.isna().sum().sum()
    print(f"  Total missing values: {total_missing:,}")

    if label == "disease":
        _detect_disease_format(df)
    elif label == "medicine":
        _detect_medicine_columns(df)
    elif label == "pubmedqa":
        _detect_pubmedqa_columns(df)


def _detect_disease_format(df: pd.DataFrame) -> None:
    """
    Detect which of the four common disease CSV formats is used.
    """
    print("\n  ── Disease format detection ──")
    cols_lower = {c.lower(): c for c in df.columns}

    # Detect disease name column
    disease_col = None
    for candidate in ["disease", "disease_name", "condition", "diagnosis"]:
        if candidate in cols_lower:
            disease_col = cols_lower[candidate]
            break
    print(f"  Likely disease column   : {disease_col!r}")

    # Detect binary vs text symptom columns
    # Binary: mostly 0 and 1 values
    non_disease_cols = [c for c in df.columns if c != disease_col]

    binary_cols = []
    text_cols   = []

    for col in non_disease_cols:
        unique_vals = df[col].dropna().unique()
        # Binary detection: only values in {0, 1, True, False, '0', '1'}
        binary_set = {0, 1, "0", "1", True, False, 0.0, 1.0}
        if all(v in binary_set for v in unique_vals):
            binary_cols.append(col)
        else:
            text_cols.append(col)

    if binary_cols:
        print(f"  Format detected        : FORMAT A — Binary symptom columns")
        print(f"  Binary columns ({len(binary_cols)})   : {binary_cols[:10]}{'...' if len(binary_cols)>10 else ''}")
        print(f"  → Clean script will convert 1→symptom_name, 0→ignore")
    elif text_cols:
        # Check if it's a single symptom list or multiple text columns
        symptom_col_names = [c for c in text_cols if "symptom" in c.lower() or "sign" in c.lower()]
        if len(symptom_col_names) == 1 and df[symptom_col_names[0]].dtype == object:
            sample = df[symptom_col_names[0]].dropna().iloc[0] if len(df) > 0 else ""
            if "," in str(sample):
                print(f"  Format detected        : FORMAT C — Single symptom-list column")
                print(f"  Symptom list column    : {symptom_col_names[0]!r}")
            else:
                print(f"  Format detected        : FORMAT B — Text symptom columns (multiple)")
                print(f"  Text symptom cols      : {symptom_col_names}")
        elif len(text_cols) >= 2 and any("symptom" in c.lower() for c in text_cols):
            print(f"  Format detected        : FORMAT B — Text symptom columns (multiple)")
            print(f"  Text columns           : {text_cols[:5]}")
        else:
            print(f"  Format detected        : FORMAT D — Unknown (manual review needed)")
            print(f"  Non-disease columns    : {text_cols[:10]}")
    else:
        print(f"  ⚠  Could not detect symptom columns automatically.")

    print(f"\n  Sample unique diseases : {df[disease_col].dropna().unique()[:8].tolist() if disease_col else 'N/A'}")


def _detect_medicine_columns(df: pd.DataFrame) -> None:
    print("\n  ── Medicine column detection ──")
    cols_lower = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    name_col    = find(["medicine_name", "drug_name", "name", "drug", "medicine"])
    uses_col    = find(["uses", "use", "indication", "indications", "purpose"])
    warn_col    = find(["warnings", "warning", "caution", "contraindications"])
    effects_col = find(["side_effects", "adverse_effects", "side effects", "adverse"])

    print(f"  Medicine name column : {name_col!r}")
    print(f"  Uses column          : {uses_col!r}")
    print(f"  Warnings column      : {warn_col!r}")
    print(f"  Side effects column  : {effects_col!r}")

    if name_col and uses_col:
        print("  ✅ Minimum required columns found (name + uses)")
    else:
        print("  ⚠  Could not find required columns — manual review needed")


def _detect_pubmedqa_columns(df: pd.DataFrame) -> None:
    print("\n  ── PubMedQA column detection ──")
    cols_lower = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    q_col       = find(["question", "query", "q"])
    ctx_col     = find(["context", "abstract", "background", "text"])
    ans_col     = find(["answer", "long_answer", "final_decision", "a"])
    pubmed_id   = find(["pubmedid", "pmid", "pubmed_id", "id"])

    print(f"  Question column   : {q_col!r}")
    print(f"  Context column    : {ctx_col!r}")
    print(f"  Answer column     : {ans_col!r}")
    print(f"  PubMed ID column  : {pubmed_id!r}")
    print("  ℹ️  PubMedQA is optional and will NOT be indexed by default.")


def main() -> None:
    print("\n" + SEPARATOR)
    print("  MEDMITRA — Dataset Inspection Report")
    print(SEPARATOR)

    for label, path in FILES.items():
        inspect_file(label, path)

    print(f"\n{SEPARATOR}")
    print("  INSPECTION COMPLETE")
    print("  Next step: python scripts\\clean_disease_data.py")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
