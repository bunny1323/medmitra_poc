import os
import shutil
from pathlib import Path

base_dir = Path(r"c:\Users\vigne\Downloads\medmitra_branch")
books_dir = base_dir / "data" / "books"
registry_dir = base_dir / "data" / "registry"

# Create directories
books_dir.mkdir(parents=True, exist_ok=True)
registry_dir.mkdir(parents=True, exist_ok=True)

# Move PDF
src_pdf = base_dir / "data" / "current-medical-diagnosis-and-treatment-2025-1.pdf"
dest_pdf = books_dir / "current-medical-diagnosis-and-treatment-2025-1.pdf"

if src_pdf.exists():
    print(f"Moving {src_pdf} to {dest_pdf}...")
    shutil.move(str(src_pdf), str(dest_pdf))
else:
    print("Source PDF does not exist or already moved.")

# Remove legacy ingest_medical_book.py
legacy_script = base_dir / "scripts" / "ingest_medical_book.py"
if legacy_script.exists():
    print(f"Removing legacy script {legacy_script}...")
    os.remove(str(legacy_script))
else:
    print("Legacy script not found.")

print("Done organizing directories.")
