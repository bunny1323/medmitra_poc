import os
import sys
import zipfile

# Safe project-root bootstrap for direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("==========================================")
    print("MedMitra Codebase ZIP Archiver Utility")
    print("==========================================\n")

    # Name of the output zip file
    zip_name = "medmitra_ml_service.zip"
    zip_path = os.path.join(project_root, zip_name)
    
    print(f"Archiving files to: {zip_path}")

    # Folders/files to exclude from zip
    exclude_dirs = {
        "venv", ".venv", "__pycache__", ".pytest_cache", 
        ".mypy_cache", ".ruff_cache", "qdrant_storage", 
        "logs", ".git", ".github"
    }
    exclude_files = {
        ".env", zip_name, "COMPLETE_TECHNICAL_DOCUMENTATION.docx"
    }

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_root):
                # Filter directories in place to prevent walks down excluded branches
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
                
                for file in files:
                    if file in exclude_files or file.endswith(".pyc") or file.endswith(".pdf"):
                        continue
                    
                    full_file_path = os.path.join(root, file)
                    # Compute relative path to keep folder structure clean inside the zip
                    rel_path = os.path.relpath(full_file_path, project_root)
                    
                    # Prefix files with root folder name so they unpack into a neat subfolder
                    archive_name = os.path.join("medmitra_ml_service", rel_path)
                    
                    zipf.write(full_file_path, archive_name)
                    print(f" -> Added: {archive_name}")

        print(f"\nSuccess: Archive created successfully at {zip_path}")
        
        # Calculate SHA-256 of the generated ZIP
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(zip_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        checksum = sha256_hash.hexdigest()
        print(f"SHA-256 Checksum: {checksum}")
        
        # Write checksum file
        checksum_path = os.path.join(project_root, "medmitra_ml_service.zip.sha256")
        with open(checksum_path, "w") as f:
            f.write(f"{checksum}  {zip_name}\n")
        print(f"Checksum saved to {checksum_path}")

    except Exception as e:
        print(f"Error compiling archive: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
