"""
scripts/ingest_books.py
────────────────────────
CLI wrapper for the multi-book ingestion pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.services import ingestion_service
from app.services.retrieval_service import get_qdrant_client

def main():
    parser = argparse.ArgumentParser(description="MedMitra Multi-Book Ingestion Tool")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["append", "replace", "delete", "rebuild"], 
        default="append",
        help="Ingestion mode: append new books, replace a book, delete a book, or rebuild collection"
    )
    parser.add_argument(
        "--id", 
        type=str, 
        help="Source UUID of the book (required for replace and delete modes)"
    )
    
    args = parser.parse_args()
    
    client = None
    try:
        client = get_qdrant_client()
        if args.mode == "append":
            print("[CLI] Running in append mode...")
            res = ingestion_service.append_books()
            print(f"[CLI] Complete. Result: {res}")
            
        elif args.mode == "replace":
            if not args.id:
                print("[CLI] Error: --id <source_uuid> is required for replace mode.")
                sys.exit(1)
            print(f"[CLI] Replacing book with ID: {args.id}...")
            res = ingestion_service.replace_book(args.id)
            print(f"[CLI] Complete. Result: {res}")
            
        elif args.mode == "delete":
            if not args.id:
                print("[CLI] Error: --id <source_uuid> is required for delete mode.")
                sys.exit(1)
            print(f"[CLI] Deleting book with ID: {args.id}...")
            res = ingestion_service.delete_book(args.id)
            print(f"[CLI] Complete. Result: {res}")
            
        elif args.mode == "rebuild":
            print("[CLI] Rebuilding entire collection...")
            res = ingestion_service.full_rebuild()
            print(f"[CLI] Complete. Result: {res}")
            
    except Exception as e:
        print(f"[CLI] Error: {e}")
        sys.exit(1)
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    main()
