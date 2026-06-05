import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path: sys.path.insert(0, root)

from app.services.container import container
from app.core.config import settings

def main():
    qdrant_ok = container.qdrant_service.is_connected()
    if not qdrant_ok:
        print("Could not connect to Qdrant database.")
        sys.exit(1)
    alias = settings.QDRANT_COLLECTION_ALIAS
    exists = container.qdrant_service.check_collection_exists(alias)
    if exists:
        count = container.qdrant_service.get_collection_points_count(alias)
        print(f"Connected. Collection alias: {alias} has {count} points.")
    else:
        print(f"Collection alias: {alias} does not exist.")

if __name__ == "__main__": main()
