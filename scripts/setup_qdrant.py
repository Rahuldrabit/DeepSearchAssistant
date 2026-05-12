#!/usr/bin/env python3
"""Initialise the Qdrant local database and verify the collection.

Run this once before first use:
    python scripts/setup_qdrant.py

Also useful to reset the index:
    python scripts/setup_qdrant.py --reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup / verify Qdrant collection")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection (deletes all indexed data)")
    args = parser.parse_args()

    from deepsearch.core.config import get_settings
    from deepsearch.storage.vector_store import COLLECTION_NAME, VectorStore

    cfg = get_settings()
    qdrant_path = cfg.qdrant_path
    qdrant_path.mkdir(parents=True, exist_ok=True)

    if args.reset:
        import shutil
        confirm = input(f"This will DELETE all indexed data at {qdrant_path}. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(qdrant_path)
        qdrant_path.mkdir(parents=True)
        print("Qdrant data directory cleared.")

    print(f"Initialising Qdrant at {qdrant_path}…")
    vs = VectorStore(qdrant_path)
    print(f"Collection '{COLLECTION_NAME}' ready.")
    print(f"Current document count: {vs.count}")
    print("Setup complete.")


if __name__ == "__main__":
    main()
