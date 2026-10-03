#!/usr/bin/env python3
"""Seed the RAG memory store from existing approved experiment results.

Usage:
    cd P4Harness
    source venv/bin/activate
    python scripts/seed_rag.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.rag_memory import RAGMemory

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = str(_ROOT / "experiments" / "results")
STORE_PATH = str(_ROOT / "memory" / "rag_store.jsonl")


def main() -> None:
    print(f"[seed_rag] Initializing RAG store at: {STORE_PATH}")
    rag = RAGMemory(store_path=STORE_PATH)

    existing = len(rag)
    print(f"[seed_rag] Existing entries in store: {existing}")

    print(f"[seed_rag] Seeding from: {RESULTS_DIR}")
    added = rag.seed_from_results(RESULTS_DIR)

    total = len(rag)
    print(f"\n[seed_rag] Done — added {added} new entries, total store size: {total}")

    # Quick sanity-check retrieval
    if total > 0:
        print("\n[seed_rag] Sanity check — retrieve top-2 for a link-failure task:")
        test_query = "Port 1 is DOWN. Reroute SPI=100 SI=255 to port 2."
        results = rag.retrieve(test_query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] scenario={r.get('scenario')} | task[:60]={r['task'][:60]!r}")


if __name__ == "__main__":
    main()
