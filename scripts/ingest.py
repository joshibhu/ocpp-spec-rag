"""Chunk the corpus, embed it, store it.

    uv run python scripts/ingest.py           # wipe and rebuild
    uv run python scripts/ingest.py --dry     # count and cost, no API calls

The corpus lives in docs/ and is NOT in git — OCPP specifications are
copyright Open Charge Alliance. Supply your own copy.
"""

import sys
import time
from pathlib import Path

from ocppqa.chunking import chunk_corpus, chunk_corpus_structured
from ocppqa.store import add, clear, count

DOCS = Path("docs")


def main() -> None:
    if not DOCS.exists() or not list(DOCS.glob("*.pdf")):
        sys.exit(f"No PDFs in {DOCS}/ — see the README.")

    # Structure-aware by default; --naive keeps the Day 15 baseline reachable
    # so the two can be compared rather than one believed better.
    naive = "--naive" in sys.argv
    chunks = chunk_corpus(DOCS) if naive else chunk_corpus_structured(DOCS)
    print(f"chunker: {'naive fixed-size' if naive else 'structure-aware'}")
    chars = sum(len(c.text) for c in chunks)
    tokens = chars // 4
    print(f"{len(chunks)} chunks, ~{tokens:,} tokens, "
          f"~Rs {tokens / 1_000_000 * 0.02 * 88:.2f} to embed")

    if "--dry" in sys.argv:
        return

    if "--fresh" in sys.argv:
        clear()
    started = time.time()
    add(
        [
            {"text": c.text, "source": c.source, "page": c.page, "section": c.section}
            for c in chunks
        ],
        progress=True,
    )
    print(f"stored {count()} chunks in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
