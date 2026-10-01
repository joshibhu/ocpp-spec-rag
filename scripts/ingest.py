"""Chunk the corpus, embed it, store it.

    uv run python scripts/ingest.py           # wipe and rebuild
    uv run python scripts/ingest.py --dry     # count and cost, no API calls

The corpus lives in docs/ and is NOT in git — OCPP specifications are
copyright Open Charge Alliance. Supply your own copy.
"""

import sys
import time
from pathlib import Path

from ocppqa.chunking import chunk_corpus
from ocppqa.store import add, clear, count

DOCS = Path("docs")


def main() -> None:
    if not DOCS.exists() or not list(DOCS.glob("*.pdf")):
        sys.exit(f"No PDFs in {DOCS}/ — see the README.")

    chunks = chunk_corpus(DOCS)
    chars = sum(len(c.text) for c in chunks)
    tokens = chars // 4
    print(f"{len(chunks)} chunks, ~{tokens:,} tokens, "
          f"~Rs {tokens / 1_000_000 * 0.02 * 88:.2f} to embed")

    if "--dry" in sys.argv:
        return

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
