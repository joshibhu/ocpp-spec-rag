"""Vector search vs keyword search, side by side.

    uv run python scripts/compare.py "what is the heartbeat interval?"
    uv run python scripts/compare.py                    # keep asking

This is the diagnostic that tells you WHICH KIND of failure you have:

  both find it          -> retrieval is fine
  keyword finds it,
    vector does not     -> RETRIEVAL failure. The answer is there; meaning
                           search missed it, usually because the identifier
                           is buried in surrounding noise.
  neither finds it      -> the answer may genuinely not be in the corpus
  vector finds it,
    keyword does not    -> the question uses different words than the spec.
                           Exactly what embeddings are for.

Keyword search here is deliberately crude — ILIKE on a couple of terms, no
ranking. It is a probe, not a feature. Day 17 builds the real thing.
"""

import re
import sys

from specrag.db import query
from specrag.store import search

STOP = {
    "what", "is", "the", "a", "an", "of", "for", "to", "in", "on", "does",
    "do", "how", "when", "why", "which", "are", "and", "or", "it", "its",
    "be", "can", "must", "should", "if", "that", "this", "with", "by",
}
WIDTH = 190


def keywords(question: str) -> list[str]:
    """The content words, longest first — the most distinctive ones."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_.]+", question)
    kept = [w for w in words if w.lower() not in STOP and len(w) > 3]
    return sorted(set(kept), key=len, reverse=True)[:3]


def vector_hits(question: str, limit: int):
    return [
        (f"{r['distance']:.3f}", r["page"], r["text"])
        for r in search(question, limit=limit)
    ]


def keyword_hits(question: str, limit: int):
    terms = keywords(question)
    if not terms:
        return [], []
    # One ILIKE per term, OR'd. Crude on purpose.
    clauses = " OR ".join(["text ILIKE %s"] * len(terms))
    rows = query(
        f"SELECT page, text FROM chunks WHERE {clauses} ORDER BY page LIMIT %s",
        tuple(f"%{t}%" for t in terms) + (limit,),
    )
    return [("—", r["page"], r["text"]) for r in rows], terms


def show(label: str, hits, note: str = "") -> None:
    print(f"\n  {label}{note}")
    if not hits:
        print("     (nothing)")
        return
    for score, page, text in hits:
        flat = " ".join(text.split())
        print(f"     {score:>6}  p.{page:<4} {flat[:WIDTH]}…")


def compare(question: str, limit: int = 3) -> None:
    print(f"\n{'=' * 78}\nQ: {question}")
    show("BY MEANING (vector)", vector_hits(question, limit))
    hits, terms = keyword_hits(question, limit)
    show("BY WORDS   (keyword)", hits, f"   terms: {terms}")


def main() -> None:
    args = sys.argv[1:]
    if args:
        compare(" ".join(args))
        return
    print("Compare vector and keyword search. Blank line to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        compare(q)


if __name__ == "__main__":
    main()
