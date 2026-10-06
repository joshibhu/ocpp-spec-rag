"""Search the OCPP corpus by meaning.

    uv run python scripts/ask.py                       # keep asking
    uv run python scripts/ask.py "your question"       # ask one thing
    uv run python scripts/ask.py -n 5 "your question"  # more results

Retrieval only. No answer is generated — this shows you WHICH PARTS of the
spec the system would hand a model. Judging whether they are the right parts
is the point of today.
"""

import sys

from specrag.store import count, search

WIDTH = 300


def show(question: str, limit: int) -> None:
    print(f"\nQ: {question}\n")
    for rank, row in enumerate(search(question, limit=limit), start=1):
        text = " ".join(row["text"].split())          # collapse newlines
        print(f"  [{rank}]  distance {row['distance']:.3f}   "
              f"{row['source']} p.{row['page']}")
        print(f"       {text[:WIDTH]}{'…' if len(text) > WIDTH else ''}\n")


def main() -> None:
    args = sys.argv[1:]
    limit = 3
    if "-n" in args:
        i = args.index("-n")
        limit = int(args[i + 1])
        del args[i : i + 2]

    print(f"{count()} chunks indexed.")

    if args:
        show(" ".join(args), limit)
        return

    print("Ask anything about OCPP 1.6. Blank line to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        show(q, limit)


if __name__ == "__main__":
    main()
