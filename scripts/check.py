"""Test the retrieval behaviour yourself.

    uv run python scripts/check.py                  # the four findings, side by side
    uv run python scripts/check.py "a question"     # ask one thing
    uv run python scripts/check.py --repl           # keep asking
    uv run python scripts/check.py --threshold      # answerable vs not, measured

Every number here is a cosine distance: 0 = same meaning, 1 = unrelated.
Lower is better.
"""

import sys

from specrag.labels import ANSWERABLE, UNANSWERABLE
from specrag.store import index_summary, search

def show(question: str, limit: int = 3) -> None:
    print(f"\nQ: {question}")
    for rank, row in enumerate(search(question, limit), 1):
        section = row["section"] or "(no section label)"
        text = " ".join(row["text"].split())
        print(f"  [{rank}] {row['distance']:.3f}  {section}  ·  p.{row['page']}")
        print(f"      {text[:200]}…")


def findings() -> None:
    """The four Day 15 failures, re-run against Day 16's chunks."""
    print("\n" + "=" * 74)
    print("FINDING 1 & 2 — clean starts, and whole enumerations")
    print("  Day 15: a chunk started at 'OtherError', mid-list, and nothing")
    print("  said the list was partial. Look at where this one starts.")
    show("what are the valid values for ChargePointErrorCode?", 1)

    print("\n" + "=" * 74)
    print("FINDING 3 — the identifier that got drowned out")
    print("  Day 15: returned p.103, about unrelated time intervals, at 0.622.")
    show("what is the heartbeat interval?", 2)

    print("\n" + "=" * 74)
    print("FINDING 5 — a question from the WRONG VERSION of the protocol")
    print("  TransactionEvent is OCPP 2.0.1. It appears in 0 of 405 chunks.")
    print("  Watch the distance: higher than a real question, but it still")
    print("  returns something. Nothing yet says 'not in these documents'.")
    show("what are transactional events?", 2)


def threshold() -> None:
    """Can a distance cut-off separate answerable from unanswerable?"""
    print("\n  Lower = better match. A threshold only works if the worst")
    print("  answerable question scores BETTER than the best unanswerable one.\n")

    best_answerable, best_unanswerable = [], []

    for case in ANSWERABLE:
        top = search(case["q"], 1)[0]
        best_answerable.append(top["distance"])
        # Compare as a PREFIX, not by truncating to two parts: the earlier
        # version read a correct "9.1.10" as a mismatch against "9.1.10"
        # because it chopped it to "9.1". My comparison was the bug, not the
        # retrieval — fifth time this sprint the instrument was wrong.
        section = (top["section"] or "")
        got = section.split()[0].rstrip(".") if section else "-"
        mark = "ok " if got.startswith(case["expect"]) else f"!! got {got}"
        print(f"  {top['distance']:.3f}  answerable  {mark:14} {case['q']}")
    print()
    for case in UNANSWERABLE:
        d = search(case["q"], 1)[0]["distance"]
        best_unanswerable.append(d)
        print(f"  {d:.3f}  ABSENT      {'':14} {case['q']}")

    worst_ok = max(best_answerable)
    best_bad = min(best_unanswerable)
    print(f"\n  worst answerable  : {worst_ok:.3f}")
    print(f"  best unanswerable : {best_bad:.3f}")
    margin = best_bad - worst_ok
    print(f"  margin            : {margin:+.3f}")

    # A margin is not a yes/no. 0.15 is a real gap; 0.02 is noise wearing a
    # decimal point, and calling it "separated" is how a tiny sample talks you
    # into shipping a threshold that refuses real questions.
    if margin <= 0:
        print("  OVERLAPPING — no cut-off can work. Something else must decide.")
    elif margin < 0.08:
        print(f"  TOO CLOSE TO USE — {margin:.3f} is within noise. A cut-off here")
        print("  would refuse real questions on a bad phrasing. The model has to")
        print("  make this call, reading the passages.")
    else:
        print(f"  usable gap — a cut-off near {(worst_ok + best_bad) / 2:.2f}")
        print("  (on this sample. Day 19's 60-100 questions settle it.)")


def main() -> None:
    print(index_summary())
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--threshold" in sys.argv:
        threshold()
    elif args:
        show(" ".join(args))
    elif "--repl" in sys.argv:
        print("Ask anything about OCPP 1.6. Blank line to quit.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
            show(q)
    else:
        findings()


if __name__ == "__main__":
    main()
