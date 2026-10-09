"""Which retrieval strategy finds the right section? No model involved.

    uv run python scripts/retrieval_test.py
    uv run python scripts/retrieval_test.py -n 10      # deeper pile

Marked against the answer key in specrag/labels.py. Two numbers per strategy:

    hits   the expected section was somewhere in the returned passages
    MRR    mean reciprocal rank — 1/position, averaged. First place scores
           1.00, fifth scores 0.20, absent scores 0. A plain hit rate cannot
           tell "found it first" from "found it barely", and the model reads
           the top of the pile most carefully.

WHY THIS MEASUREMENT IS TRUSTWORTHY, unlike Day 17's: retrieval makes no model
call. Same question, same index, same result, every time. Day 17's fabrication
count moved between identical runs because the model samples differently;
there is nothing here to sample.

Separate the deterministic half of a system from the random half, and measure
the deterministic half properly. Most of what you can actually fix lives there.
"""

import sys

from specrag.labels import ANSWERABLE
from specrag.retrieval import STRATEGIES
from specrag.store import index_summary


def section_prefix(section: str | None) -> str:
    """"6.31. MeterValues.req" -> "6.31" — compared as a prefix, never
    truncated. Day 16 read a correct 9.1.10 as a mismatch by chopping it."""
    return section.split()[0].rstrip(".") if section else "-"


def position_of(expected: str, rows: list[dict]) -> int | None:
    """1-based position of the expected section, or None."""
    for position, row in enumerate(rows, start=1):
        if section_prefix(row["section"]).startswith(expected):
            return position
    return None


def main() -> None:
    limit = 5
    if "-n" in sys.argv:
        limit = int(sys.argv[sys.argv.index("-n") + 1])

    print(index_summary())
    print(f"\n  top {limit} passages per question, marked against the answer key\n")

    names = list(STRATEGIES)
    print(f"  {'expected':<9} {'question':<46} " + " ".join(f"{n:>9}" for n in names))
    print(f"  {'-' * 9} {'-' * 46} " + " ".join("-" * 9 for _ in names))

    positions: dict[str, list[int | None]] = {name: [] for name in names}

    for case in ANSWERABLE:
        cells = []
        for name in names:
            rows = STRATEGIES[name](case["q"], limit)
            position = position_of(case["expect"], rows)
            positions[name].append(position)
            cells.append(f"{position:>9}" if position else f"{'MISS':>9}")
        question = case["q"] if len(case["q"]) <= 46 else case["q"][:43] + "..."
        print(f"  {case['expect']:<9} {question:<46} " + " ".join(cells))

    total = len(ANSWERABLE)
    print()
    print(f"  {'':<56} " + " ".join(f"{n:>9}" for n in names))
    hits = {n: sum(1 for p in positions[n] if p) for n in names}
    print(f"  {'hits (of ' + str(total) + ')':<56} "
          + " ".join(f"{hits[n]:>9}" for n in names))
    mrr = {n: sum(1 / p for p in positions[n] if p) / total for n in names}
    print(f"  {'MRR':<56} " + " ".join(f"{mrr[n]:>9.3f}" for n in names))
    print(f"  {'found at rank 1':<56} "
          + " ".join(f"{sum(1 for p in positions[n] if p == 1):>9}" for n in names))

    print()
    baseline = positions["vector"]
    for name in names:
        if name == "vector":
            continue
        better = [c["q"] for c, b, o in zip(ANSWERABLE, baseline, positions[name])
                  if (o or 99) < (b or 99)]
        worse = [c["q"] for c, b, o in zip(ANSWERABLE, baseline, positions[name])
                 if (o or 99) > (b or 99)]
        print(f"  {name} vs vector: {len(better)} better, {len(worse)} worse")
        for q in worse:
            print(f"      REGRESSED: {q}")

    print()
    print("  6 questions. A difference of one is not a result — it is a reason")
    print("  to label more questions, which is Day 19.")


if __name__ == "__main__":
    main()
