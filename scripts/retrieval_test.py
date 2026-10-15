"""Which retrieval strategy finds the right section? No model involved.

    uv run python scripts/retrieval_test.py
    uv run python scripts/retrieval_test.py -n 10      # deeper pile

Marked against evals/questions.yaml. Three numbers per strategy:

    hits   at least one expected section was somewhere in the passages
    MRR    mean reciprocal rank — 1/position, averaged. First place scores
           1.00, fifth scores 0.20, absent scores 0. A plain hit rate cannot
           tell "found it first" from "found it barely", and the model reads
           the top of the pile most carefully. With several expected sections
           the rank of the FIRST one found is used, which is what MRR means in
           the literature.
    cover  of all the expected sections, how many were retrieved. MRR says
           how fast you found SOMETHING relevant; coverage says whether you
           found EVERYTHING the question needs. "What are transactional
           events?" names five sections, and a top-ranked overview that
           misses Start Transaction and Meter Values is not a good answer.

WHY THIS MEASUREMENT IS TRUSTWORTHY, unlike Day 17's: retrieval makes no model
call. Same question, same index, same result, every time. Day 17's fabrication
count moved between identical runs because the model samples differently;
there is nothing here to sample.

Separate the deterministic half of a system from the random half, and measure
the deterministic half properly. Most of what you can actually fix lives there.
"""

import sys

from specrag.questions import Question, load
from specrag.retrieval import STRATEGIES
from specrag.store import index_summary


def section_prefix(section: str | None) -> str:
    """"6.31. MeterValues.req" -> "6.31" — compared as a prefix, never
    truncated. Day 16 read a correct 9.1.10 as a mismatch by chopping it."""
    return section.split()[0].rstrip(".") if section else "-"


def position_of(question: Question, rows: list[dict]) -> int | None:
    """1-based position of the first expected section, or None.

    The source check is not decoration. Section numbers are not unique across
    the corpus: "3.7" is Transaction-related messages in the core spec and
    "From", an XML header field, in the SOAP spec. Without it the SOAP section
    scores as a hit — and a false hit pushes the number UP, which reads as an
    improvement and so never gets investigated.
    """
    for position, row in enumerate(rows, start=1):
        if question.source and row.get("source") != question.source:
            continue
        label = section_prefix(row["section"])
        if any(label.startswith(prefix) for prefix in question.expect):
            return position
    return None


def main() -> None:
    limit = 5
    if "-n" in sys.argv:
        limit = int(sys.argv[sys.argv.index("-n") + 1])

    questions = [q for q in load().scored if q.expect]

    print(index_summary())
    print(f"\n  top {limit} passages per question, marked against the answer key\n")

    names = list(STRATEGIES)
    print(f"  {'expected':<9} {'question':<46} " + " ".join(f"{n:>9}" for n in names))
    print(f"  {'-' * 9} {'-' * 46} " + " ".join("-" * 9 for _ in names))

    positions: dict[str, list[int | None]] = {name: [] for name in names}
    covered: dict[str, list[tuple[int, int]]] = {name: [] for name in names}

    for case in questions:
        cells = []
        for name in names:
            rows = STRATEGIES[name](case.q, limit)
            position = position_of(case, rows)
            positions[name].append(position)
            covered[name].append(case.coverage(rows))
            got, want = covered[name][-1]
            # One cell has to carry both numbers. "2 3/5" reads as: first
            # relevant section at rank 2, three of five sections retrieved.
            # A lookup question shows just the rank, since 1/1 adds nothing.
            cell = f"{position or 'MISS'}" + (f" {got}/{want}" if want > 1 else "")
            cells.append(f"{cell:>9}")
        label = case.expect[0] + (f"+{len(case.expect) - 1}" if len(case.expect) > 1 else "")
        question = case.q if len(case.q) <= 46 else case.q[:43] + "..."
        print(f"  {label:<9} {question:<46} " + " ".join(cells))

    total = len(questions)
    print()
    print(f"  {'':<56} " + " ".join(f"{n:>9}" for n in names))
    hits = {n: sum(1 for p in positions[n] if p) for n in names}
    print(f"  {'hits (of ' + str(total) + ')':<56} "
          + " ".join(f"{hits[n]:>9}" for n in names))
    mrr = {n: sum(1 / p for p in positions[n] if p) / total for n in names}
    print(f"  {'MRR':<56} " + " ".join(f"{mrr[n]:>9.3f}" for n in names))
    print(f"  {'found at rank 1':<56} "
          + " ".join(f"{sum(1 for p in positions[n] if p == 1):>9}" for n in names))
    # Sections found over sections wanted, across the whole set. Unlike hits
    # and MRR this one notices a partially answered question.
    print(f"  {'sections covered':<56} "
          + " ".join(f"{sum(g for g, _ in covered[n])}/"
                     f"{sum(w for _, w in covered[n]):<7}" for n in names))

    # PER KIND, because the single average above is now misleading. 46 of the
    # 64 questions are title-derived lookups: easy, and phrased in the
    # document's own words. They outvote the 16 hand-written ones 3 to 1, so a
    # strategy could get worse at every hard question and the headline number
    # would barely move. An average over unlike things hides the thing you
    # most need to see.
    print()
    print(f"  {'by kind':<56} " + " ".join(f"{n:>9}" for n in names))
    for kind in sorted({c.kind for c in questions}, key=lambda k: k.value):
        rows_of_kind = [i for i, c in enumerate(questions) if c.kind is kind]
        cells = []
        for name in names:
            got = sum(covered[name][i][0] for i in rows_of_kind)
            want = sum(covered[name][i][1] for i in rows_of_kind)
            cells.append(f"{got}/{want}".rjust(9))
        label = f"{kind.value} ({len(rows_of_kind)} q)"
        print(f"  {label:<56} " + " ".join(cells))

    print()
    baseline = positions["vector"]
    for name in names:
        if name == "vector":
            continue
        better = [c.q for c, b, o in zip(questions, baseline, positions[name])
                  if (o or 99) < (b or 99)]
        worse = [c.q for c, b, o in zip(questions, baseline, positions[name])
                 if (o or 99) > (b or 99)]
        print(f"  {name} vs vector: {len(better)} better, {len(worse)} worse")
        for q in worse:
            print(f"      REGRESSED: {q}")

    print()
    print(f"  {total} questions. A difference of one is not a result — it is a")
    print("  reason to label more questions, which is what Day 19 is for.")


if __name__ == "__main__":
    main()
