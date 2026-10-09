"""Run the labelled questions through the whole pipeline and score it.

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py -v                    # show each answer
    uv run python scripts/evaluate.py --strategy vector     # compare strategies
    uv run python scripts/evaluate.py --runs 3              # same question 3x

Every question costs one embedding call and one chat call, so this is cheap
but not free. At 8 questions it is pennies; at Day 19's 100 it is worth
knowing the number.

THE POINT OF SEPARATING THE COLUMNS: an answer can fail in two unrelated
places, and the fix is different for each.

    found  = the expected section was among the retrieved passages
    cited  = the model actually used it

    found NO,  cited NO   -> RETRIEVAL failed. Fix chunking or search.
    found YES, cited NO   -> the model had it and ignored it. Fix the prompt.
    cited YES, still bad  -> the answer is wrong for a reason only a domain
                             expert can name.

Collapsing those into one "pass/fail" is how you end up tuning the wrong half
of the system for a week.

WHY --runs EXISTS: on Day 17 the same question, asked three times, produced
three different verdicts — three fabricated quotes, then clean, then clean.
Retrieval is deterministic; generation is not. So a single run cannot tell you
whether a fabrication was fixed or merely absent this time. Anything reported
from one run is an anecdote.
"""

import sys

from specrag.answer import answer
from specrag.labels import ANSWERABLE, UNANSWERABLE
from specrag.retrieval import STRATEGIES
from specrag.store import index_summary
from specrag.verify import verify


def section_prefix(section: str | None) -> str:
    """"9.1.10. HeartbeatInterval (1/2)" -> "9.1.10"

    Compared as a PREFIX, never truncated to two parts. Day 16 read a correct
    9.1.10 as a mismatch because it chopped it to 9.1.
    """
    if not section:
        return "-"
    return section.split()[0].rstrip(".")


def run(verbose: bool = False, strategy: str = "ident+vec", runs: int = 1) -> None:
    retrieve = STRATEGIES[strategy]

    print(index_summary())
    print(f"  strategy: {strategy}   runs per question: {runs}")
    print()
    print("  ans = answered (did not refuse)   found = expected section retrieved")
    print("  cited = model used it             flags = verifier findings")
    print()

    answered_right = found_right = cited_right = 0
    errors = warnings = 0

    for case in ANSWERABLE:
        expected = case["expect"]
        hits = {"ans": 0, "found": 0, "cited": 0}
        flags: list[str] = []

        for _ in range(runs):
            result = answer(case["q"], retrieve=retrieve)
            report = verify(result)

            hits["ans"] += result.reply.answerable
            hits["found"] += any(section_prefix(r["section"]).startswith(expected)
                                 for r in result.passages)
            hits["cited"] += any(section_prefix(r["section"]).startswith(expected)
                                 for r in result.cited())
            errors += sum(f.level == "error" for f in report.findings)
            warnings += sum(f.level == "warning" for f in report.findings)
            flags.extend(f.code for f in report.findings)

            if verbose:
                print(f"           {' '.join(result.reply.answer.split())[:220]}")

        # A question only counts as passing if it passed EVERY run. One good
        # run out of three is not a working system, it is a lucky sample.
        answered_right += hits["ans"] == runs
        found_right += hits["found"] == runs
        cited_right += hits["cited"] == runs

        print(f"  {tally(hits['ans'], runs, 'REFUSED'):>8}"
              f"  {tally(hits['found'], runs, 'NO'):>7}"
              f"  {tally(hits['cited'], runs, 'NO'):>7}"
              f"  {summarise(flags):<30} {case['q']}")
        if verbose:
            print(f"           expected {expected} · {case['why'][:90]}")

    print()
    refusals_right = 0
    for case in UNANSWERABLE:
        refused_count = 0
        flags = []
        for _ in range(runs):
            result = answer(case["q"], retrieve=retrieve)
            report = verify(result)
            refused_count += not result.reply.answerable
            errors += sum(f.level == "error" for f in report.findings)
            flags.extend(f.code for f in report.findings)
            if verbose:
                print(f"           {' '.join(result.reply.answer.split())[:220]}")
        refusals_right += refused_count == runs
        print(f"  {tally(refused_count, runs, 'ANSWERED', 'REFUSED'):>8}"
              f"  {'':>7}  {'':>7}  {summarise(flags):<30} {case['q']}")

    total = len(ANSWERABLE)
    print()
    print(f"  answered when it should      : {answered_right}/{total}")
    print(f"  expected section retrieved   : {found_right}/{total}")
    print(f"  expected section cited       : {cited_right}/{total}")
    print(f"  refused when it should       : {refusals_right}/{len(UNANSWERABLE)}")
    print(f"  verifier: {errors} fabrications, {warnings} warnings"
          f"  (over {runs * (total + len(UNANSWERABLE))} answers)")
    print()
    print("  8 questions is not a measurement — it is a smoke test. Day 19")
    print("  turns this into 60-100 labelled questions, which is where a")
    print("  percentage starts meaning something.")


def tally(hit: int, runs: int, bad: str, good: str = "yes") -> str:
    """Collapse N runs into one cell, without hiding disagreement.

    All runs agreed -> a word. They disagreed -> "2/3", which is the single
    most important thing to see, because a system that works two times in
    three is not a system that works.
    """
    if runs == 1:
        return good if hit else bad
    if hit == runs:
        return good
    if hit == 0:
        return bad
    return f"{hit}/{runs}"


def summarise(flags: list[str]) -> str:
    """Count repeats instead of listing them: altered-quote x2."""
    if not flags:
        return "-"
    counts: dict[str, int] = {}
    for code in flags:
        counts[code] = counts.get(code, 0) + 1
    return ",".join(c if n == 1 else f"{c} x{n}" for c, n in counts.items())


def option(name: str, default: str) -> str:
    """--name value, or the default."""
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    chosen = option("--strategy", "ident+vec")
    if chosen not in STRATEGIES:
        sys.exit(f"unknown strategy {chosen!r} — pick from {', '.join(STRATEGIES)}")
    run(verbose="-v" in sys.argv,
        strategy=chosen,
        runs=int(option("--runs", "1")))
