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

    cover  how many of the expected sections were retrieved
    cited  how many of them the model actually used

    cover 0/1, cited 0/1  -> RETRIEVAL failed. Fix chunking or search.
    cover 1/1, cited 0/1  -> the model had it and ignored it. Fix the prompt.
    cover 1/1, cited 1/1  -> if it is still wrong, the reason is one only a
                             domain expert can name.
    cover 3/5             -> partially answered, which a pass/fail column
                             cannot express. "What are transactional events?"
                             needs five sections; three of them is an answer
                             missing Start Transaction and Meter Values.

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
from specrag.questions import Question, load
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
    questions = load()

    print(index_summary())
    print(f"  {questions.summary()}")
    print(f"  strategy: {strategy}   runs per question: {runs}")
    print()
    print("  ans = answered (did not refuse)   cover = expected sections retrieved")
    print("  cited = of those, model used      flags = verifier findings")
    print("  ~ = reported but not scored (ambiguous)")
    print()

    answered_right = covered_right = cited_right = refusals_right = 0
    sections_got = sections_want = 0
    errors = warnings = 0
    scored_answerable = scored_refusals = 0

    for case in questions.questions:
        answered = 0
        cover_got, want = 0, 0          # retrieval is deterministic: one value
        cited_gots: list[int] = []      # generation is not: one per run
        flags: list[str] = []

        for _ in range(runs):
            result = answer(case.q, retrieve=retrieve)
            report = verify(result)

            answered += result.reply.answerable
            cover_got, want = case.coverage(result.passages)
            cited_gots.append(case.coverage(result.cited())[0])
            sections_got += cover_got
            sections_want += want
            errors += sum(f.level == "error" for f in report.findings)
            warnings += sum(f.level == "warning" for f in report.findings)
            flags.extend(f.code for f in report.findings)

            if verbose:
                print(f"           {' '.join(result.reply.answer.split())[:220]}")

        # A question passes only if it passed EVERY run. One good run out of
        # three is a lucky sample, not a working system.
        if case.scored:
            if case.answerable:
                scored_answerable += 1
                answered_right += answered == runs
                covered_right += want > 0 and cover_got == want
                cited_right += want > 0 and all(c == want for c in cited_gots)
            else:
                scored_refusals += 1
                refusals_right += answered == 0

        print(f"  {'~' if not case.scored else ' '}"
              f" {verdict(case, answered, runs):>8}"
              f"  {span([cover_got], want):>7}"
              f"  {span(cited_gots, want):>7}"
              f"  {summarise(flags):<30} {case.q}")
        if verbose and case.expect:
            print(f"           expect {case.expect} · {case.why.strip()[:90]}")

    print()
    print(f"  answered when it should      : {answered_right}/{scored_answerable}")
    print(f"  all expected sections found  : {covered_right}/{scored_answerable}")
    print(f"  all expected sections cited  : {cited_right}/{scored_answerable}")
    print(f"  sections retrieved overall   : {sections_got}/{sections_want}")
    print(f"  refused when it should       : {refusals_right}/{scored_refusals}")
    print(f"  verifier: {errors} fabrications, {warnings} warnings"
          f"  (over {runs * len(questions.questions)} answers)")
    print()
    print(f"  {len(questions.questions)} questions is not a measurement — it is a smoke")
    print("  test. 60-100 labelled questions is where a percentage starts")
    print("  meaning something.")


def verdict(case: Question, answered: int, runs: int) -> str:
    """The answered/refused cell, read from the question's point of view.

    For an answerable question the good outcome is "it answered". For an
    unanswerable one the good outcome is "it refused" — so the count is
    inverted rather than the column being relabelled.
    """
    if case.answerable:
        return tally(answered, runs, "REFUSED")
    return tally(runs - answered, runs, "ANSWERED", "REFUSED")


def span(got: list[int], want: int) -> str:
    """Sections found over sections wanted, across runs.

    Retrieval is deterministic, so the cover column always has one value and
    prints "3/5". Citation is not: the model may use four of five sections in
    one run and two in the next, and that prints "2-4/5". A single number
    there would hide the variance, which is the one thing Day 17 taught.
    """
    if not want:
        return ""
    low, high = min(got), max(got)
    return f"{low}/{want}" if low == high else f"{low}-{high}/{want}"


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
