"""Run the labelled questions through the whole pipeline and score it.

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py -v        # show each answer

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
"""

import sys

from specrag.answer import answer
from specrag.labels import ANSWERABLE, UNANSWERABLE
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


def run(verbose: bool = False) -> None:
    print(index_summary())
    print()
    print("  ans = answered (did not refuse)   found = expected section retrieved")
    print("  cited = model used it             flags = verifier findings")
    print()

    answered_right = found_right = cited_right = 0
    errors = warnings = 0

    for case in ANSWERABLE:
        result = answer(case["q"])
        report = verify(result)

        expected = case["expect"]
        found = any(section_prefix(r["section"]) .startswith(expected)
                    for r in result.passages)
        cited = any(section_prefix(r["section"]).startswith(expected)
                    for r in result.cited())

        answered_right += result.reply.answerable
        found_right += found
        cited_right += cited
        errors += sum(f.level == "error" for f in report.findings)
        warnings += sum(f.level == "warning" for f in report.findings)

        flags = ",".join(f.code for f in report.findings) or "-"
        print(f"  {'yes' if result.reply.answerable else 'REFUSED':>8}"
              f"  {'yes' if found else 'NO ':>5}"
              f"  {'yes' if cited else 'NO ':>5}"
              f"  {flags:<28} {case['q']}")
        if verbose:
            print(f"           expected {expected} · {case['why'][:90]}")
            print(f"           {' '.join(result.reply.answer.split())[:220]}")

    print()
    refusals_right = 0
    for case in UNANSWERABLE:
        result = answer(case["q"])
        report = verify(result)
        refused = not result.reply.answerable
        refusals_right += refused
        errors += sum(f.level == "error" for f in report.findings)
        flags = ",".join(f.code for f in report.findings) or "-"
        print(f"  {'REFUSED' if refused else 'ANSWERED':>8}"
              f"  {'':>5}  {'':>5}  {flags:<28} {case['q']}")
        if verbose:
            print(f"           {' '.join(result.reply.answer.split())[:220]}")

    total = len(ANSWERABLE)
    print()
    print(f"  answered when it should      : {answered_right}/{total}")
    print(f"  expected section retrieved   : {found_right}/{total}")
    print(f"  expected section cited       : {cited_right}/{total}")
    print(f"  refused when it should       : {refusals_right}/{len(UNANSWERABLE)}")
    print(f"  verifier: {errors} fabrications, {warnings} warnings")
    print()
    print("  8 questions is not a measurement — it is a smoke test. Day 19")
    print("  turns this into 60-100 labelled questions, which is where a")
    print("  percentage starts meaning something.")


if __name__ == "__main__":
    run(verbose="-v" in sys.argv)
