"""Ask a question and get an answer, with references.

    uv run python scripts/answer.py "must a charge point send StatusNotification?"
    uv run python scripts/answer.py                    # keep asking
    uv run python scripts/answer.py -v "..."           # show the passages too

Compare with scripts/ask.py, which shows retrieval only. Running both on the
same question is how you tell the two failure modes apart:

    ask.py has the right passage, answer.py gets it wrong  -> GENERATION bug
    ask.py does not have it either                         -> RETRIEVAL bug

Same symptom, opposite fixes.
"""

import sys

from specrag.answer import answer
from specrag.store import index_summary
from specrag.trust import label
from specrag.verify import verify

WIDTH = 160


def show(question: str, verbose: bool = False) -> None:
    result = answer(question)
    reply = result.reply

    print(f"\nQ: {question}\n")

    if not reply.answerable:
        print(f"  NOT IN THESE DOCUMENTS")
        print(f"  {reply.answer}")
    else:
        print(f"  {reply.answer}")

    # Day 22. Two signals that are computed, not reported by the model.
    #
    # The first is the one a reader cannot work out for themselves. A citation
    # that resolves to a real section of a real document looks identical
    # whether that document is the specification or a vendor PDF somebody
    # dropped in the corpus - and our pipeline attaches the section number, so
    # a fabricated claim arrives wearing a genuine, checkable reference. A lie
    # with a fake citation is caught by opening the spec. A lie with a real
    # one is not. This line is the only thing standing between those two.
    if result.unverified():
        print("\n  GROUNDED ON AN UNVERIFIED SOURCE — this answer rests on a")
        print("  document in the corpus that is not an Open Charge Alliance")
        print("  publication. Do not treat it as what OCPP 1.6 requires.")

    if reply.instructed:
        passages = ", ".join(f"[{n}]" for n in reply.instructed)
        print(f"\n  INSTRUCTION ATTEMPT in passage {passages} — a retrieved")
        print("  document tried to direct how this question was answered.")

    cited = result.cited()
    if cited:
        print("\n  Sources:")
        for row in cited:
            section = row["section"] or "(unlabelled)"
            print(f"    {section}  ·  {row['source']} p.{row['page']}"
                  f"   (distance {row['distance']:.3f})")

    # Every answer is checked. Not as a test run separately and forgotten —
    # the check is part of showing the answer, because an unchecked answer is
    # what a client would actually be reading.
    report = verify(result)
    if report.findings:
        print()
        for finding in report.findings:
            tag = "FABRICATION" if finding.level == "error" else "warning"
            print(f"  [{tag}] {finding.code}")
            for line in finding.detail.splitlines():
                print(f"      {line}")
    elif reply.answerable:
        print("\n  checked: every quote verbatim, every citation real")

    if verbose:
        print("\n  Passages it was given:")
        for number, row in enumerate(result.passages, start=1):
            mark = "*" if number in reply.used else " "
            text = " ".join(row["text"].split())
            print(f"   {mark}[{number}] {row['distance']:.3f}  "
                  f"{row['section'] or '-'}  p.{row['page']}  "
                  f"({label(row.get('source'))})")
            print(f"        {text[:WIDTH]}…")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv

    print(index_summary())

    if args:
        show(" ".join(args), verbose)
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
        show(q, verbose)


if __name__ == "__main__":
    main()
