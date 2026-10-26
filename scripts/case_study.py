"""Regenerate the measured numbers inside CASE_STUDY.md from the run log.

    uv run python scripts/case_study.py            # rewrite the block
    uv run python scripts/case_study.py --check    # fail if it is stale

CASE_STUDY.md is the document someone decides with, and every number in it
is a hostage: it was true on the day it was pasted and nothing tells the
reader when that was. Day 19 shipped exactly that - a README headline that
had stopped being true, discovered on Day 20 by accident.

So the prose is written by hand and the numbers are not. Everything between
the markers is derived from evals/runs.jsonl on demand, and --check belongs
in CI: a case study whose numbers no longer match the log is a broken build,
not a documentation chore.

THE HEADLINE RUN IS CHOSEN BY RULE, NOT BY ID. Naming a run id here would be
hand-typing a number with extra steps - and it would quietly become a
cherry-pick the moment a better run existed. The rule is "the most recent
full run of the system as it ships", which means the provider's default
reasoning effort, because reasoning_effort=low was measured and NOT adopted.
Experiments appear underneath as experiments.
"""

import sys
from pathlib import Path

from specrag.questions import load as load_questions
from specrag.runs import RunRecord, load, stable

DOC = Path(__file__).resolve().parents[1] / "CASE_STUDY.md"
OPEN = "<!-- generated: results -->"
CLOSE = "<!-- /generated -->"


def full_runs(runs: list[RunRecord]) -> list[RunRecord]:
    """Runs that measured the system. PARTIAL ones tested the harness."""
    return [r for r in runs if not r.config.note.startswith("PARTIAL")]


def headline(runs: list[RunRecord]) -> RunRecord:
    shipped = [r for r in runs if not r.config.effort]
    if not shipped:
        sys.exit("no run of the shipped configuration — see the docstring")
    return shipped[-1]


def pct(got: int, want: int) -> str:
    return f"{got}/{want} ({100 * got / want:.0f}%)" if want else "-"


def block(runs: list[RunRecord]) -> str:
    main = headline(runs)
    t = main.totals
    key = load_questions()
    lines = [
        "", f"Measured on {len(key.scored)} labelled questions "
        f"({t['scored_answerable']} that should be answered, "
        f"{t['scored_refusals']} that should be refused), "
        f"{main.config.runs} runs each, {main.calls} answers. A question "
        "counts only if it passes every run; two of three is not a working "
        "system.", "",
        f"| | |", "|---|---|",
        f"| answered when it should | {pct(t['answered'], t['scored_answerable'])} |",
        f"| all expected sections retrieved | {pct(t['covered'], t['scored_answerable'])} |",
        f"| all expected sections cited | {pct(t['cited'], t['scored_answerable'])} |",
        f"| refused when it should | {pct(t['refusals'], t['scored_refusals'])} |",
        f"| sections retrieved overall | {pct(*t['sections'])} |",
        f"| fabrications caught by the verifier | {t['errors']} over {main.calls} answers |",
        f"| cost | Rs {main.usd * 88:.2f} per run, "
        f"Rs {main.usd * 88 / main.calls:.2f} per answer |"
        if main.usd is not None else "| cost | unpriced model |",
        f"| latency | {main.seconds / main.calls:.0f}s per answer |",
        "",
        "Broken out by question kind, because an average over unlike "
        "questions hides the group that matters. The 46 lookups outvote the "
        "14 operational questions three to one:", "",
        "| kind | n | answered | all sections retrieved | sections |",
        "|---|---|---|---|---|",
    ]
    for kind, (n, ans, full, got, want) in sorted(main.per_kind.items()):
        lines.append(f"| {kind} | {n} | {ans}/{n} | {full}/{n} | {got}/{want} |")

    lines += [
        "", "Every run is recorded with the configuration that produced it. "
        "A number is comparable to another number only while the measuring "
        "setup is frozen, so the configuration is fingerprinted rather than "
        "described:", "",
        "| run | configuration | answered | retrieved | cited | time | cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        cost = f"Rs {r.usd * 88:.2f}" if r.usd is not None else "unpriced"
        lines.append(
            f"| `{r.totals['id']}` | {r.config.strategy}, {r.config.model}, "
            f"k={r.config.top_k}, effort={r.config.effort or 'default'} | "
            f"{r.totals['answered']}/{r.totals['scored_answerable']} | "
            f"{r.totals['covered']}/{r.totals['scored_answerable']} | "
            f"{r.totals['cited']}/{r.totals['scored_answerable']} | "
            f"{r.seconds / 60:.0f} min | {cost} |")

    lines += ["", f"Index: {stable('index', main.config.index)}. "
              f"Answer key: {key.summary()}. "
              f"Prompt fingerprint `{main.config.prompt}`, "
              f"answer key fingerprint `{main.config.questions}`.", ""]
    return "\n".join(lines)


def splice(text: str, generated: str) -> str:
    start, end = text.index(OPEN) + len(OPEN), text.index(CLOSE)
    return text[:start] + generated + text[end:]


if __name__ == "__main__":
    runs = full_runs(load())
    if not runs:
        sys.exit("no full runs recorded — run scripts/evaluate.py")
    if not DOC.exists():
        sys.exit(f"{DOC.name} does not exist yet — write the prose first")

    current = DOC.read_text()
    wanted = splice(current, block(runs))
    if "--check" in sys.argv:
        if current != wanted:
            sys.exit(f"{DOC.name} is stale — run scripts/case_study.py")
        print(f"{DOC.name} matches the run log")
    else:
        DOC.write_text(wanted)
        print(f"{DOC.name}: rewrote {len(block(runs).splitlines())} generated lines")
