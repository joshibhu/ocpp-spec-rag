"""Read evals/runs.jsonl and say whether two runs may be compared at all.

    uv run python scripts/compare_runs.py                 # list recorded runs
    uv run python scripts/compare_runs.py c9c73069 3f40a1 # compare two

THE REASON THIS IS A SCRIPT AND NOT A HABIT: on Day 19 comparability broke
three times, and every time it broke silently. Two numbers were put side by
side, the difference was attributed to the change being tested, and the real
cause was a relabelled question or a larger question set. Nothing on screen
said "these two were not measuring the same thing".

So the comparison refuses to be quiet about it. A clean experiment moves
exactly one field of the configuration. Two fields moved means the result
cannot be attributed to either of them, and this prints that in place of the
comparison rather than underneath it.

THE PER-QUESTION DIFF IS THE POINT. A headline that goes from 46 to 48 is
two questions, and the only useful question is WHICH two - a fix that helps
four questions and breaks two reads identically to a fix that helps two and
breaks none, until the ids are on screen.
"""

import re
import sys
from dataclasses import fields

from specrag.runs import Config, RunRecord, load

# Derived from Config, never restated. Written out by hand, this list missed
# `effort` the moment Config gained it - so the one experiment whose single
# changed field was effort compared as though nothing had been configured
# differently. A tool built to catch unattributable comparisons produced one.
#
# `note` is excluded because it describes a run instead of configuring it.
# `runs` is NOT excluded: a 1-pass and a 3-pass number are different
# statistics, not the same statistic measured twice.
CONFIG_FIELDS = tuple(f.name for f in fields(Config) if f.name != "note")


def label(run: RunRecord) -> str:
    return (f"{run.totals.get('id', '????????')}  {run.started}  "
            f"{run.config.strategy:<9} {run.config.model:<12} "
            f"k={run.config.top_k} x{run.config.runs}  "
            f"prompt {run.config.prompt}  {run.config.note[:40]}")


def listing() -> None:
    runs = load()
    if not runs:
        sys.exit("no runs recorded yet — run scripts/evaluate.py")
    for run in runs:
        print(f"  {label(run)}")
    print(f"\n  {len(runs)} runs. Pass two ids to compare.")


def find(runs: list[RunRecord], wanted: str) -> RunRecord:
    hits = [r for r in runs if r.totals.get("id", "").startswith(wanted)]
    if not hits:
        sys.exit(f"no run id starts with {wanted!r}")
    if len(hits) > 1:
        sys.exit(f"{wanted!r} matches {len(hits)} runs — use more characters")
    return hits[0]


AGE = re.compile(r"\s*\([^)]*ago\)")


def comparable_value(name: str, value: object) -> object:
    """Strip the parts of a stored value that move without being changed.

    Runs recorded before index_identity() existed hold index_summary()'s
    prose, which carries a relative age. Normalising it here rather than
    rewriting those lines keeps the log append-only: the reader tolerates
    what an older writer produced, exactly as LEGACY_DEFAULTS does.
    """
    if name == "index" and isinstance(value, str):
        return AGE.sub("", value)
    return value


def differences(before: RunRecord, after: RunRecord) -> list[str]:
    return [f for f in CONFIG_FIELDS
            if comparable_value(f, getattr(before.config, f))
            != comparable_value(f, getattr(after.config, f))]


def compare(first: str, second: str) -> None:
    runs = load()
    before, after = find(runs, first), find(runs, second)

    print(f"  before  {label(before)}")
    print(f"  after   {label(after)}")
    print()

    moved = differences(before, after)
    for name in moved:
        print(f"  changed  {name}: {getattr(before.config, name)}"
              f"  ->  {getattr(after.config, name)}")
    if not moved:
        print("  configuration identical — this measures run-to-run variance,")
        print("  which is a real thing to measure and not an experiment.")
    elif len(moved) > 1:
        print()
        print(f"  NOT COMPARABLE: {len(moved)} fields moved together, so a")
        print("  difference in the result cannot be attributed to any one of")
        print("  them. Re-run with one change at a time.")
    print()

    # PARTIAL runs are excluded from attribution entirely rather than warned
    # about, because a three-question run next to a seventy-four-question run
    # produces percentages that look like a finding.
    for run, side in ((before, "before"), (after, "after")):
        if run.config.note.startswith("PARTIAL"):
            print(f"  {side} is a PARTIAL run — it tested the harness, not the")
            print("  system. Nothing below it is a result.")
            print()

    for key in ("answered", "covered", "cited", "refusals"):
        was, now = before.totals.get(key, 0), after.totals.get(key, 0)
        base = ("scored_refusals" if key == "refusals" else "scored_answerable")
        print(f"  {key:<10} {was}/{before.totals.get(base, 0)}"
              f"  ->  {now}/{after.totals.get(base, 0)}"
              f"   {arrow(now - was)}")

    was, now = before.totals.get("sections", [0, 0]), after.totals.get("sections", [0, 0])
    print(f"  {'sections':<10} {was[0]}/{was[1]}  ->  {now[0]}/{now[1]}"
          f"   {arrow(now[0] - was[0])}")

    print()
    print("  by kind          answered        full cover")
    for kind in sorted(set(before.per_kind) | set(after.per_kind)):
        b = before.per_kind.get(kind, [0, 0, 0, 0, 0])
        a = after.per_kind.get(kind, [0, 0, 0, 0, 0])
        print(f"  {kind:<15} {b[1]}/{b[0]} -> {a[1]}/{a[0]}"
              f"       {b[2]}/{b[0]} -> {a[2]}/{a[0]}")

    moved_questions(before, after)

    print()
    print(f"  {cost(before)}  ->  {cost(after)}")


def moved_questions(before: RunRecord, after: RunRecord) -> None:
    """Which questions changed verdict, by id.

    Net zero is the case this exists for: four fixed and four broken is not
    "no change", it is a fix that works on one group and a regression on
    another, and the headline cannot tell them apart.
    """
    was = {q["id"]: q for q in before.per_question}
    gains, losses, premise_moves = [], [], []

    for q in after.per_question:
        old = was.get(q["id"])
        if old is None:
            continue
        if q["answered"] > old["answered"]:
            gains.append(q["id"])
        elif q["answered"] < old["answered"]:
            losses.append(q["id"])
        if set(q.get("premise", [])) != set(old.get("premise", [])):
            premise_moves.append(
                f"{q['id']}: {'/'.join(sorted(set(old.get('premise', []))))}"
                f" -> {'/'.join(sorted(set(q.get('premise', []))))}")

    print()
    show("now answered", gains)
    show("now refused", losses)
    show("premise changed", premise_moves)
    if not (gains or losses or premise_moves):
        print("  no question changed verdict")


def show(title: str, items: list[str]) -> None:
    if items:
        print(f"  {title} ({len(items)}): {', '.join(items)}")


def cost(run: RunRecord) -> str:
    if run.usd is None:
        return f"{run.seconds:.0f}s, cost unknown"
    return f"{run.seconds:.0f}s, Rs {run.usd * 88:.2f} over {run.calls} calls"


def arrow(delta: int) -> str:
    if delta > 0:
        return f"+{delta}"
    return str(delta) if delta else "="


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) == 2:
        compare(*args)
    else:
        listing()
