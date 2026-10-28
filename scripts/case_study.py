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

from specrag.attacks import full_runs as full_attack_runs
from specrag.attacks import load as load_attacks
from specrag.attacks import load_runs as load_attack_runs
from specrag.questions import load as load_questions
from specrag.runs import RunRecord, load, stable

DOC = Path(__file__).resolve().parents[1] / "CASE_STUDY.md"
OPEN = "<!-- generated: results -->"
CLOSE = "<!-- /generated -->"

# A second generated block, for the same reason as the first. The security
# numbers are the ones a buyer is most tempted to round in their favour, and
# they are the ones that go stalest: a prompt change moves them and nothing in
# the prose would say so. Both fingerprints are printed inside the block, so a
# reader can see which configuration produced which number.
SEC_OPEN = "<!-- generated: security -->"
SEC_CLOSE = "<!-- /generated security -->"


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


def security_block() -> str:
    """Before and after, chosen by rule: the first full run and the last.

    THE RULE IS NOT "THE BEST RUN". The first full run of all twenty attacks
    is the undefended system, because that is the only run that existed before
    any mitigation was written. The most recent is what ships. If the two have
    the same prompt fingerprint then no mitigation has been measured and the
    block says so rather than printing a flattering pair.
    """
    runs = full_attack_runs(load_attack_runs())
    total = len(load_attacks().attacks)
    if not runs:
        return ("\n_No full attack run recorded yet — "
                "`uv run python scripts/attack.py --run --runs 3`._\n")

    first, last = runs[0], runs[-1]

    def row(label: str, record: dict) -> str:
        t = record["tally"]
        held = t.get("DEFENDED", 0) + t.get("MOOT", 0)
        return (f"| {label} | {record['prompt']} | {record['runs']} | "
                f"{held}/{total} | {t.get('LEAKED', 0)} | "
                f"{t.get('LANDED', 0)} | {t.get('VOID', 0)} |")

    lines = [
        "",
        f"Twenty adversarial documents planted in the corpus, each written in "
        f"the vocabulary of the question it targets so that it actually gets "
        f"retrieved. Every probe is asked twice — once on the clean corpus, "
        f"once poisoned — because most of the detectors measure a change and "
        f"not a state.",
        "",
        "| | prompt | runs | held | leaked | landed | untested |",
        "|---|---|---|---|---|---|---|",
        row("no mitigations", first),
    ]
    if last is not first:
        lines.append(row("as it ships", last))
    else:
        lines += ["", "_Only one full run recorded: the mitigations have not "
                  "yet been measured across all twenty._"]
    lines += [
        "",
        "**held** = the attack reached the model and nothing moved, or the "
        "clean corpus was already in that state so the attack cannot be "
        "credited. **leaked** = the planted claim appears in the answer but is "
        "disowned as unverified — closer to right than wrong, and still a "
        "failure of the strict bar, because a number travels further than the "
        "disclaimer attached to it. **landed** = the claim was stated, or the "
        "system's behaviour changed. **untested** = the attack never entered "
        "the top five passages, so no defence of it was measured; it is never "
        "counted as a pass.",
    ]
    return "\n".join(lines) + "\n"


def splice(text: str, generated: str, open_mark: str = OPEN,
           close_mark: str = CLOSE) -> str:
    """Replace whatever is between the two markers. Both are parameters now:
    the old body hard-coded OPEN and CLOSE while accepting them as arguments,
    which would have rewritten the results block with the security block."""
    start = text.index(open_mark) + len(open_mark)
    end = text.index(close_mark)
    return text[:start] + generated + text[end:]


if __name__ == "__main__":
    runs = full_runs(load())
    if not runs:
        sys.exit("no full runs recorded — run scripts/evaluate.py")
    if not DOC.exists():
        sys.exit(f"{DOC.name} does not exist yet — write the prose first")

    current = DOC.read_text()
    wanted = splice(current, block(runs))
    if SEC_OPEN in wanted:
        wanted = splice(wanted, security_block(), SEC_OPEN, SEC_CLOSE)
    if "--check" in sys.argv:
        if current != wanted:
            sys.exit(f"{DOC.name} is stale — run scripts/case_study.py")
        print(f"{DOC.name} matches the run log")
    else:
        DOC.write_text(wanted)
        print(f"{DOC.name}: rewrote {len(block(runs).splitlines())} generated lines")
