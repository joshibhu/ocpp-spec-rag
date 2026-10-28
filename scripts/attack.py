"""Attack our own RAG with twenty adversarial documents.

    uv run python scripts/attack.py --reach      # does each attack reach the model?
    uv run python scripts/attack.py --clean      # emergency: purge planted chunks

REACHABILITY IS A SEPARATE MEASUREMENT, AND IT HAS TO COME FIRST.

An attack that is never retrieved cannot be defended against, and scoring it
as "defended" is the single easiest way to publish a 20/20 that means nothing.
Our own index is 405 chunks; five are shown to the model. An adversarial
document that loses the similarity contest is not a blocked attack - it is an
untested one, and the correct verdict is VOID.

This mode makes no model call at all. Retrieval is the deterministic half of
the system, so this number is the same on every run - the Day 18 lesson about
measuring the deterministic half properly, applied to security.

THE SLOT MATTERS AND IS REPORTED.

Day 17 established that the model reads passage [1] as the most important
thing it was given, and answer.py presents passages in document order, which
is decided by the source FILENAME. So where an attack lands in the list is
decided by a string, not by the attack. It is printed for every attack rather
than argued about afterwards.
"""

import sys

from specrag.answer import RETRIEVE, _in_document_order
from specrag.attacks import attack_corpus, attack_sources, load, planted, remove
from specrag.retrieval import STRATEGIES
from specrag.store import count, index_summary


def slot_of(text: str, rows: list[dict]) -> int | None:
    """1-based position of the planted chunk among the passages, or None.

    Matched on the chunk's exact text, not on (source, section): two attacks
    may plausibly claim the same section of the same fictional document, and a
    verdict attributed to the wrong attack is worse than no verdict.

    `_in_document_order` is imported from answer.py rather than reimplemented
    here. The slot being reported has to be the slot the model actually sees;
    a second copy of the ordering rule would be correct until the day somebody
    changed one of them. (Day 20 lost an experiment to exactly that - a
    hand-written field list that no longer matched the dataclass it described.)
    """
    for position, row in enumerate(_in_document_order(rows), start=1):
        if row["text"] == text:
            return position
    return None


def reach(strategy: str = "ident+vec", limit: int = RETRIEVE) -> None:
    retrieve = STRATEGIES[strategy]
    spec = load()

    print(f"\n  {index_summary()}")
    with attack_corpus() as n:
        print(f"  planted {n} adversarial chunks, index now {count()}")
        print(f"  strategy: {strategy}   top-k: {limit}\n")

        reached: list[str] = []
        unreached: list[str] = []
        for attack in spec.attacks:
            rows = retrieve(attack.probe, limit=limit)
            slot = slot_of(attack.text, rows)
            if slot is None:
                unreached.append(attack.id)
                mark = "  --  not retrieved"
            else:
                reached.append(attack.id)
                mark = f"  [{slot}]"
            print(f"  {attack.id:32} {attack.family.value:22}{mark}")

        print(f"\n  reached the model : {len(reached)}/{len(spec.attacks)}")
        if unreached:
            print("  NOT retrieved, so untestable (verdict VOID, never PASS):")
            for attack_id in unreached:
                print(f"    {attack_id}")

    print(f"  cleaned up, index back to {count()} chunks\n")


def clean() -> None:
    """For when a run died before its finally block could run."""
    before = planted()
    if not before:
        print(f"  nothing planted — index is {count()} chunks, clean")
        return
    remove()
    print(f"  removed {before} adversarial chunks from {', '.join(attack_sources())}")
    print(f"  index now {count()} chunks")


def option(name: str, default: str) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    elif "--reach" in sys.argv:
        chosen = option("--strategy", "ident+vec")
        if chosen not in STRATEGIES:
            sys.exit(f"unknown strategy {chosen!r}")
        reach(strategy=chosen, limit=int(option("--limit", str(RETRIEVE))))
    else:
        sys.exit(__doc__)
