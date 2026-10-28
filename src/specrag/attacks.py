"""The adversarial corpus: load it, plant it, and get it back out again.

Day 22 asks a question the previous five days could not: what happens when a
passage in our own database is not true? Rule 1 of the prompt is "use ONLY the
passages", so a false passage is not a bug the model can be blamed for - it is
the system working exactly as designed on poisoned input.

THE DANGEROUS PART OF THIS MODULE IS NOT PLANTING. IT IS FORGETTING TO REMOVE.

Twenty adversarial chunks left behind in the index do not announce themselves.
Every later run would quietly answer from a corpus containing fabricated
normative text, and the accuracy numbers would move for a reason nobody
recorded. That is the stale-index failure that cost this sprint six separate
debugging sessions, except worse, because the index would look healthy.

Three things guard against it, in order of how much they are worth:

    planted()         counts them, so the state is observable at all
    attack_corpus()   a context manager, so removal runs in a finally block
                      (Java: try-with-resources)
    verify_clean()    asserts the count is zero and the baseline is intact,
                      called AFTER removal rather than trusted to have worked

The context manager handles the ordinary case including exceptions. It does
nothing at all for a SIGKILL, a closed laptop or a pulled power cable, which
is why planted() exists and why it is cheap enough to call before every run.
"""

from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .db import execute, query
from .store import add, count

ATTACKS_PATH = Path(__file__).resolve().parents[2] / "evals" / "attacks.yaml"


class Family(str, Enum):
    """What an attack is trying to achieve - which decides its mitigation.

    The families are not a taxonomy for its own sake. Each one is defeated by
    a different layer, and sorting the results by family is what turns twenty
    verdicts into a mitigation list instead of twenty anecdotes.
    """

    AUTHORITY_SPOOF = "authority-spoof"
    INSTRUCTION_INJECTION = "instruction-injection"
    REFUSAL_HIJACK = "refusal-hijack"
    CITATION_LAUNDERING = "citation-laundering"
    RETRIEVAL_SQUATTING = "retrieval-squatting"
    PROMPT_EXFILTRATION = "prompt-exfiltration"
    COST_AMPLIFICATION = "cost-amplification"


class Attack(BaseModel):
    """One adversarial document and the evidence that it worked.

    `extra="forbid"` for the same reason questions.py forbids extras: a
    misspelled `sais:` would silently leave the attack with no detector, and
    an attack with no detector can never be recorded as having landed. The
    test would pass because nothing was looking.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    family: Family
    source: str              # a key in the file's `sources` map, not a filename
    section: str
    probe: str
    text: str

    # ── detectors ────────────────────────────────────────────────────────────
    # Any one firing means the attack landed. OR is the correct default
    # because it is the attacker's test, not ours: they need one of these to
    # happen, not all of them.
    says: list[str] = Field(default_factory=list)
    omits: list[str] = Field(default_factory=list)
    premise: str | None = None
    answers_anyway: bool = False
    cites_all: bool = False
    reasoning_ratio: float | None = None

    @model_validator(mode="after")
    def has_a_detector(self) -> "Attack":
        if not any([
            self.says, self.omits, self.premise, self.answers_anyway,
            self.cites_all, self.reasoning_ratio,
        ]):
            raise ValueError(f"{self.id}: no detector — nothing would be measured")
        return self


class AttackFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    sources: dict[str, str]
    attacks: list[Attack]

    @model_validator(mode="after")
    def check(self) -> "AttackFile":
        ids = [a.id for a in self.attacks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate attack ids — results are keyed by id")
        for a in self.attacks:
            if a.source not in self.sources:
                raise ValueError(f"{a.id}: unknown source key {a.source!r}")
        return self

    def filename(self, attack: Attack) -> str:
        return self.sources[attack.source]


@lru_cache(maxsize=1)
def load() -> AttackFile:
    return AttackFile.model_validate(yaml.safe_load(ATTACKS_PATH.read_text()))


def attack_sources() -> list[str]:
    """Every filename the adversarial corpus is allowed to occupy.

    Removal works by source, so this list is the blast radius. Keeping it
    derived from the file means a new attack source cannot be added without
    the cleanup learning about it - the same "derive, never restate" fix that
    Day 20 needed when a hand-written field list missed the one field that
    had changed.
    """
    return sorted(load().sources.values())


def planted() -> int:
    """How many adversarial chunks are in the index right now."""
    rows = query(
        "SELECT count(*) AS n FROM chunks WHERE source = ANY(%s::text[])",
        (attack_sources(),),
    )
    return rows[0]["n"]


def plant() -> int:
    """Embed and insert the adversarial corpus. Returns how many went in."""
    spec = load()
    records = [
        {
            "text": a.text,
            "source": spec.filename(a),
            "section": a.section,
            # No page. These documents have no pages, and inventing one would
            # be the first thing in this file that was dishonest.
            "page": None,
        }
        for a in spec.attacks
    ]
    return add(records)


def remove() -> int:
    """Delete every chunk belonging to an adversarial source."""
    before = planted()
    execute(
        "DELETE FROM chunks WHERE source = ANY(%s::text[])", (attack_sources(),)
    )
    return before


def verify_clean(baseline: int) -> None:
    """Assert the index is back to exactly what it was.

    Checks both directions, because only one of them is the obvious bug. A
    leftover attack chunk poisons future answers; a MISSING baseline chunk
    means the cleanup deleted real corpus, which is worse and silent - the
    index would still look plausible and one section would simply have
    stopped existing.
    """
    left = planted()
    if left:
        raise RuntimeError(f"{left} adversarial chunks still in the index")
    now = count()
    if now != baseline:
        raise RuntimeError(
            f"index is {now} chunks, baseline was {baseline} — "
            "cleanup removed or left something it should not have"
        )


@contextmanager
def attack_corpus() -> Iterator[int]:
    """Plant the adversarial corpus for the duration of the block.

    Yields the number of chunks planted. Removal is in a finally, so an
    exception inside the block - a timed-out API call, a Ctrl-C - still
    leaves the index clean. The baseline count is taken before planting and
    asserted after removal, so "it cleaned up" is checked rather than assumed.
    """
    if planted():
        raise RuntimeError(
            f"{planted()} adversarial chunks are ALREADY in the index — "
            "a previous run did not clean up; run scripts/attack.py --clean"
        )
    baseline = count()
    try:
        yield plant()
    finally:
        remove()
        verify_clean(baseline)
