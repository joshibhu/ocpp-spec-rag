"""Which passages may ground a normative claim, and which may not.

THE INFORMATION NEEDED TO DEFEND WAS ALREADY IN THE DATABASE.

Every chunk has carried a `source` since Day 11, for citations. Nothing ever
read it to decide whether the text could be BELIEVED. So a planted chunk
claiming "ConnectionTimeOut SHALL default to 47 seconds" was handed to the
model indistinguishable from section 9.1 of the specification - not because
its provenance was unknown, but because we never asked.

That is the shape of most of what Day 22 found. The attacks did not defeat a
defence; they walked through a place where no defence had been put.

TIER, NOT FILENAME - AND THIS IS THE LOAD-BEARING DECISION HERE.

answer.py deliberately hides the section and page from the model: "if the
model can see 9.1.10, p.103 it will copy it into its prose, and then we are
back to trusting a string it wrote". A document NAME is the same hazard. So
the model is told the class of a passage, never its source:

    [1] (normative)
    [4] (UNVERIFIED SOURCE)

A class has three possible values and we assign it. A filename is a string,
and a model that can read "errata-community-ocpp-1.6.pdf" can write "per the
community errata sheet" - a citation we did not grant and cannot check.

WHY A LABEL IS NOT THE WHOLE FIX

A label in the prompt is advice, and an attack whose entire content is advice
is well placed to argue with it. The label is layer one. Layer two is
`grounding()`, which does not ask the model anything: the model returns
passage NUMBERS, we know each number's tier, so we compute what the answer
actually rested on. Same trick as citations - a number the model picks can be
validated against a list we control, a string it writes cannot.
"""

from enum import Enum
from typing import Any

from .settings import settings


class Tier(str, Enum):
    """How far a passage may be trusted.

    Two tiers, not a score. A number would invite arithmetic - averaging the
    trust of five passages - and the answer to "may this ground a SHALL" is
    yes or no, never 0.6.
    """

    NORMATIVE = "normative"
    UNVERIFIED = "unverified"


def normative_sources() -> set[str]:
    """The allow-list, from settings so a deployment can change it.

    An allow-list and not a deny-list, which is the whole point: a new vendor
    PDF dropped into the corpus next month is untrusted by DEFAULT. A
    deny-list would have to be updated by the person adding the document, and
    that person is exactly who the list is protecting us from.
    """
    return {name.strip() for name in settings.normative_sources.split(",") if name.strip()}


def tier(source: str | None) -> Tier:
    """A passage with no source at all is unverified, not normative.

    `None` reaching here would mean a chunk was stored without provenance,
    which the schema forbids - but if the schema is ever relaxed, failing
    closed is the only safe direction.
    """
    return Tier.NORMATIVE if source in normative_sources() else Tier.UNVERIFIED


def label(source: str | None) -> str:
    """What the model is shown about a passage's provenance. Tier only."""
    return "normative" if tier(source) is Tier.NORMATIVE else "UNVERIFIED SOURCE"


def grounding(passages: list[dict[str, Any]], used: list[int]) -> set[Tier]:
    """Which tiers the answer actually rested on. Computed, never asked for.

    Out-of-range numbers are ignored here rather than treated as untrusted:
    they are already reported as fabricated citations by the verifier, and
    double-counting one failure as two makes a report harder to act on.
    """
    return {
        tier(passages[n - 1].get("source"))
        for n in used
        if 1 <= n <= len(passages)
    }


def rests_on_unverified(passages: list[dict[str, Any]], used: list[int]) -> bool:
    """True when any passage the answer used came from outside the allow-list.

    The caller's decision what to do about it. A support agent's console might
    show a banner; a billing integration might refuse to display the answer at
    all. What matters is that the distinction exists at the API boundary and is
    not something the reader has to infer from prose.
    """
    return Tier.UNVERIFIED in grounding(passages, used)
