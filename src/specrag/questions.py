"""Load and validate the answer key.

The question set is data, not code — it lives in evals/questions.yaml so that
someone who does not write Python can add to it, and so a changed label shows
up in a diff. This module is the gate between that file and everything that
reads it.

WHY VALIDATE AT ALL, FOR A FILE WE WRITE OURSELVES

Because the file can be silently wrong, and a silently wrong answer key is the
worst failure mode in this project. Day 16 proved it: one question labelled
unanswerable when it was answerable collapsed the measured threshold gap from
0.15 to 0.022 and flipped the day's conclusion. The code was correct
throughout. The key was wrong.

Four mistakes this file makes impossible:

    kind: lookpu                 -> a typo'd kind is an error, not a silently
                                    unmatched category
    expct: ["9.1.10"]            -> a typo'd FIELD NAME is an error too
    kind: unanswerable           -> cannot carry an expected section: if the
      expect: ["4.9"]               answer is not in the document, no section
                                    can contain it
    two questions, one id        -> an error, because results are keyed by id

The point is that every one of those four is a contradiction a human eye skims
straight past, and all four get louder the closer the file gets to 100 entries.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The file sits beside the repo, not inside the package: it is an input to the
# project, like the PDFs, not part of the shipped library.
QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "evals" / "questions.yaml"


class Kind(str, Enum):
    """What sort of question this is — which decides how it is scored.

    Inheriting from `str` as well as `Enum` means `kind == "lookup"` works and
    the value serialises as plain text, which keeps the YAML readable. Closest
    Java analogue is an enum with a `String` field, except here the enum IS
    the string.
    """

    LOOKUP = "lookup"
    CROSS_SECTION = "cross-section"
    FALSE_PREMISE = "false-premise"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"


# Kinds that must name at least one section, and kinds that must name none.
EXPECTS_A_SECTION = {Kind.LOOKUP, Kind.CROSS_SECTION, Kind.FALSE_PREMISE}
EXPECTS_NOTHING = {Kind.UNANSWERABLE, Kind.AMBIGUOUS}


class Question(BaseModel):
    """One labelled question.

    `extra="forbid"` is the important line. Pydantic's default is to IGNORE
    unknown fields, which is the opposite of Jackson's default — so a
    misspelled `expct:` would be silently dropped and the question would
    quietly score as having no expected section. Forbidding extras turns that
    into a loud error at load time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    q: str = Field(min_length=10)
    kind: Kind

    # Which document the expected sections live in. Left out of most entries
    # and filled in from the set's `default_source` — because section numbers
    # are NOT unique across the corpus. "3.7" is "Transaction-related
    # messages" in the core spec and "From", an XML header field, in the SOAP
    # spec. Scoring on the number alone counts the wrong document as a hit,
    # and inflates the score while doing it.
    source: str | None = None

    # Section numbers, compared as PREFIXES. "9.1.10" matches the chunk
    # labelled "9.1.10. HeartbeatInterval (1/2)". Never truncated — Day 16
    # read a correct 9.1.10 as a mismatch by chopping it to 9.1.
    expect: list[str] = []

    # A phrase that must appear in a retrieved passage. Section numbers only
    # work because the OCPP spec is numbered; a client's Confluence export or
    # a folder of PDFs will not be. A distinctive phrase identifies the right
    # passage in ANY corpus, and unlike a chunk id it survives re-chunking.
    # Optional here, because our corpus happens to have numbers.
    expect_contains: list[str] = []

    # Not documentation. This is the field that would have caught the Day 16
    # bug: a label you cannot justify in a sentence is a label you cannot
    # trust. The minimum length exists to stop "obvious" being an answer.
    why: str = Field(min_length=20)

    @property
    def answerable(self) -> bool:
        """Should the system answer this, or refuse it?

        `ambiguous` counts as answerable: the corpus does mention the topic,
        so a refusal is not obviously right either. That is exactly why these
        are reported and not scored.
        """
        return self.kind is not Kind.UNANSWERABLE

    def coverage(self, rows: list[dict]) -> tuple[int, int]:
        """How many of the expected sections are among these passages.

        One metric for both shapes of question. A `lookup` expects one section
        and scores 0/1 or 1/1 — the binary case falls out of the general one.
        A `cross-section` question expecting five scores 3/5, which says far
        more than "failed" does.

        This is why `transactional-events` now names five sections instead of
        one. A complete answer has to walk the whole charging session:
        Authorize, Start, MeterValues, Stop, plus the overview. Labelled with
        3.7 alone, a partial answer looked complete.

        Note what full coverage demands: five sections out of five retrieved
        passages means every slot has to be exactly right. Expect these to
        score poorly at RETRIEVE=5, and expect that to be the honest result
        rather than a bug.
        """
        if not self.expect:
            return (0, 0)

        seen = set()
        for row in rows:
            if self.source and row.get("source") != self.source:
                continue                      # right number, wrong document
            label = (row.get("section") or "").split()[0].rstrip(".")
            for prefix in self.expect:
                # Compared as a prefix, never truncated: Day 16 read a correct
                # 9.1.10 as a mismatch by chopping it to 9.1.
                if label.startswith(prefix):
                    seen.add(prefix)
        return (len(seen), len(self.expect))

    @property
    def scored(self) -> bool:
        """Does this question count towards the headline numbers?

        Ambiguous ones do not. The corpus mentions the topic without
        specifying it, so neither answering nor refusing is clearly correct,
        and grading it either way would put a number on a coin toss. They are
        still run and still printed — dropping them loses the information,
        which is what the previous answer key did.
        """
        return self.kind is not Kind.AMBIGUOUS

    @model_validator(mode="after")
    def sections_match_kind(self) -> "Question":
        if self.kind in EXPECTS_A_SECTION and not (self.expect or self.expect_contains):
            raise ValueError(
                f"{self.id!r}: kind {self.kind.value!r} must name an expected "
                f"section in `expect` or a phrase in `expect_contains` — "
                f"otherwise there is nothing to score retrieval against"
            )
        if self.kind in EXPECTS_NOTHING and self.expect:
            raise ValueError(
                f"{self.id!r}: kind {self.kind.value!r} cannot have `expect` "
                f"{self.expect} — if the answer is not specified in the "
                f"document, no section can be expected to contain it"
            )
        return self


class QuestionSet(BaseModel):
    """The whole file."""

    model_config = ConfigDict(extra="forbid")

    version: int
    default_source: str
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def resolve_sources(self) -> "QuestionSet":
        """Fill in the document name wherever a question left it out.

        Done once, here, so nothing downstream has to remember to do it. A
        question that forgets its source would otherwise score against every
        PDF in the corpus at once.
        """
        for question in self.questions:
            if question.source is None:
                question.source = self.default_source
        return self

    @model_validator(mode="after")
    def ids_are_unique(self) -> "QuestionSet":
        seen: set[str] = set()
        for question in self.questions:
            if question.id in seen:
                raise ValueError(
                    f"duplicate id {question.id!r} — results are keyed by id, "
                    f"so a repeat would overwrite the other question's score"
                )
            seen.add(question.id)
        return self

    def of_kind(self, *kinds: Kind) -> list[Question]:
        return [q for q in self.questions if q.kind in kinds]

    @property
    def scored(self) -> list[Question]:
        return [q for q in self.questions if q.scored]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.kind.value] = counts.get(question.kind.value, 0) + 1
        parts = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
        return (f"{len(self.questions)} questions ({parts}); "
                f"{len(self.scored)} scored")


@lru_cache(maxsize=1)
def load(path: Path = QUESTIONS_PATH) -> QuestionSet:
    """Read, parse and validate the answer key.

    Cached because every script reads it and the file does not change inside
    a run. `lru_cache` is roughly `@Cacheable` with no eviction policy — fine
    here, wrong for anything whose underlying data moves.

    A failure here is deliberately fatal. A partly-valid answer key is worse
    than none: it produces a number, and the number is wrong.
    """
    if not path.exists():
        raise FileNotFoundError(f"answer key not found at {path}")
    raw = yaml.safe_load(path.read_text())
    return QuestionSet.model_validate(raw)
