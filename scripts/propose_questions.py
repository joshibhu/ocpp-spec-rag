"""Propose candidate questions from the spec's own table of contents.

    uv run python scripts/propose_questions.py            # 50 candidates
    uv run python scripts/propose_questions.py -n 80
    uv run python scripts/propose_questions.py >> evals/questions.yaml

WHY THIS IS NOT ME WRITING THE ANSWER KEY

A section titled "7.6. ChargePointErrorCode" tells you that a question about
ChargePointErrorCode is answered in 7.6. That is a fact about the document's
structure, not a judgement about its content — so this script can produce the
question and the expected section WITHOUT knowing the answer, and without me
being both author and examiner.

WHAT THIS CANNOT PRODUCE, AND WHY IT MATTERS

Every candidate here is `kind: lookup`: one section, named in its own title,
the easiest shape of question there is. A set of 80 of these would measure one
thing well and everything else not at all. The categories that actually hurt
cannot come from a table of contents:

    cross-section   requires knowing which sections belong together
    false-premise   requires knowing what engineers wrongly assume
    ambiguous       requires knowing the spec is vague on a point

Those need a domain expert. This script exists to make the boring half cheap,
so the expensive half gets the attention.

EVERY CANDIDATE IS A DRAFT. Review each one and delete the ones that are not
real questions. A plausible-looking question whose expected section is wrong
is worse than no question at all — Day 16 lost a day to exactly that.
"""

import random
import re
import sys

from specrag.db import query
from specrag.questions import load

CORE = "ocpp-1.6-edition-2.pdf"

# Section titles that name no subject, so no question can be built from them.
GENERIC = re.compile(
    r"^(introduction|general|scope|references|terminology|definitions?|"
    r"overview|abbreviations|conventions|contents|appendix|glossary|"
    r"document history|copyright|disclaimer)\b",
    re.I,
)

# Errata-sheet style headings leak section numbers of their own; they are not
# subjects. "4.10. Page 47, section: 4.10: Typo description in stop transaction"
ERRATUM = re.compile(r"\bpage\s+\d+|section:\s*\d|\btypo\b", re.I)


def question_for(chapter: str, name: str, shape: str = "") -> str | None:
    """Turn a section title into a question, by chapter.

    The chapter decides the shape because OCPP 1.6 is organised by kind:
    6 is message definitions, 7 is datatypes and enumerations, 9 is
    configuration keys, 4 and 5 are operations. One template each, rather
    than one generic template that reads like a form letter.
    """
    if chapter == "6":                      # MeterValues.req, Authorize.conf
        if name.endswith(".req"):
            return f"what fields are in a {name}?"
        if name.endswith(".conf"):
            return f"what does a {name} contain?"
        return f"what is {name} used for?"
    if chapter == "7":
        # Chapter 7 holds BOTH enumerations and classes, and asking for the
        # "valid values" of a class is a question with no answer — IdTagInfo
        # has fields, not values. The spec says which it is, in the sentence
        # after the title: "7.27. IdTagInfo Class" / "7.38.
        # RegistrationStatus Enumeration". So the document is read rather
        # than assumed. Anything unlabelled is skipped, not guessed.
        if shape == "enumeration":
            return f"what are the valid values for {name}?"
        if shape == "class":
            return f"what fields does {name} contain?"
        return None
    if chapter == "9":                      # configuration keys
        return f"what does the {name} configuration key control?"
    # "how does X work?" only reads as English when X is a NAME. Section
    # titles in chapters 3-5 are sometimes whole clauses — "Transactions
    # before being accepted by a Central System" — and "how does Transactions
    # before being accepted by a Central System work?" is not a question
    # anyone would type. Long titles get the neutral template instead.
    if len(name.split()) <= 3 and chapter in {"4", "5"}:
        return f"how does {name} work in OCPP 1.6?"
    if chapter in {"3", "4", "5"}:
        # NOT lowercased: "Parent idTag" became "parent idtag", and idTag is
        # camelCase in OCPP. Destroying the case of an identifier changes what
        # the question is asking about.
        return f"what does OCPP 1.6 say about {name}?"
    return None


def candidates() -> list[tuple[str, str, str, str]]:
    """(section number, title, chapter, shape) per usable core-spec section.

    `shape` is "class", "enumeration" or "" — taken from the spec's own
    wording immediately after the heading, not inferred from the name.
    """
    rows = query(
        """
        SELECT regexp_replace(section, ' \\(\\d+/\\d+\\)$', '') AS label,
               min(left(text, 400)) AS head
        FROM chunks
        WHERE source = %s AND section IS NOT NULL
        GROUP BY label
        """,
        (CORE,),
    )

    out = []
    for row in rows:
        label, head = row["label"], row["head"]
        shape = ""
        if re.search(r"\bEnumeration\b", head):
            shape = "enumeration"
        elif re.search(r"\bClass\b", head):
            shape = "class"
        number, _, title = label.partition(" ")
        number = number.rstrip(".")
        # Some headings end in a full stop — "Setting a charging profile
        # during a transaction." — which produced "...transaction.?".
        title = title.strip().rstrip(".")
        if not title or GENERIC.match(title) or ERRATUM.search(title):
            continue
        # A number like "6.31" has chapter 6; "9.1.10" has chapter 9.
        out.append((number, title, number.split(".")[0], shape))
    return sorted(out, key=lambda row: [int(p) for p in row[0].split(".")])


def main() -> None:
    wanted = 50
    if "-n" in sys.argv:
        wanted = int(sys.argv[sys.argv.index("-n") + 1])

    already = {s for q in load().questions for s in q.expect}
    pool = [c for c in candidates()
            if c[0] not in already and question_for(c[2], c[1], c[3])]

    # The generator can produce the same question twice: "Data Transfer" is
    # section 4.3 when the Charge Point starts it and 5.6 when the Central
    # System does. One question, two expected sections — which is a
    # cross-section question, not two lookups, and not something this script
    # should decide. Duplicates are dropped and listed for a human to merge.
    seen: dict[str, str] = {}
    unique, collisions = [], []
    for row in pool:
        text = question_for(row[2], row[1], row[3]) or ""
        if text in seen:
            collisions.append((text, seen[text], row[0]))
            continue
        seen[text] = row[0]
        unique.append(row)
    pool = unique

    # Stratify by chapter rather than taking the first N, which would be 50
    # questions about chapter 3 and nothing about configuration keys. Seeded,
    # so re-running gives the same set and a reviewed candidate does not
    # silently turn into a different one.
    by_chapter: dict[str, list] = {}
    for row in pool:
        by_chapter.setdefault(row[2], []).append(row)

    random.Random(19).shuffle(pool)
    picked: list = []
    chapters = sorted(by_chapter)
    while len(picked) < wanted and any(by_chapter.values()):
        for chapter in chapters:
            if by_chapter[chapter] and len(picked) < wanted:
                picked.append(by_chapter[chapter].pop(
                    random.Random(19 + len(picked)).randrange(len(by_chapter[chapter]))
                ))

    picked.sort(key=lambda row: [int(p) for p in row[0].split(".")])

    print(f"# {len(picked)} CANDIDATES — drafts, not labels. Review every one.")
    print("#")
    print("# Generated from section titles by scripts/propose_questions.py.")
    print("# All are kind: lookup, which is the easiest shape of question there")
    print("# is. Delete any that is not a real question. Change `kind` where the")
    print("# answer actually needs more than the one section named.")
    print("#")
    print(f"# Chapters covered: {', '.join(sorted({row[2] for row in picked}))}")
    print()
    if collisions:
        print("# ONE QUESTION, SEVERAL SECTIONS — merge these by hand:")
        for text, first, second in collisions:
            print(f"#   {text}")
            print(f"#     -> expect: [\"{first}\", \"{second}\"]  kind: cross-section")
        print()

    for number, title, chapter, shape in picked:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].rstrip("-")
        print(f"  - id: {slug}")
        print(f"    q: {question_for(chapter, title, shape)}")
        print("    kind: lookup")
        print(f'    expect: ["{number}"]')
        print(f"    why: >\n      The section is titled {title!r}, which names")
        print("      the subject of the question directly. Generated candidate,")
        print("      kind not yet reviewed by a domain expert.")
        print()


if __name__ == "__main__":
    main()
