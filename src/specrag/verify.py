"""Check an answer against the passages it was given.

WHAT THIS CAN CHECK, without knowing anything about OCPP:

    - a cited passage number actually exists
    - every quoted sentence appears, word for word, in a passage we sent
    - an answer that claims to be grounded cites something
    - a refusal does not cite anything

WHAT IT CANNOT CHECK, ever:

    whether the answer is the RIGHT answer.

Day 17 produced an answer whose every quote was verbatim and whose every
citation was real, and which still led the reader to the opposite of the
truth, because it quoted a rule about message ordering in reply to a question
about when to send. This file would have passed it.

That boundary is the honest shape of the problem. Verification catches
FABRICATION. Only a labelled test set catches BEING WRONG — which is Day 19,
and which is why those labels have to come from someone who knows the domain.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

# Typographic characters that break exact matching. The spec PDF is full of
# them: "the Charge Point's current status" uses U+2019, not an apostrophe.
#
# Day 3 cost an afternoon to exactly this: a refusal measurement scored 0/3
# everywhere because the models wrote curly apostrophes and the match list
# used straight ones. The code was right and the instrument was broken.
TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'",      # curly single quotes
    "“": '"', "”": '"',      # curly double quotes
    "–": "-", "—": "-",      # en dash, em dash
    " ": " ",                      # non-breaking space
})

# Quoted spans, double OR single, and only long ones.
#
# The first version matched double quotes only, reasoning that single quotes
# are used for field names ('status', 'errorCode') rather than for claims
# about the spec's wording. That reasoning was right and the conclusion was
# wrong: the model alternates between the two delimiters run to run, so
# whether a fabrication got caught depended on a formatting choice. One run of
# this question reported three altered quotes; the next reported none and had
# a fabricated quote in single quotes.
#
# The length floor is what actually excludes field names — no field name is 25
# characters long — so it does the job the delimiter was wrongly doing.
MIN_QUOTE_CHARS = 25
QUOTED = re.compile(
    r'"([^"]{%d,}?)"' % MIN_QUOTE_CHARS      # "…"
    + r"|'([^']{%d,}?)'" % MIN_QUOTE_CHARS   # '…'
)

# Below this similarity, the closest passage is not the source of the quote,
# so showing a diff against it would be noise rather than a hint.
HINT_THRESHOLD = 0.6


def normalise(text: str) -> str:
    """Comparable form: straight quotes, single spaces, lower case.

    Lower case on purpose. A model writing "Shall" for "SHALL" has not changed
    the meaning; a model dropping "not" has. Case-insensitive matching removes
    false alarms without hiding the failure that matters.
    """
    return " ".join(text.translate(TYPOGRAPHY).split()).lower()


@dataclass
class Finding:
    level: str      # "error" = fabrication; "warning" = sloppy but not false
    code: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def add(self, level: str, code: str, detail: str) -> None:
        self.findings.append(Finding(level, code, detail))


def _closest_sentence(quote: str, passages: list[dict[str, Any]]) -> tuple[float, str, str]:
    """The sentence in any passage most like this quote.

    Used only to explain a failure. Showing "you wrote X, the spec says Y" is
    the difference between a report that is actionable and one that just says
    no.
    """
    best = (0.0, "", "")
    for row in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", row["text"]):
            if len(sentence) < MIN_QUOTE_CHARS:
                continue
            ratio = difflib.SequenceMatcher(
                None, quote, normalise(sentence)
            ).ratio()
            if ratio > best[0]:
                best = (ratio, " ".join(sentence.split()), row["section"] or "-")
    return best


def _diff_window(quote: str, nearest: str, width: int = 70) -> tuple[str, str]:
    """Show the two strings AROUND THE FIRST DIFFERENCE, not from the start.

    The first version printed the first 120 characters of each, which were
    identical — the dropped "not" was at character 190, past the cut. A report
    that shows you matching text and calls it a mismatch is worse than no
    report, and this is the seventh time in this sprint that the instrument
    needed fixing before the finding was visible.
    """
    matcher = difflib.SequenceMatcher(None, quote.lower(), nearest.lower())
    for tag, i1, _i2, j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = quote[max(0, i1 - width // 3): i1 + width]
        right = nearest[max(0, j1 - width // 3): j1 + width]
        return f"…{left}…", f"…{right}…"
    return quote[:width], nearest[:width]


def verify(result) -> Report:
    """Check a Result from answer.answer()."""
    report = Report()
    reply = result.reply
    passages = result.passages
    total = len(passages)

    # 1. Does every cited number exist? A model citing [7] when it was given
    #    five passages has invented a source, and we know that without
    #    reading a word of the answer.
    bad = [n for n in reply.used if not 1 <= n <= total]
    if bad:
        report.add("error", "fabricated-source",
                   f"cited passage(s) {bad}, but only {total} were provided")

    cited_rows = result.cited()

    # 2. The agreement between the two fields.
    #
    #    `not bad` guards against double-reporting: cited() drops out-of-range
    #    numbers, so a model citing [9] of 5 leaves cited_rows empty and would
    #    also trip "no-source". One fault, one finding — a report that inflates
    #    its own count is a report you stop reading.
    if reply.answerable and not cited_rows and not bad:
        report.add("error", "no-source",
                   "claims an answer but cites no passage")
    if not reply.answerable and cited_rows:
        report.add("warning", "source-without-answer",
                   "says the answer is absent, yet cites a passage")

    # 3. Is every quotation real?
    #
    #    THIS IS THE CHECK THAT EARNS THE FILE. On Day 17 the model quoted
    #    "...and that do inform the Central System of Charge" when the spec
    #    says "do not inform". One word, meaning inverted, inside quotation
    #    marks. Nobody reading the answer would have caught it.
    answer_text = reply.answer.translate(TYPOGRAPHY)
    # Keep the ORIGINAL wording for the report and the normalised form for
    # matching. Printing the lower-cased version made the report itself look
    # like the thing that had altered the quote.
    # findall returns a tuple per match, one slot per alternative in the
    # pattern; exactly one is non-empty.
    found = [double or single for double, single in QUOTED.findall(answer_text)]
    quotes = [(" ".join(q.split()), normalise(q)) for q in found]

    cited_text = [normalise(r["text"]) for r in cited_rows]
    all_text = [normalise(r["text"]) for r in passages]

    for raw, quote in quotes:
        if any(quote in body for body in cited_text):
            continue
        if any(quote in body for body in all_text):
            # Real spec text, but from a passage it did not cite. The quote is
            # honest; the reference is wrong, which sends the reader to the
            # wrong page.
            report.add("warning", "uncited-quote",
                       f'genuine quote, but from a passage it did not cite: "{raw[:90]}…"')
            continue

        ratio, nearest, section = _closest_sentence(quote, passages)
        if ratio >= HINT_THRESHOLD:
            mine, theirs = _diff_window(raw, nearest)
            report.add("error", "altered-quote",
                       f"answer:        {mine}\n"
                       f"spec ({section}): {theirs}")
        else:
            report.add("error", "unverified-quote",
                       f'quoted text is in no passage we sent: "{raw[:120]}"')

    # 4. Did it quote anything at all for a normative claim? The prompt asks
    #    for exact wording on SHALL/MAY/MUST/SHOULD, and an unquoted normative
    #    claim is a paraphrase of a rule — the one thing rule 4 forbids.
    if reply.answerable and not quotes:
        normative = re.search(r"\b(shall|must|may|should)\b", reply.answer, re.I)
        if normative:
            report.add("warning", "unquoted-normative",
                       f'states "{normative.group(0)}" without quoting the spec')

    return report
