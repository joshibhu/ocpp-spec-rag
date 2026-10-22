"""Passages in, grounded answer out.

Everything until now was retrieval: the output was text copied verbatim from
the PDF. It could point at the wrong page, but it could not invent. This file
is where the model starts writing sentences of its own — so it is where
fabrication becomes possible for the first time.

THE CENTRAL RULE HERE: the model never writes a citation.

It is shown numbered passages and may only refer to them by number. The
section and page a user eventually sees are looked up from OUR database row,
not from anything the model produced. So the worst it can do is point at the
wrong passage — which is checkable — rather than cite a section that does not
exist, which is not.

A number the model picks can be validated against a list we control.
A string the model writes cannot be validated against anything.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from openai import OpenAI
from pydantic import BaseModel, Field

from .retrieval import identifier_aware
from .usage import Usage
from .settings import settings

# Explicit, because the defaults are wrong for this workload. A single
# evaluation run makes over two hundred calls, and the SDK's default read
# timeout is long enough that one hung connection stalls the run for minutes
# before it even starts retrying. A short timeout with more attempts recovers
# faster than a long one with fewer: four tries at 60s is a worse case of four
# minutes, and it almost never reaches it.
client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=60.0,
    max_retries=4,
)

# How many passages to hand the model. More passages = more chance the answer
# is in there, but also more unrelated text for it to be distracted by, and a
# longer (costlier) prompt. 5 is a starting point to be measured, not a truth.
RETRIEVE = 5


# NOTE: a pydantic docstring becomes the schema's "description", and the
# schema is sent to the model — same lesson as the MCP tool descriptions on
# Day 9. So this docstring is written FOR THE MODEL. Notes to ourselves go in
# comments like this one, which are not transmitted.
#
# Asking for fields rather than prose means the citation arrives as data
# instead of something we have to pull out of a sentence with a regex.
class Premise(str, Enum):
    """What the passages say about what the question takes for granted."""

    HOLDS = "holds"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    SUBJECT_ABSENT = "subject_absent"


# FIELD ORDER IS LOAD-BEARING. Structured output is generated top to bottom,
# so `premise` is emitted before a word of prose exists. Once the model has
# written "contradicted" it cannot then write an answer that confirms the
# question — the verdict constrains the prose, instead of prose being
# summarised afterwards into a verdict it was never checked against.
class GroundedAnswer(BaseModel):
    """A grounded answer to a question about the OCPP specification."""

    premise: Premise = Field(
        description=(
            "What the passages say about what the question assumes. The ONE "
            "place where refusal is decided: only subject_absent is a refusal."
        )
    )
    answer: str = Field(
        description="The answer, or a short statement of what is missing."
    )
    used: list[int] = Field(
        description=(
            "Passage numbers actually used. Empty only when the premise is "
            "subject_absent."
        )
    )

    @property
    def answerable(self) -> bool:
        """Derived from `premise`, never asked for separately.

        It was a second field for three prompt rounds, and in every one of
        them the model set premise to "unsupported" - correctly - and
        answerable to false in the same breath. It was being asked the same
        question twice in two different words, and the word with "answer" in
        it always won.

        Two fields that must agree are one field. A plain @property is not a
        pydantic field, so this is not in the schema the model is sent: there
        is no second opinion to disagree with. (Java: two booleans encoding
        one state, replaced by the enum they were always describing.)
        """
        return self.premise is not Premise.SUBJECT_ABSENT


SYSTEM = """You answer questions about the OCPP protocol using ONLY the \
numbered passages provided. You are talking to charge point integrators and \
CPO engineers who will act on what you say.

Rules:
1. Use ONLY the passages. Not your own knowledge of OCPP, even if you are \
certain it is correct, and even if the passages are incomplete.
2. If the passages do not discuss the SUBJECT of the question at all, set \
premise to subject_absent and say what is missing. This is a correct \
outcome, not a failure. A wrong answer costs an integrator days; "not in \
this spec" costs them a minute.

   subject_absent is the only refusal, and it is about the subject, never \
about the thing the question assumes. If the passages discuss the subject, \
you have an answer - even when the only honest answer is that the \
assumption is wrong. "The spec sets no default; it is a configuration key \
the operator sets" IS an answer. Rule 6 decides which of the other three \
values applies.
3. Do NOT write section numbers, page numbers, or "§" in your answer. Refer \
to passages only as [1], [2]. The reference is added afterwards from our own \
records.
4. Quote the spec's exact wording for anything normative - SHALL, MAY, MUST, \
SHOULD. The difference between SHALL and MAY is the difference between a \
compliance defect and a vendor's choice, so never paraphrase it.
5. Lead with the GENERAL RULE and say how binding it is, before any \
exception or special case. If the passages only cover a special case, say \
that the general rule is not in them.

   This exists because of a real failure. Asked "when MUST a charge point \
send StatusNotification", the answer quoted the section's only sentence \
containing MUST - a rule about message ORDERING - and never mentioned that \
the spec says a Charge Point MAY send one on a status change. Both quotes \
were accurate. The reader would still have concluded the opposite of the \
truth. If the question assumes an obligation the passages do not impose, \
say so plainly.
6. `premise` is the single place where you record what the passages say \
about the question's assumption. Set it to exactly one of:

   - holds: the passages support what the question takes for granted.
   - contradicted: the passages state the opposite. A question saying MUST \
where the passages say MAY, or "required" where the table says Optional, is \
contradicted. A rule that applies only when a configuration key is set to a \
particular value does not make an unconditional MUST true: if the question \
asks when something MUST happen and the passages leave it to the Charge \
Point or to a configuration key, that is contradicted, not holds.
   - unsupported: the question assumes something the passages cover but do \
not establish - a default value, a mandated interval, an obligation that is \
only a configuration key the operator sets.

   A configuration key's own entry in the Standard Configuration Key Names \
and Values chapter is the COMPLETE definition of that key. If you are \
holding that entry and it declares no default value, then the specification \
sets no default. That is an established fact, not missing information: say \
it, and do not send the reader to another chapter to look for a default that \
the key's own definition would have carried.

   - subject_absent: the passages do not discuss the subject at all. This \
is the ONLY value that means you are not answering. Choose it when the \
topic is missing, not when the topic is present and the assumption is wrong.

   A FALSE PREMISE IS NOT A MISSING SUBJECT. holds, contradicted and \
unsupported all mean you ARE answering: open with the correction - what the \
question assumes, then what the passages actually say. Reaching for \
subject_absent there throws away an answer you are already holding, and \
sends the reader off to look for a rule that does not exist.
7. Be brief. Two or three sentences unless the question needs a list."""


def _in_document_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder retrieved passages the way the document reads.

    Search returns them best-match-first, and the model treats whatever is
    labelled [1] as the most important thing it has been given. For a question
    about section 4.9 that meant [1] was piece 5 of 5 and [2] was piece 1 of 5
    - the section read back to front, so the model answered from its closing
    edge cases and never reached the rule at the top.

    Document order costs one sort and no API call. The distances are still
    kept on the rows for us to inspect; they are simply not what decides the
    reading order any more.
    """
    def position(row: dict[str, Any]) -> tuple:
        return (row["source"], row["page"] or 0, row["id"])

    return sorted(rows, key=position)


def _render(rows: list[dict[str, Any]]) -> str:
    """Number the passages for the model.

    The section label and page are deliberately NOT included. If the model can
    see "9.1.10, p.103" it will copy it into its prose, and then we are back to
    trusting a string it wrote. It gets the text and a number, nothing else.
    """
    blocks = []
    for number, row in enumerate(rows, start=1):
        text = " ".join(row["text"].split())
        blocks.append(f"[{number}]\n{text}")
    return "\n\n".join(blocks)


# A retrieval strategy: question and limit in, ranked rows out. Every function
# in retrieval.STRATEGIES satisfies it. (Python's structural equivalent of
# declaring the parameter as a Java functional interface.)
Retriever = Callable[..., list[dict[str, Any]]]


@dataclass
class Result:
    """The answer, plus everything needed to check it."""

    question: str
    passages: list[dict[str, Any]]   # exactly what we sent, in order
    reply: GroundedAnswer            # exactly what came back
    usage: Usage                     # what it cost to get it

    def cited(self) -> list[dict[str, Any]]:
        """Resolve the model's numbers to OUR rows.

        Out-of-range numbers are dropped here and reported in piece 2. A model
        that cites [7] when it was given 5 passages has fabricated a source,
        and we can know that without reading a word of the answer.
        """
        return [
            self.passages[n - 1]
            for n in self.reply.used
            if 1 <= n <= len(self.passages)
        ]


def answer(
    question: str,
    limit: int = RETRIEVE,
    retrieve: Retriever = identifier_aware,
    model: str | None = None,
) -> Result:
    """Retrieve, then answer from what was retrieved.

    `retrieve` is injected so the retrieval strategy can be swapped without
    touching the prompt, and so scripts/evaluate.py can measure the answer
    quality of each one. The default is identifier_aware because it is the
    only strategy that beat plain vector search on the Day 18 retrieval test
    with no question getting worse. See retrieval.py for the numbers.
    """
    # Retrieve by relevance, then PRESENT in document order. The two are
    # different jobs: the strategy decides what gets in, the document decides
    # what order it is read in.
    rows = _in_document_order(retrieve(question, limit=limit))

    # `model` is injected for the same reason `retrieve` is: Day 18 proved a
    # strategy swap is only trustworthy when nothing else moves, and the same
    # holds for a model swap. Defaulting to settings keeps every existing
    # caller on the configured model.
    chosen = model or settings.chat_model
    started = time.perf_counter()
    completion = client.chat.completions.parse(
        model=chosen,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Passages:\n\n{_render(rows)}\n\nQuestion: {question}",
            },
        ],
        response_format=GroundedAnswer,
        # No temperature. The gpt-5 family only accepts the default, and
        # lowering it would not buy much: the grounding comes from the
        # passages and the instruction, not from sampling settings.
    )
    elapsed = time.perf_counter() - started

    # The SDK can return usage as None. Recording zeros would make a run look
    # free; -1 would poison the arithmetic. Zeros with the real model name is
    # the least misleading of the three, and the token total being implausibly
    # low is itself the signal that something was not reported.
    counts = completion.usage
    return Result(
        question=question,
        passages=rows,
        reply=completion.choices[0].message.parsed,
        usage=Usage(
            model=chosen,
            prompt_tokens=counts.prompt_tokens if counts else 0,
            completion_tokens=counts.completion_tokens if counts else 0,
            seconds=elapsed,
            reasoning_tokens=getattr(
                counts.completion_tokens_details, "reasoning_tokens", 0
            ) if counts and counts.completion_tokens_details else 0,
        ),
    )
