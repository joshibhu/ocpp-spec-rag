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

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from .settings import settings
from .store import search

client = OpenAI(api_key=settings.openai_api_key)

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
class GroundedAnswer(BaseModel):
    """A grounded answer to a question about the OCPP specification."""

    answerable: bool = Field(
        description="True only if the passages below contain the answer."
    )
    answer: str = Field(
        description="The answer, or a short statement of what is missing."
    )
    used: list[int] = Field(
        description="Passage numbers actually used. Empty if not answerable."
    )


SYSTEM = """You answer questions about the OCPP protocol using ONLY the \
numbered passages provided. You are talking to charge point integrators and \
CPO engineers who will act on what you say.

Rules:
1. Use ONLY the passages. Not your own knowledge of OCPP, even if you are \
certain it is correct, and even if the passages are incomplete.
2. If the passages do not contain the answer, set answerable to false and \
say what is missing. This is a correct outcome, not a failure. A wrong \
answer costs an integrator days; "not in this spec" costs them a minute.
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
6. Be brief. Two or three sentences unless the question needs a list."""


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


@dataclass
class Result:
    """The answer, plus everything needed to check it."""

    question: str
    passages: list[dict[str, Any]]   # exactly what we sent, in order
    reply: GroundedAnswer            # exactly what came back

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


def answer(question: str, limit: int = RETRIEVE) -> Result:
    """Retrieve, then answer from what was retrieved."""
    # Retrieve by relevance, then PRESENT in document order. The two are
    # different jobs: distance decides what gets in, the document decides what
    # order it is read in.
    rows = _in_document_order(search(question, limit=limit))

    completion = client.chat.completions.parse(
        model=settings.chat_model,
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
    return Result(question=question, passages=rows, reply=completion.choices[0].message.parsed)
