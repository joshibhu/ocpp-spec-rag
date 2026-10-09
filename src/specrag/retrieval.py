"""Four ways to fetch passages, so they can be compared.

Day 17 left one clean retrieval failure: "what is in a MeterValues message?"
never retrieved section 6.31, which is titled MeterValues. The passages handed
to the model did not contain the answer, so the model invented a quote. A
retrieval gap, filled by the model.

This module exists to find out whether keyword search is worth keeping, and
the answer is allowed to be no.
"""

import re
from typing import Any, Callable

from .store import keyword_search, search

# ── combining two ranked lists ───────────────────────────────────────────────
#
# The two searches return numbers that cannot be combined:
#
#     cosine distance   0.282    LOWER is better, bounded 0..2
#     ts_rank_cd        1.600    HIGHER is better, no units, no fixed range
#
# Adding them is meaningless and normalising them is guesswork — the ranges
# shift with every query. So both scores are thrown away and only the POSITION
# in each list is used. Position is comparable even when scores are not: "2nd
# by meaning, 1st by wording" says something, "0.282 plus 1.6" does not.
#
# Reciprocal Rank Fusion: each list votes 1/(K + rank) for the chunks it
# returned, and the votes are added.

# K flattens the difference between the top positions. At K=60 the gap between
# rank 1 and rank 2 is small, so a chunk needs to do well in BOTH lists to
# beat one that came first in only one. Lower K makes a single first place
# decisive. 60 is the value the original RRF paper used and a reasonable
# default to question later, not a truth.
RRF_K = 60

# A spec identifier: two or more capitalised words run together, optionally
# dotted — MeterValues, StatusNotification.req, HeartbeatInterval.
IDENTIFIER = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+(?:\.[a-z]+)?\b")


def fuse(lists: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of several ranked lists.

    A chunk appearing at rank 3 in one list and rank 1 in another scores
    1/63 + 1/61. A chunk appearing first in one list and nowhere in the other
    scores 1/61 alone. So agreement between the two searches is what wins,
    which is the property we want: both methods pointing at the same passage
    is stronger evidence than either shouting alone.
    """
    scores: dict[int, float] = {}
    rows: dict[int, dict[str, Any]] = {}

    for ranked in lists:
        for position, row in enumerate(ranked, start=1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (RRF_K + position)
            # Keep the first version of the row we see, but remember both
            # scores where we have them, so the caller can still inspect why.
            rows.setdefault(row["id"], dict(row))
            if "rank" in row:
                rows[row["id"]]["rank"] = row["rank"]
            if "distance" in row:
                rows[row["id"]]["distance"] = row["distance"]

    best = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    out = []
    for chunk_id in best[:limit]:
        row = rows[chunk_id]
        row["rrf"] = scores[chunk_id]
        out.append(row)
    return out


# ── the four strategies ──────────────────────────────────────────────────────


def vector_only(question: str, limit: int = 5) -> list[dict[str, Any]]:
    return search(question, limit=limit)


def keyword_only(question: str, limit: int = 5) -> list[dict[str, Any]]:
    return keyword_search(question, limit=limit)


def hybrid(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Both searches, fused by rank.

    Each side retrieves MORE than `limit` before fusing. Fusing two top-5
    lists can only ever consider 10 chunks; fusing two top-20 lists lets a
    chunk that was 12th by meaning and 2nd by wording surface, which is
    exactly the case this is meant to catch.
    """
    deep = max(limit * 4, 20)
    return fuse([search(question, deep), keyword_search(question, deep)], limit)


def identifier_aware(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Vector search, plus an exact lookup for identifiers in the question.

    The narrow alternative to full hybrid search: most questions are prose and
    need meaning-matching, but a question naming MeterValues or
    StatusNotification.req is telling you exactly what to look up. Less
    machinery than hybrid — and if it scores the same, it is the better
    engineering.
    """
    identifiers = IDENTIFIER.findall(question)
    if not identifiers:
        return search(question, limit=limit)

    deep = max(limit * 4, 20)
    return fuse(
        [search(question, deep), keyword_search(" ".join(identifiers), deep)],
        limit,
    )


STRATEGIES: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "vector": vector_only,
    "keyword": keyword_only,
    "hybrid": hybrid,
    "ident+vec": identifier_aware,
}
