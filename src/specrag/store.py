"""Storing and searching chunks."""

import hashlib

from datetime import datetime, timezone

from typing import Any

from .db import execute, executemany, query
from .embeddings import embed, embed_one
from .trust import tier


# How many texts to embed per API call. 661 separate calls would be 661
# round trips for no reason; one call with 661 texts risks the request-size
# limit. 100 is comfortably inside both.
BATCH = 100


def fingerprint(record: dict) -> str:
    """Stable id for a chunk: hash of what identifies it.

    Includes the TEXT, so re-chunking with different settings produces
    different fingerprints and the new chunks get embedded. Includes source
    and section so identical boilerplate in two places stays two chunks.
    """
    key = f"{record['source']}|{record.get('section') or ''}|{record['text']}"
    return hashlib.sha256(key.encode()).hexdigest()


def existing_fingerprints() -> set[str]:
    return {r["fingerprint"] for r in query("SELECT fingerprint FROM chunks")}


def add(records: list[dict], progress: bool = False) -> int:
    """Store chunks. Each needs `text` and `source`; page/section optional.

    Embeds in batches and inserts in batches. Both matter at 661 rows: one
    API call per chunk is 661 round trips, and one INSERT per chunk is 661
    transactions.
    """
    if not records:
        return 0

    # Skip what is already stored. Re-running the pipeline on unchanged
    # documents should cost nothing — and on a large corpus, re-embedding
    # everything to add one new document is the difference between seconds
    # and an afternoon.
    for r in records:
        r["fingerprint"] = fingerprint(r)
    known = existing_fingerprints()
    fresh = [r for r in records if r["fingerprint"] not in known]

    if progress and len(fresh) < len(records):
        print(f"    {len(records) - len(fresh)} already stored, "
              f"{len(fresh)} to embed", flush=True)
    records = fresh
    if not records:
        return 0

    for start in range(0, len(records), BATCH):
        batch = records[start : start + BATCH]
        vectors = embed([r["text"] for r in batch])

        rows = [
            (
                r["source"], r.get("page"), r.get("section"),
                r["text"], len(r["text"].split()),
                str(v),                    # pgvector accepts the list literal
                r["fingerprint"],
            )
            for r, v in zip(batch, vectors, strict=True)
        ]
        executemany(
            """
            INSERT INTO chunks
                (source, page, section, text, tokens, embedding, fingerprint)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fingerprint) DO NOTHING
            """,
            rows,
        )
        if progress:
            print(f"    {min(start + BATCH, len(records))}/{len(records)}", flush=True)

    return len(records)


def search(question: str, limit: int = 3) -> list[dict[str, Any]]:
    """Find the chunks nearest in meaning to the question.

    `<=>` is pgvector's COSINE DISTANCE operator: 0 means identical
    direction, 1 means unrelated, 2 means opposite. It must match the
    `vector_cosine_ops` the index was built with — use a different operator
    and Postgres silently ignores the index and scans every row.
    """
    vector = str(embed_one(question))
    return query(
        """
        SELECT id, source, page, section, text,
               embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, vector, limit),
    )


# A query word appearing in more than this share of chunks is useless for
# finding anything. "message" is in most of the corpus; "MeterValues" is in
# six chunks. Postgres ranks them equally, so the common ones are dropped
# before they can drown out the rare ones.
#
# This is inverse document frequency in its crudest usable form: not a weight,
# just a cutoff. The proper version is BM25, which is what Elasticsearch
# ranks with and one real reason people run a search engine next to their
# database.
COMMON_LEXEME_SHARE = 0.25

# {D, C, B, A} — the order Postgres expects. The section title is indexed as
# A, the body as D, so a title match counts ten times a body match.
TS_WEIGHTS = "{0.1, 0.2, 0.4, 1.0}"


def _useful_lexemes(question: str) -> list[str]:
    """The question's words, stemmed, with the over-common ones removed.

    Going through to_tsvector rather than splitting the string means the
    question gets the same stemming and stop-word removal the documents got —
    "sending" and "sent" both become "send" on both sides, or neither matches.
    """
    rows = query(
        "SELECT unnest(tsvector_to_array(to_tsvector('english', %s))) AS lex",
        (question,),
    )
    lexemes = [r["lex"] for r in rows]
    if not lexemes:
        return []

    total = count()
    # One index lookup per word. quote_literal makes each lexeme a literal
    # tsquery term, so a lexeme containing a dot ("metervalues.req") is not
    # re-parsed into something else.
    rows = query(
        """
        SELECT l.lex,
               (SELECT count(*) FROM chunks c
                 WHERE c.tsv @@ quote_literal(l.lex)::tsquery) AS ndoc
        FROM unnest(%s::text[]) AS l(lex)
        """,
        (lexemes,),
    )
    keep = [r["lex"] for r in rows if r["ndoc"] <= total * COMMON_LEXEME_SHARE]

    # If every word is common, searching for nothing finds nothing — fall back
    # to all of them and let the ranking do what it can.
    return keep or lexemes


def keyword_search(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find chunks containing the question's WORDS, not its meaning.

    Why this exists, when vector search already works: on Day 17 "what is in a
    MeterValues message?" never retrieved section 6.31, which is titled
    MeterValues. Meaning-search can miss a word printed in the text.

    THREE ATTEMPTS FAILED BEFORE THIS ONE. Each failure is the lesson.

    (1) `to_tsvector('english', text)` cannot find identifiers.

            MeterValues      ->  'metervalu'         (stemmed)
            MeterValues.req  ->  'metervalues.req'   (ONE token — the dot)

        No shared lexeme, so one never finds the other. The PARSER causes
        this, not the stemmer. Fixed by indexing three spellings of every
        chunk — see db/04_fulltext.sql.

    (2) `websearch_to_tsquery` ANDs the words. The question above parses to
        'metervalu' & 'messag', and section 6.31 never says "message" — it
        says PDU. One word the document happens not to use and the match is
        gone. So the words are OR'd instead.

    (3) OR'ing made it worse: the top five became chunks about messages in
        general, because ts_rank has no inverse document frequency and
        "message" is everywhere. Fixed two ways — over-common words are
        dropped before the query is built, and the section title is weighted
        ten times the body.

    The score is returned as `rank`, where HIGHER is better, deliberately not
    as `distance`, where lower is better. Two numbers pointing opposite ways
    under one name is a bug waiting to happen.
    """
    lexemes = _useful_lexemes(question)
    if not lexemes:
        return []

    tsquery = " | ".join("'" + lex.replace("'", "''") + "'" for lex in lexemes)

    return query(
        """
        SELECT id, source, page, section, text,
               ts_rank_cd(%s::float4[], tsv, %s::tsquery) AS rank
        FROM chunks
        WHERE tsv @@ %s::tsquery
        ORDER BY rank DESC
        LIMIT %s
        """,
        (TS_WEIGHTS, tsquery, tsquery, limit),
    )


def index_identity() -> str:
    """What the index IS, with nothing in it that moves on its own.

    index_summary() exists to shout the index's AGE, because six times this
    sprint a stale index was the bug. That makes it exactly wrong for a run
    log: "14.7 days ago" becomes "14.8 days ago" after three hours, so two
    runs over the same unchanged index compare as differently configured -
    and a comparison that reports a change nobody made is as useless as one
    that misses a change somebody did.

    One function, two callers, two incompatible requirements. The fix is two
    functions, not a flag.
    """
    row = query("SELECT count(*) AS n, max(created_at) AS built FROM chunks")[0]
    if not row["n"]:
        return "empty"
    return f"{row['n']} chunks @ {row['built'].isoformat(timespec='seconds')}"


def inventory() -> list[tuple[str, str, int]]:
    """Every document in the index, with its trust tier and chunk count.

    An index is not a single thing you either trust or do not. This is the
    list a client asks for when they ask what the system is answering from,
    and it is the list Day 22 needed to exist before anything could be
    allow-listed.
    """
    rows = query(
        "SELECT source, count(*) AS n FROM chunks GROUP BY source ORDER BY n DESC"
    )
    return [(r["source"], tier(r["source"]).value, r["n"]) for r in rows]


def unverified_chunks() -> tuple[int, int]:
    """(chunks, documents) in the index from outside the allow-list."""
    off = [(s, n) for s, t, n in inventory() if t != "normative"]
    return sum(n for _, n in off), len(off)


def index_summary() -> str:
    """One line describing the index, printed before anything trusts it.

    The age is the point. Six times this sprint the measuring instrument was
    the bug — a stale server, a stale index, a stale assumption — and in every
    case the system looked fine because nothing on screen said how old it was.

    DAY 22 ADDED THE SECOND HALF OF THAT LESSON, and it was our own test
    fixture that taught it. Twenty adversarial chunks planted into this index
    do not announce themselves: a run that died before its cleanup would leave
    them there, every later answer would be drawn from a corpus containing
    fabricated normative text, and the accuracy numbers would move for a
    reason nobody recorded. The index would look perfectly healthy.

    Age and provenance are the same class of fact: something true about the
    corpus that nothing on screen was saying out loud. So the warning lives
    here, in the line every script already prints before it trusts anything.
    """
    row = query("SELECT count(*) AS n, max(created_at) AS built FROM chunks")[0]
    if not row["n"]:
        return "0 chunks indexed — run scripts/ingest.py"

    built = row["built"]
    age = datetime.now(timezone.utc) - built
    minutes = age.total_seconds() / 60
    if minutes < 60:
        ago = f"{minutes:.0f} min ago"
    elif minutes < 60 * 24:
        ago = f"{minutes / 60:.1f} hours ago"
    else:
        ago = f"{minutes / 60 / 24:.1f} days ago"

    local = built.astimezone()
    line = f"{row['n']} chunks indexed, built {local:%Y-%m-%d %H:%M} ({ago})"

    chunks, docs = unverified_chunks()
    if chunks:
        line += (f"\n  WARNING: {chunks} chunks from {docs} UNVERIFIED "
                 f"document(s) — see scripts/attack.py --clean")
    return line


def count() -> int:
    return query("SELECT count(*) AS n FROM chunks")[0]["n"]


def clear() -> None:
    execute("TRUNCATE chunks RESTART IDENTITY")
