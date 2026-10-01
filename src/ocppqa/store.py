"""Storing and searching chunks."""

from typing import Any

from .db import execute, executemany, query
from .embeddings import embed, embed_one


# How many texts to embed per API call. 661 separate calls would be 661
# round trips for no reason; one call with 661 texts risks the request-size
# limit. 100 is comfortably inside both.
BATCH = 100


def add(records: list[dict], progress: bool = False) -> int:
    """Store chunks. Each needs `text` and `source`; page/section optional.

    Embeds in batches and inserts in batches. Both matter at 661 rows: one
    API call per chunk is 661 round trips, and one INSERT per chunk is 661
    transactions.
    """
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
            )
            for r, v in zip(batch, vectors, strict=True)
        ]
        executemany(
            """
            INSERT INTO chunks (source, page, section, text, tokens, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
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


def count() -> int:
    return query("SELECT count(*) AS n FROM chunks")[0]["n"]


def clear() -> None:
    execute("TRUNCATE chunks RESTART IDENTITY")
