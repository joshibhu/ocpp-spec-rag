-- The chunk store.
--
-- One row per chunk, one vector per row. The vector is ALWAYS 1536 numbers
-- however long the text is — which is why a chunk holding five ideas
-- represents all five badly, and a chunk holding one represents it sharply.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
    id         SERIAL PRIMARY KEY,

    -- Provenance. Without these, retrieval finds text you cannot attribute,
    -- and an answer you cannot cite is a claim rather than a fact.
    source     TEXT    NOT NULL,          -- which document
    page       INTEGER,                   -- which page
    section    TEXT,                      -- e.g. "4.5" — filled in on Day 16

    text       TEXT    NOT NULL,
    tokens     INTEGER,                   -- roughly, for cost accounting

    -- 1536 = the size text-embedding-3-small produces. The column type is
    -- fixed-width, so changing embedding model means rebuilding this table.
    embedding  vector(1536) NOT NULL
);

-- Without an index, finding the nearest chunk compares the question to EVERY
-- row. Fine at 300 rows, hopeless at 300,000.
--
-- HNSW is an approximate index: it trades a small chance of missing the true
-- nearest neighbour for a large speed gain. `vector_cosine_ops` must match
-- the distance operator used in queries (<=> for cosine).
CREATE INDEX chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX chunks_source_idx ON chunks (source);
