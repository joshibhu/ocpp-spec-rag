-- When was this chunk generated?
--
-- Added on Day 17, after an answer was traced back to a chunk and the first
-- question was "is this chunk even from the current chunker?" There was no way
-- to tell. Day 11 had the same problem in another form: an MCP server quietly
-- serving three-day-old code.
--
-- timestamptz, not timestamp. A bare `timestamp` in Postgres stores no offset,
-- so it is a wall-clock reading with no idea which wall — and the first time
-- the container runs in UTC while you read it in IST, every number is wrong by
-- 5.5 hours with nothing to indicate it. timestamptz stores an absolute
-- instant and renders it in the reader's zone.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

-- Rows that already existed get the time this ran, which is NOT when they were
-- generated. They are about to be replaced by a --fresh ingest anyway, so the
-- lie is short-lived — but it is a lie, and worth knowing about before you
-- trust the column on an old row.

-- Cheap enough to always have: "show me the newest chunks" is the query you
-- run when you suspect the index is stale.
CREATE INDEX IF NOT EXISTS chunks_created_at_idx ON chunks (created_at DESC);
