-- Idempotent re-ingestion.
--
-- A fingerprint of (source, section, text). Re-running the pipeline with the
-- same documents produces the same fingerprints, so unchanged chunks are
-- skipped rather than re-embedded.
--
-- The UNIQUE constraint is what enforces it. Checking in Python would work
-- until two ingests ran at once, or until someone wrote a second code path
-- and forgot. Day 11's lesson: put the guarantee where it cannot be bypassed.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fingerprint TEXT;

-- Clear first: existing rows have no fingerprint and the index would reject
-- NULLs duplicating. (Postgres treats NULLs as distinct, so this is belt and
-- braces for a rebuild.)
TRUNCATE chunks RESTART IDENTITY;

ALTER TABLE chunks ALTER COLUMN fingerprint SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS chunks_fingerprint_idx
    ON chunks (fingerprint);
