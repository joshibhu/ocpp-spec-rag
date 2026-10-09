-- Keyword search, alongside the vector search.
--
-- WHY: on Day 17 "what is in a MeterValues message?" never retrieved section
-- 6.31, which is literally titled MeterValues. The passages handed to the
-- model did not contain the answer, so the model quoted something that exists
-- in no passage at all. A retrieval gap, filled by the model.
--
-- Embeddings are blind in two ways that matter for a spec:
--   1. an identifier like MeterValues carries meaning to an engineer and
--      almost none to a language model's geometry
--   2. a table of status transitions -- "A Available A2 A3 A4 B1" -- has no
--      semantic content to embed AT ALL, so no amount of better chunking
--      makes it findable by meaning

-- THE FIRST ATTEMPT AT THIS FAILED, and the failure is the lesson.
--
-- Indexing to_tsvector('english', text) looked obviously correct. It is not:
--
--     MeterValues        ->  'metervalu'          (stemmed)
--     MeterValues.req    ->  'metervalues.req'    (ONE token -- the dot)
--     StatusNotification ->  'statusnotif'
--
--     to_tsvector('english','The MeterValues.req PDU')
--         @@ websearch_to_tsquery('english','MeterValues')   ->  FALSE
--
-- A keyword search for MeterValues misses every mention of MeterValues.req.
-- The cause is the PARSER, not the stemmer: Postgres sees a dotted token as a
-- single unit (it classifies it much like a filename), so the two forms never
-- share a lexeme. A spec is nothing but dotted identifiers, so the obvious
-- configuration is the wrong one for this corpus.
--
-- The fix is to index SEVERAL SPELLINGS of the same text: the original, plus a
-- version with punctuation turned into spaces, plus camelCase split apart. All
-- three go into one tsvector, so any of these finds the row:
--
--     MeterValues      MeterValues.req      "meter values"
--
-- The cost is a tsvector roughly twice the size. At 405 rows that is nothing;
-- the reason to know about it is that at 4 million rows it is the whole budget.
DROP INDEX IF EXISTS chunks_tsv_idx;
ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;

-- AND the SECTION TITLE is weighted above the body.
--
-- Second failure: with the words OR'd together, "what is in a MeterValues
-- message?" returned five chunks about messages in general, because "message"
-- appears everywhere and ts_rank has no inverse document frequency -- a match
-- on a word in 300 chunks counts the same as a word in 6.
--
-- Weighting fixes half of that. An engineer looking something up in a spec
-- reads the HEADINGS first, and a heading is a 25-character summary written by
-- the authors. "MeterValues.req" in a title should beat "message" in a body.
--
-- Postgres gives four weight labels, A to D. Here: title = A, body = D, and
-- the query passes a weights array so A counts ten times what D does.
ALTER TABLE chunks
    ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (
        setweight(
            to_tsvector(
                'english',
                coalesce(section, '')
                || ' ' ||
                regexp_replace(coalesce(section, ''), '[._/\-]+', ' ', 'g')
                || ' ' ||
                regexp_replace(coalesce(section, ''), '([a-z0-9])([A-Z])', '\1 \2', 'g')
            ),
            'A'
        )
        ||
        setweight(
            to_tsvector(
                'english',
                text
                || ' ' ||
                -- MeterValues.req -> MeterValues req
                regexp_replace(text, '[._/\-]+', ' ', 'g')
                || ' ' ||
                -- MeterValues -> Meter Values, so "meter values" matches too
                regexp_replace(text, '([a-z0-9])([A-Z])', '\1 \2', 'g')
            ),
            'D'
        )
    ) STORED;

-- GIN is the inverted index: lexeme -> rows containing it. The same structure
-- Lucene builds, maintained by the database on every write because the column
-- is GENERATED (closest Java analogue: a @Formula field, but stored and
-- indexed).
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
