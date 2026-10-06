# ocpp-answers

Ask questions about the **OCPP 1.6** specification and get the relevant
passages back — eventually with citations, and with an honest *"that isn't in
these documents"* when it isn't.

Week 3 of a 30-day AI engineering sprint. Weeks 1 and 2 are in
[`ai-sprint`](https://github.com/joshibhu/ai-sprint) and
[`charge-ops-mcp`](https://github.com/joshibhu/charge-ops-mcp).

This README is also the learning log: what was built, the ideas worth keeping,
and — the useful part — what went wrong and what the numbers said.

---

## The corpus is not in this repository

OCPP specifications are **copyright Open Charge Alliance**. Committing them
here would be redistributing someone else's document, so `docs/` is
gitignored.

Download your own copy from [openchargealliance.org](https://openchargealliance.org)
(free, registration required) and put the PDFs in `docs/`. The pipeline reads
whatever it finds there.

What *is* committed: the pipeline, the evaluation questions, and the measured
results. Those are my work.

---

## Quick start

```bash
docker compose up -d                      # Postgres with pgvector, on :5435
cp .env.example .env                      # then add your OPENAI_API_KEY

# put the OCPP PDFs in docs/ first
uv run python scripts/ingest.py --dry     # how many chunks, and what it costs
uv run python scripts/ingest.py           # ~15s, about Rs 0.20

uv run python scripts/ask.py              # search by meaning
uv run python scripts/compare.py          # vector vs keyword, side by side
```

---

## Day 15 — Embeddings and vector search

**Built:** Postgres + pgvector, a deliberately naive chunker, 661 chunks from
217 pages, and a CLI that searches them by meaning.

**Core concepts**

- **An embedding is coordinates for meaning.** A model turns any text into a
  list of 1,536 numbers; similar meanings land close together. Finding the
  right passage becomes nearest-neighbour arithmetic.
- **One vector per chunk, and always the same size.** Three words or four
  hundred — still 1,536 numbers. Which is exactly why chunk size matters: a
  fixed container holding five ideas represents all five badly.
- **It beats keyword search when the words differ.** *"how often does the
  charger check in?"* found *"the Charge Point sends a Heartbeat.req"* with
  **no shared words at all**.
- **`<=>` is cosine distance** and must match the `vector_cosine_ops` the HNSW
  index was built with — use a different operator and Postgres silently
  ignores the index and scans every row.
- **HNSW is an approximate index**: a small chance of missing the true nearest
  neighbour, for a large speed gain. At 661 rows it is irrelevant; at 661,000
  it is the difference between a search and a table scan.
- **Metadata is not optional.** Source and page travel with every chunk,
  because an answer you cannot cite is a claim rather than a fact.
- **Batch the embedding calls.** 661 chunks one at a time is 661 round trips.
  Batches of 100 took 15.6 seconds.
- **Cost before spend.** `--dry` prints the chunk count and the price before
  anything is called. Rs 0.20 here, so it did not matter — the habit does.

**The chunker is bad on purpose.** Fixed 800-character windows, 15% overlap,
no awareness of words or sections. It gives a BASELINE, so Day 16's
structure-aware version has to beat a measured number rather than a feeling.

### What went wrong — five findings

**1. Chunks start mid-word.**

    "otification.req This contains the field definition of the
     FirmwareStatusNotifitacion.req PDU…"

`FirmwareStatusN` is in the previous chunk. *(The `Notifitacion` typo is in
the actual spec.)*

**2. Enumeration tables get split, and the partial list looks complete.**
Asking for `ChargePointErrorCode` values returned a chunk starting at
`OtherError` — everything alphabetically before it was in a different chunk
and never retrieved. Nothing signals that the list is partial. Day 3's silent
wrong answer, in a new place.

**3. Vector search missed an answer that keyword search found instantly.**
*"what is the heartbeat interval?"* returned pages 103 and 40. The answer is
on pages 15 and 20 — `BootNotification.conf(currentTime, heartbeatInterval,
status)`. A plain `ILIKE` found them at once.

Why: those chunks are mostly a JSON example and a sequence diagram, so the
chunk's single vector represents *"an example of a BootNotification response"*
rather than *"the definition of the heartbeat interval"*. **Exact identifiers
get diluted by whatever surrounds them** — and identifiers are precisely what
people search for in a specification. That is the measured case for hybrid
search, not a theoretical one.

**4. Distance cannot tell "no answer" from "found it".**

    in the corpus       0.276   0.311   0.622
    NOT in the corpus           0.526   0.575   0.716
                                 └─── overlap ───┘

A threshold at 0.5 would reject the heartbeat question (answerable) and accept
"what are transactional events?" (a 2.0.1 concept, absent from this 1.6
corpus). Wrong in both directions.

In the five-sentence toy test the gap was clean — 0.43 vs 0.88. **The toy
test lied.** It validated the machinery, which is what it was for, but I had
started treating it as evidence that the approach scales. It was not.

**5. Asked a 2.0.1 question of a 1.6 corpus, and got a confident-looking
answer.** `TransactionEvent` appears in **zero** of 661 chunks;
`StartTransaction` appears in 65. The system returned 1.6 passages with an
ordinary distance and no signal that the concept was absent.

A system that answered *"transactional events are…"* from 1.6 text would be
wrong in the worst way — it would sound right, and someone would build 1.6
semantics believing they had built 2.0.1.

**"Nearest" is not "relevant", and distance alone cannot tell the
difference.** That rewrites the Day 18 plan: the refusal decision has to come
from the model reading the passages, not from a number.

### What bit me

A stray leading space before a docstring, from writing files with shell
heredocs — `IndentationError` on line 1. Cheap, because Python said exactly
what was wrong. Worth contrasting with the week's expensive bugs, none of
which raised anything.

And my own diagnostic was weak: the keyword probe in `compare.py` orders by
page, not relevance, so "keyword also failed" can mean "my two-minute tool is
bad". **Suspect the instrument** — fourth time this sprint.

### The to-do list this produced

| Finding | Fix | Day |
|---|---|---|
| mid-word chunks, split tables | structure-aware chunking on `4.5.` headings | 16 |
| identifiers diluted in surrounding text | hybrid search: vector **and** keyword | 17 |
| distance cannot gate refusal | the model decides, reading the passages | 18 |

---

## Day 16 — Chunk on structure, not on character counts

**Built:** a structure-aware chunker, idempotent re-ingestion, and a test
harness whose labels carry their own justification.

```
Day 15   661 chunks, cut every 800 characters
Day 16   405 chunks, cut on the document's own 231 section headings
```

**The spec told us where the boundaries are**

    4.5.  Firmware Status Notification
    6.4.  BootNotification.conf
    7.6.  ChargePointErrorCode

231 real headings. Median section 466 characters, p90 1,744, and only 10 over
2,400 — so **most sections are already chunk-sized**. The contents page matches
the same pattern, so it is filtered by its dot leaders.

**Core concepts**

- **Generic chunkers are for documents you do not understand.** If the document
  has structure, use what it tells you. Markdown → headers. Code → functions.
  A spec → sections.
- **Put the heading INSIDE the chunk text, not only in metadata.** The heading
  is the most topically dense line in the section, so including it pulls the
  chunk's vector toward the right meaning. This turned out to matter more than
  I expected (see below).
- **A split section keeps its heading on every piece.** A chunk that does not
  say what it is about embeds as if it were about nothing in particular.
- **Section label + page = a real citation.** "§7.6, page 81" is checkable;
  "somewhere in a 116-page PDF" is not.
- **Idempotence belongs in the database.** A `fingerprint` column with a UNIQUE
  index, plus `ON CONFLICT DO NOTHING`. The Python check that skips known
  chunks is an optimisation; the index is what makes duplicates impossible.
  Re-ingest went from 9.0s / Rs 0.16 to **0.0s / Rs 0** on an unchanged corpus.

**All four Day 15 findings improved — and I predicted only two would**

| Finding | Day 15 | Day 16 |
|---|---|---|
| mid-word chunk starts | `"otification.req…"` | starts at the heading |
| enumeration split | started at `OtherError`, mid-list | whole, 1,316 chars, from `ConnectorLockFailure` |
| identifier diluted | heartbeat → p.103, 0.622 | `9.1.10. HeartbeatInterval`, 0.411 |
| answerable vs not | overlapping | *see the correction below* |

The enumeration is the clean illustration: **1,316 characters.** An
800-character window could never have held it whole, however lucky the cut.
The document told us how big the chunk needed to be.

**Where I was wrong, part 1: I called dilution a search problem**

I predicted structure-aware chunking would fix findings 1 and 2 only, and that
the heartbeat failure needed hybrid search — tomorrow's job. It fixed itself.

Why: the identifier was drowned out *because it sat inside a chunk about
something else*. Give it a section-bounded chunk with `HeartbeatInterval` as
the heading and it is no longer buried — it is the subject.

> **A sharp chunk needs less clever search.**

Which weakens the case for Day 17's hybrid search. I would not have learned
that if I had built both changes at once.

**Where I was wrong, part 2: a ground-truth bug, and it flipped the result**

I built a six-question test — three answerable, three not — and measured
whether a distance cut-off could separate them. It appeared to work: a gap of
0.15, a usable threshold around 0.48.

Then Bhuwan told me what he had actually meant by "what are transactional
events?" — not OCPP 2.0.1's `TransactionEvent` message, which I had assumed,
but the transaction lifecycle: Start, MeterValues, Stop. Section **3.7** of
this spec, titled "Transaction-related messages".

**The question was answerable. I had filed it as unanswerable.**

Moving that single label:

    gap 0.15  ->  gap 0.022

A cut-off at 0.48 would **refuse a question the spec can answer.**

Retrieval had been right all along — §3.7 was the top hit for all three
phrasings. I did not recognise the correct answer because I was looking for
the wrong one.

And the reason it scores 0.560 despite being correct: the question says
"events", and OCPP 1.6 never uses that word — it says "messages".

    "what are transactional events?"          0.560
    "which messages make up a transaction?"   0.400
    "transaction related messages"            0.413

Same correct chunk, 0.16 apart. **The distance measures the phrasing at least
as much as whether an answer exists.** That is disqualifying for a threshold.

**What a ground-truth bug looks like:** nothing crashed, the code was correct,
the numbers were correct. One label was wrong, and that was enough to produce
a confident, wrong conclusion. It is the hardest category of bug in this work
because there is nothing to debug.

**So the harness was rebuilt**, and the format is the fix:

- every label carries its justification
- answerable cases name the section the answer should come from
- "absent" labels are **verified** against the corpus, not remembered — which
  immediately caught a second bad label: ISO 15118 *is* mentioned in 2 chunks,
  so it is partially answerable and belongs in neither bucket
- ambiguous cases go in **neither** list, with a comment saying why. Forcing
  one into a bucket is how a test set lies to you.
- the verdict reports a **margin**, and refuses to call 0.022 a separation

**The term/concept distinction** is why this needs a domain expert: a keyword
search can tell you `TransactionEvent` is absent. Only someone who knows OCPP
can tell you the *concept* is covered anyway. **The labels are Bhuwan's job,
not mine** — I have now demonstrated what happens otherwise.

**Two more instrument bugs, found while fixing the first one**

- my section comparison truncated `9.1.10` to `9.1`, reading a correct answer
  as a mismatch
- the verdict printed "SEPARATED" for a margin of 0.022, which would have
  talked us into shipping the threshold anyway

**Fifth and sixth time this sprint the measuring tool was the problem** rather
than the system. That is no longer a coincidence; it is the dominant failure
mode of this kind of work.

**What it changes for Day 18**

The refusal decision cannot be a number. The model has to read the passages
and judge — it can see that §3.7 answers a question about "transactional
events" even though the spec never uses the word. Distance might still serve
as a cheap pre-filter at a generous cut-off (~0.75, to catch the genuinely
absent), but it cannot be the decision.

Third time this plan has changed, each time because of a measurement.

**Run**

```bash
uv run python scripts/ingest.py --fresh       # structure-aware, ~9s, Rs 0.16
uv run python scripts/ingest.py               # idempotent: 0 to embed
uv run python scripts/ingest.py --naive --fresh   # Day 15's chunker, to compare

uv run python scripts/check.py                # the findings, side by side
uv run python scripts/check.py --threshold    # answerable vs not, with margins
uv run python scripts/check.py --repl         # ask your own
```

---

*Days 17–21 to follow.*
