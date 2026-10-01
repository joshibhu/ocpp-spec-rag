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

*Days 16–21 to follow.*
