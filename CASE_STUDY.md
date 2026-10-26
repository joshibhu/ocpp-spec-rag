# A question-answering system over the OCPP 1.6 specification, with a measured accuracy report

**What it does.** Answers natural-language questions about the OCPP 1.6
specification by retrieving passages from the published documents and
answering only from what was retrieved. Every answer carries the section and
page it came from, resolved from the database rather than written by the
model.

**Why that last sentence matters.** The model is never allowed to write a
citation. It sees numbered passages and may refer to them only as `[1]`,
`[2]`; the section and page a reader sees are looked up from our own row. The
worst it can do is point at the wrong passage, which is checkable. A section
number it typed itself would not be.

**Who it is for.** Charge point integrators and CPO engineers, where the
difference between SHALL and MAY is the difference between a compliance
defect and a vendor's choice. The prompt requires exact quotation of
normative wording for that reason, and a verifier checks every quote in every
answer back against the passage it claims to come from.

**The specifications themselves are copyright Open Charge Alliance and are
not in this repository.** Nor is any spec prose in the answer key or the run
log: the answer key stores questions and expected *section numbers*, and the
run log stores verdicts and counts. The measurement is reproducible by anyone
with their own copy of the documents.

---

## Measured results

<!-- generated: results -->
Measured on 72 labelled questions (70 that should be answered, 2 that should be refused), 3 runs each, 222 answers. A question counts only if it passes every run; two of three is not a working system.

| | |
|---|---|
| answered when it should | 63/70 (90%) |
| all expected sections retrieved | 47/70 (67%) |
| all expected sections cited | 45/70 (64%) |
| refused when it should | 2/2 (100%) |
| sections retrieved overall | 204/381 (54%) |
| fabrications caught by the verifier | 49 over 222 answers |
| cost | Rs 26.33 per run, Rs 0.12 per answer |
| latency | 28s per answer |

Broken out by question kind, because an average over unlike questions hides the group that matters. The 46 lookups outvote the 14 operational questions three to one:

| kind | n | answered | all sections retrieved | sections |
|---|---|---|---|---|
| cross-section | 14 | 10/14 | 1/14 | 15/61 |
| false-premise | 10 | 7/10 | 4/10 | 11/20 |
| lookup | 46 | 46/46 | 42/46 | 42/46 |

Every run is recorded with the configuration that produced it. A number is comparable to another number only while the measuring setup is frozen, so the configuration is fingerprinted rather than described:

| run | configuration | answered | retrieved | cited | time | cost |
|---|---|---|---|---|---|---|
| `1ab4e34a` | ident+vec, gpt-5-nano, k=5, effort=default | 63/70 | 47/70 | 45/70 | 104 min | Rs 26.33 |
| `2d754cca` | ident+vec, gpt-5-nano, k=5, effort=low | 66/70 | 47/70 | 42/70 | 21 min | Rs 5.84 |

Index: 405 chunks indexed, built 2026-10-07 18:36. Answer key: 74 questions (2 ambiguous, 14 cross-section, 10 false-premise, 46 lookup, 2 unanswerable); 72 scored. Prompt fingerprint `dde45f39`, answer key fingerprint `d7711b79`.
<!-- /generated -->

Two scripts measure the retrieval half independently of the model, and they
agree: 53% of expected sections retrieved, computed two different ways. That
agreement is the only reason either number is quotable.

## What broke, and what it taught

Eleven times across seven days the bug was in the measuring instrument rather
than in the system. Every one of them is in the day-by-day record in
`README.md`; this is the index.

| # | What broke | What it taught |
|---|---|---|
| 1 | A correct section `9.1.10` was scored as a mismatch, because the comparison truncated it to `9.1` | Compare identifiers as prefixes, never by chopping them to a fixed depth |
| 2 | One mislabelled question inverted a conclusion: the code was right and the answer key was wrong | When the answer key *is* the instrument, a typo in it is a silent measurement error, not a bug |
| 3 | A strategy comparison ran against a stale index and a stale server | Print the age of everything you are about to trust, before you trust it |
| 4 | The first retrieved passage was piece 5 of 5, so the model answered a section back to front | Retrieve by relevance, present in document order. They are different jobs |
| 5 | Section numbers are not unique across the corpus — `3.7` exists in four of five documents, and in one of them it is an XML header field | An identifier is only an identifier within its document. Scope every lookup by source |
| 6 | Questions generated from section titles scored well and tested nothing: an exam whose questions were copied off the answer sheet | Who writes the questions decides what you are able to discover |
| 7 | A headline moved 80% → 30% → 39% → 62% → 54% with no pipeline change at all | A metric is comparable only while the measuring setup is frozen, and "setup" includes the answer key |
| 8 | A twenty-minute evaluation run died on one API timeout and took all completed work with it | 222 calls at a 1% failure rate lose the whole run nine times in ten. Tolerate the failure, record it, finish |
| 9 | A cross-section headline of "1 in 14" could not be achieved: two questions expect six sections while retrieval returns five | A headline that is arithmetically impossible is not a headline. State the ceiling |
| 10 | `cross-section 21/54` looked like a regression. The arithmetic showed the old number never summed: 75 × 3 ≠ 203 | One lost call had removed a question from one tally and not the other. Check internal consistency before believing a delta |
| 11 | The tool built to catch unattributable comparisons produced one: its field list was hand-written and predated the field that changed | Derive the list of things that must not move from the definition, never restate it |

Two more from the same seven days that were not instrument bugs, and cost
more than any of the above:

| What broke | What it taught |
|---|---|
| Three planned cost experiments were based on the instinct that a shorter prompt is a cheaper prompt. Measurement showed input tokens are 4–16% of the bill and invisible reasoning is ~89% | Two of the three died unrun, a third was already the cheapest model available — and raising top-k, previously unthinkable on cost grounds, turned out to cost about one rupee per run |
| Three rounds of prompt engineering could not stop the model contradicting itself, because `premise` and `answerable` were two fields encoding one state | Two fields that must agree are one field. Ask once and derive the rest; do not negotiate with the model about consistency you can make structural |

## The diagnosis order, and the one time it was broken on purpose

Retrieval → chunking → prompt → model, cheapest first. Five diagnosed
failures across seven days, and **not one of them was the model's fault.**
Two were retrieval, one was presentation order, one was the response schema,
one was the answer key.

It was broken once, deliberately. Five calls on a stronger model — identical
prompt, identical passages, about two rupees — settled in one shot what a
fourth round of prompt rewriting could not: the stronger model obeyed the
prompt on one failure (so that one was a model limit) and made the identical
mistake on another (so that one was not). **The order is an order for fixing.
When a diagnostic is five calls, running it out of order is cheap; shipping
it as the fix is not.**

## A limit that no amount of work removes

Asked "what is the default value of X", the system refuses, and a stronger
model explained why while getting it wrong:

> You need the part of the specification that declares the default value to
> answer this.

It holds five passages out of 405 chunks. Proving "no default exists anywhere
in this specification" from five passages is proving a negative from a
sample, and retrieval never reports that what it returned was complete.
**Absence does not retrieve.** This is a property of top-k retrieval, not a
bug in the prompt or the model, and it is the kind of thing a buyer should
hear before signing rather than after.

A domain expert *can* answer it, because they know one fact the model does
not: a configuration key's entry in chapter 9.1 is the complete definition of
that key, so a missing Default row means the specification sets none.
Supplying that fact to the prompt is legitimate. Editing the answer key to
stop asking would not be.

## What this is not

**Not deployed.** It runs locally against Postgres with pgvector. There is no
service, no CI, no monitoring, no authentication.

**Answer correctness is not measured — only grounding is.** The verifier
checks that every quote appears in the passage it cites and flags normative
wording that was paraphrased. It cannot tell a correct answer from a
plausible one. An LLM-as-judge pass validated against human labels is the
honest next step and has not been done.

**The refusal gap is open.** Three of ten false-premise questions moved from
never answering to sometimes answering. At a three-run bar that counts as no
improvement, and the remaining obstacle is run-to-run non-determinism rather
than the prompt.

**Two planned experiments were not run** — an errata-source filter and
raising top-k from 5 to 8 — and both still have evidence behind them: three
documented cases of errata passages displacing real ones, and a retrieval
ceiling of 59 of 61 expected cross-section sections at top-5.

**The prices are arithmetic, not a bill.** They come from the table in
`src/specrag/usage.py` and have not been checked against the provider's
pricing page. The ratios would have to be wrong by an order of magnitude to
change any conclusion here.

**`reasoning_effort=low` is measured and not adopted.** It is 4.9x faster and
4.5x cheaper and answers three more questions — and it stops quoting. Altered
quotes fell 74% while unquoted normative statements rose 53%, both from the
same cause. For a reader who will be held to SHALL versus MAY, that trade
lands on the wrong axis.

## Run it

```bash
docker compose up -d                                   # Postgres + pgvector
uv run python scripts/ingest.py                        # your own copy of the specs
uv run python scripts/ask.py "when must a charge point send StatusNotification?"
uv run python scripts/evaluate.py --runs 3             # the numbers above
uv run python scripts/compare_runs.py <id-a> <id-b>    # two runs, if comparable
```

Day-by-day engineering record, including every number that turned out to be
wrong: [`README.md`](README.md).
