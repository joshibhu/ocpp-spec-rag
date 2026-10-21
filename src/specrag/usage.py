"""What an answer cost: tokens, time, and money.

Day 20 compares five experiments against each other. An accuracy figure with
no price beside it is not a result anybody can act on — "3% better" is a buy
at one price and a refusal at ten times it.

The split here is deliberate. TOKENS AND SECONDS ARE FACTS: the API reports
the first, the clock reports the second, and both are stored. MONEY IS A
LOOKUP, and lookups go stale; a price table committed today is wrong the
first time a provider changes a number, and nothing in the code can tell.
So money is derived on demand, never stored, and PRICES carries the date it
was checked. Every run prints that date, which is the only way a stale figure
becomes visible instead of silently wrong.
"""

from dataclasses import dataclass

# USD per 1M tokens, as (input, output). CHECKED 2026-10-21 — verify against
# the provider's pricing page before quoting any number from a run, and
# update PRICES_CHECKED when you do.
#
# A model missing from this table is not an error. Its tokens and latency are
# still recorded; only the money column goes blank, which is the honest
# output for a price we do not know.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5-nano": (0.05, 0.40),
    "text-embedding-3-small": (0.02, 0.0),
}
PRICES_CHECKED = "2026-10-21"

# One number, in one place. The repo already quotes rupees in ingest.py, and
# two conversion rates in two files is how they drift apart.
USD_TO_INR = 88.0


@dataclass(frozen=True)
class Usage:
    """What a single answer consumed."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    # Recorded separately because it is where the money actually goes, and it
    # is invisible in the answer. On gpt-5-nano a one-line reply to "what is
    # the heartbeat interval?" measured 283 characters of visible text and
    # 3,648 reasoning tokens. At $0.05 in and $0.40 out per 1M, input is 4%
    # of the bill. Shrinking the prompt — fewer passages, smaller chunks — is
    # an ACCURACY experiment, not a cost one, and Day 20 planned it as both.
    reasoning_tokens: int = 0

    @property
    def usd(self) -> float | None:
        """None when the model's price is unknown — not zero.

        Zero would quietly add nothing to a total and make an unpriced
        experiment look free, which is the most expensive kind of wrong.
        """
        price = PRICES.get(self.model)
        if price is None:
            return None
        return (self.prompt_tokens * price[0]
                + self.completion_tokens * price[1]) / 1_000_000

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def total(runs: list[Usage]) -> tuple[int, float, float | None]:
    """Tokens, seconds and money across a run.

    Money is None if ANY answer used an unpriced model. A partial total is
    worse than no total: it reads as complete.
    """
    tokens = sum(u.tokens for u in runs)
    seconds = sum(u.seconds for u in runs)
    if any(u.usd is None for u in runs):
        return tokens, seconds, None
    return tokens, seconds, sum(u.usd for u in runs)


def rupees(usd: float | None) -> str:
    return "—" if usd is None else f"Rs {usd * USD_TO_INR:,.2f}"
