"""Store every evaluation run with the configuration that produced it.

Day 19 produced five different headline numbers from one unchanged pipeline:
80%, 30%, 39%, 62%, 54%. Not one of those moves was a code change. They came
from relabelling a question and from growing the question set, and the only
reason that was discoverable is that each number happened to still be on
screen when the next one appeared.

Day 20 runs five experiments. Scrollback will not survive that, and
remembering which number came from which configuration is exactly the kind
of thing that feels reliable and is not. So a run is not finished until it is
written down next to the configuration that produced it.

WHAT MAKES TWO RUNS COMPARABLE is the thing this module actually exists for.
A number is only comparable to another number while the measuring setup is
frozen, and "frozen" has to include things nobody thinks of as inputs: the
prompt text, the response schema, the answer key. Day 19 broke comparability
three times without touching a line of pipeline code.

So the configuration carries FINGERPRINTS, not version numbers. A version
number is a thing a human remembers to increment, which means it is a thing a
human forgets to increment, and a forgotten bump makes two incomparable runs
look comparable - worse than no record at all. A hash of the prompt and the
schema cannot be forgotten.

WHAT IS DELIBERATELY NOT STORED: the model's answer text. It quotes the
specification, and this file is committed to a public repository. The OCPP
specifications are copyright Open Charge Alliance. Verdicts, counts and
section numbers carry the measurement; the prose would carry the copyright.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_PATH = Path(__file__).resolve().parents[2] / "evals" / "runs.jsonl"

# JSON Lines, appended, never rewritten. One run is one line, so two runs
# recorded from different branches merge without a conflict, and a truncated
# write costs the last run instead of the file. (The Java instinct here would
# be a table; a table needs a migration every time a column is added, and the
# columns of an experiment log are not known in advance.)


def fingerprint(*parts: str) -> str:
    """Eight hex characters standing in for everything that must not move.

    Short on purpose: it goes in a printed comparison table and it is only
    ever tested for equality, never for order or distance. Eight characters
    collide once in four billion, and a collision would merely let two runs
    be compared that should not have been - the same risk the alternative
    (no fingerprint at all) carries every single time.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")      # so ("ab","c") and ("a","bc") differ
    return digest.hexdigest()[:8]


@dataclass(frozen=True)
class Config:
    """Everything that can move the number, in one object.

    If a field is missing from here, two runs that differ in it will look
    identical in the log, and the difference between them will be attributed
    to whichever change was being tested at the time. That is the failure
    this whole module is built to prevent, so the bar for leaving something
    out is that it provably cannot affect the result.
    """

    strategy: str
    model: str
    top_k: int
    runs: int
    effort: str          # "" means the provider's default was not overridden
    prompt: str          # fingerprint of SYSTEM + the response schema
    questions: str       # fingerprint of the answer key file
    index: str           # chunk count and build date
    note: str = ""       # what this run was testing, in a few words


@dataclass
class RunRecord:
    """One evaluation run: its configuration, its totals, its details."""

    config: Config
    totals: dict[str, Any]
    per_kind: dict[str, Any]
    per_question: list[dict[str, Any]] = field(default_factory=list)
    started: str = ""
    seconds: float = 0.0
    usd: float | None = None
    calls: int = 0

    def line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def record(run: RunRecord, path: Path = RUNS_PATH) -> str:
    """Append the run and return the id it can be referred to by.

    The id is a fingerprint of the configuration plus the start time, so two
    runs of the same configuration are distinguishable (generation is not
    deterministic, so repeating a configuration is a legitimate thing to do)
    while the id still changes the moment anything in the setup does.
    """
    run_id = fingerprint(run.config.prompt, run.config.questions,
                         run.config.strategy, run.config.model,
                         str(run.config.top_k), run.started)
    run.totals = {"id": run_id, **run.totals}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(run.line() + "\n")
    return run_id


# Fields added to Config after runs had already been written. An append-only
# log is read by code newer than the code that wrote it, so the reader has to
# tolerate lines from before a field existed - the one migration a table would
# have forced and a file quietly does not.
#
# The value here must mean "not set", never the value a later run happened to
# use. "" is honest for effort: those runs really did use the provider's
# default. Backfilling a guess would make old runs look like they were
# configured the new way, which is the exact failure fingerprints exist to
# prevent, arriving through the back door.
LEGACY_DEFAULTS: dict[str, Any] = {"effort": ""}

# Runs recorded before index_identity() existed stored index_summary()'s
# prose, which carries a RELATIVE age - "14.7 days ago". Frozen in a log and
# reprinted later, that is a stale number presenting itself as a current one,
# which is instrument bug #2 arriving through a side door: it was true when
# written and nothing on the page says when that was.
#
# Normalised on read rather than rewritten in the log, so append-only holds.
# It lives here rather than in the script that first needed it because two
# readers need it now, and restating a rule in its second caller is exactly
# the duplication that made bug #11.
_AGE = re.compile(r"\s*\([^)]*ago\)")


def stable(name: str, value: object) -> object:
    """A stored config value with its self-moving parts removed."""
    if name == "index" and isinstance(value, str):
        return _AGE.sub("", value)
    return value


def load(path: Path = RUNS_PATH) -> list[RunRecord]:
    """Every recorded run, oldest first. Missing file means no runs yet."""
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        data = json.loads(raw)
        config = {**LEGACY_DEFAULTS, **data.pop("config")}
        out.append(RunRecord(config=Config(**config), **data))
    return out


def now() -> str:
    """UTC, to the second. Local time in a log that outlives the laptop is a
    number you cannot subtract from another number."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rupees_or_unpriced(spend: float, unpriced: int) -> str:
    """Spend, or an honest refusal to state one.

    usage.Usage.usd returns None for a model that is not in the price table,
    and a run that mixed a priced model with an unpriced one has no total at
    all - only a floor. Printing the floor as if it were the total is the
    mistake usage.py already refuses to make at the single-call level, so it
    is not reintroduced here at the run level.
    """
    from .usage import rupees

    if unpriced:
        return f"cost unknown — {unpriced} calls on an unpriced model"
    return rupees(spend)
