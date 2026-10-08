"""The labelled questions. The answer key.

SEPARATE FROM THE CODE THAT USES IT, on purpose. Two scripts now read these
labels, and a second copy is how they drift apart.

Day 19 moves this to evals/questions.yaml and grows it to 60-100 questions
labelled by someone who knows OCPP. The structure here is the structure that
file will have: every entry carries its justification, and `expect` names the
section the answer should come from.

Why a file under version control rather than a database table: on Day 16 one
wrong label flipped the entire conclusion about distance thresholds. An answer
key needs more review than the code does — you want to see in a diff when a
label changed, and from what.
"""

# ── the test questions, and WHY each is labelled that way ───────────────────
#
# This format exists because the previous one let me get a label wrong without
# noticing. I filed "what are transactional events?" as unanswerable, assuming
# it meant OCPP 2.0.1's TransactionEvent message. It does not — it means the
# transaction lifecycle, which section 3.7 covers. That single wrong label
# collapsed the measured threshold gap from 0.15 to 0.022 and flipped the
# conclusion.
#
# So every entry now carries its justification, and `expect` names the section
# the answer should come from. A label you cannot justify is a label you
# cannot trust.
#
# THESE LABELS NEED A DOMAIN EXPERT. Checking whether a TERM appears is easy
# (see the `verified` notes). Judging whether a CONCEPT is covered is not —
# "transactional events" never appears in 1.6, yet the question is answerable.

ANSWERABLE = [
    {"q": "what is the heartbeat interval?",
     "expect": "9.1.10",  "why": "HeartbeatInterval configuration key"},
    {"q": "when must a charge point send StatusNotification?",
     "expect": "4.9",     "why": "Status Notification operation"},
    {"q": "how does RemoteStopTransaction work?",
     "expect": "5.12",    "why": "Remote Stop Transaction operation"},
    {"q": "what are the valid values for ChargePointErrorCode?",
     "expect": "7.6",     "why": "the enumeration itself"},
    {"q": "what are transactional events?",
     "expect": "3.7",     "why": "transaction lifecycle: Start, MeterValues, Stop. "
                                 "The WORD 'events' never appears in 1.6 — which is "
                                 "why this scores badly despite being answerable."},
    {"q": "what is in a MeterValues message?",
     "expect": "6.31",    "why": "MeterValues.req field definition"},
]

# Genuinely absent. Each one VERIFIED by a keyword search over the corpus
# returning zero chunks — not assumed from memory.
UNANSWERABLE = [
    {"q": "how does SetVariables work?",
     "why": "OCPP 2.0.1 message. 1.6 uses ChangeConfiguration. "
            "verified: 0 chunks contain 'setvariables'"},
    {"q": "what is a BootReason?",
     "why": "OCPP 2.0.1 field. verified: 0 chunks contain 'bootreason'"},
]

# Deliberately NOT in either list — I could not label them honestly:
#   "how does ISO 15118 plug and charge work?"
#       '15118' appears in 2 chunks, so 1.6 mentions it without specifying it.
#       Partially answerable. An ambiguous case belongs in neither bucket, and
#       pretending otherwise is how a test set lies to you.
