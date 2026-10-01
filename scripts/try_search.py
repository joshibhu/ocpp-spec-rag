"""Play with meaning-search on five hand-written sentences.

    uv run python scripts/try_search.py                    # load + a few questions
    uv run python scripts/try_search.py "your question"    # ask one thing
    uv run python scripts/try_search.py --repl             # keep asking

Nothing here touches the PDFs. The point is to confirm the machinery finds
text by MEANING before pointing it at 70,000 words.
"""

import sys

from ocppqa.store import add, clear, count, search

SENTENCES = [
    "The charge point sends a heartbeat every 30 seconds.",
    "Payment is processed when the charging session ends.",
    "Bananas are yellow when they are ripe.",
    "The connector locks automatically during charging.",
    "Mumbai is on the west coast of India.",
    "A firmware update is downloaded over HTTP and installed on reboot.",
    "The driver must present an RFID card before the session starts.",
]

DEMO_QUESTIONS = [
    "how often does the charger check in?",
    "when do I get billed?",
    "how do I identify myself at the station?",
    "what fruit turns yellow?",
    "what is the capital of France?",      # in NOTHING here — watch the distance
]


def load() -> None:
    clear()
    add([{"text": s, "source": "hand-written"} for s in SENTENCES])
    print(f"loaded {count()} sentences\n")


def ask(question: str, limit: int = 3) -> None:
    print(f"Q: {question}")
    for row in search(question, limit=limit):
        # Distance: 0 = identical meaning, 1 = unrelated, 2 = opposite.
        marker = "  <-- best" if row is search(question, limit=1)[0] else ""
        print(f"   {row['distance']:.3f}  {row['text']}")
    print()


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--repl"]
    repl = "--repl" in sys.argv

    load()

    if args:
        ask(" ".join(args))
    elif not repl:
        for q in DEMO_QUESTIONS:
            ask(q, limit=2)

    if repl:
        print("Ask anything. Blank line to quit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
            ask(q)


if __name__ == "__main__":
    main()
