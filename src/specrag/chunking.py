"""Cutting documents into chunks.

THIS IS THE NAIVE VERSION, ON PURPOSE.

Fixed-size windows over the raw text, with a little overlap. It knows nothing
about sections, sentences, or the difference between SHALL and MAY. It will
cut words in half and put two unrelated message definitions in one chunk.

That is the point: it gives a BASELINE. Day 16's structure-aware chunker has
to beat a measured number rather than a feeling, and the failures this one
produces are the list of things worth fixing.

Keep this function when it is replaced — comparing the two is the experiment.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# Characters, not tokens. Roughly 4 characters per token, so ~200 tokens —
# small enough that one chunk is about one idea, if the boundaries happen to
# fall well. They mostly will not.
CHUNK_CHARS = 800

# Repeat the tail of each chunk at the head of the next, so an answer that
# straddles a boundary appears whole in at least one chunk. 15% is a common
# starting point; it is a parameter to tune, not a truth.
OVERLAP_CHARS = 120


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    section: str | None = None


def read_pages(pdf: Path) -> list[tuple[int, str]]:
    """(page number, text) for each page. 1-indexed, as a human would cite it."""
    reader = PdfReader(str(pdf))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_fixed(text: str, source: str, page: int | None = None) -> list[Chunk]:
    """Slice text into fixed-size overlapping windows.

    No awareness of words, sentences or sections. `range(0, n, stride)` walks
    the string by character count and takes whatever happens to be there.
    """
    stride = CHUNK_CHARS - OVERLAP_CHARS
    chunks = []
    for start in range(0, len(text), stride):
        window = text[start : start + CHUNK_CHARS].strip()
        # Skip scraps: a 40-character tail is page furniture, not content,
        # and it would occupy a row and dilute search results.
        if len(window) >= 100:
            chunks.append(Chunk(text=window, source=source, page=page))
    return chunks


def chunk_pdf(pdf: Path) -> list[Chunk]:
    """Chunk one PDF, page by page.

    Chunking PER PAGE rather than over the whole document is a deliberate
    compromise: it means every chunk has a page number to cite, at the cost
    of forcing a boundary at every page break — including mid-sentence.
    A real loss, traded for citability.
    """
    chunks = []
    for page_number, text in read_pages(pdf):
        chunks.extend(chunk_fixed(text, source=pdf.name, page=page_number))
    return chunks


def chunk_corpus(folder: Path) -> list[Chunk]:
    """Every PDF in a folder."""
    chunks = []
    for pdf in sorted(folder.glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf))
    return chunks


# ── structure-aware chunking (Day 16) ───────────────────────────────────────
#
# The naive chunker above cut every 800 characters and produced: chunks
# starting mid-word, enumerations split in half, and no way to cite anything.
# This one cuts where the DOCUMENT says a topic ends.

import re

# A real heading: "4.5. Firmware Status Notification" or "6.4. BootNotification.conf"
#
# The negative lookahead and the dot checks below exclude TABLE OF CONTENTS
# entries, which match the same shape but carry dot leaders:
#     "4.5. Firmware Status Notification . . . . . . . . . 33"
# Without that filter half the "sections" are contents-page fragments.
HEADING = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\.?\s+(?!\.)(.{2,70}?)\s*$", re.MULTILINE)

# Sections under this are noise — a heading with nothing after it, or a page
# artefact. 231 real sections have a median of 466 chars, so this only drops
# scraps.
MIN_SECTION_CHARS = 120

# Above this, a section is split. Only 10 of 231 exceed it — but section 4.9 is
# 12,000 characters, and those 10 sections are where the damage happened.
MAX_SECTION_CHARS = 2400

# A sentence end: punctuation, then whitespace, then something that can START a
# sentence. The lookahead is what makes this usable on a spec — without it,
# "StatusNotification.req" and "4.9." both split, because the document is full
# of dotted identifiers.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(•])")

# Endings that look like a sentence end and are not. Without this, "e.g. insert
# plug" becomes two sentences.
ABBREV = re.compile(r"(?:e\.g|i\.e|etc|vs|Fig|No|no|Sec|Ch|cf|resp)\.$")


def _page_offsets(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Join pages into one string, and record where each page starts.

    Needed because headings are found by position in the joined text, but a
    citation needs a page number.
    """
    parts, offsets, cursor = [], [], 0
    for number, text in pages:
        offsets.append((cursor, number))
        parts.append(text)
        cursor += len(text) + 1          # +1 for the newline we join with
    return "\n".join(parts), offsets


def _page_at(offset: int, offsets: list[tuple[int, int]]) -> int | None:
    """Which page does this character offset fall on?"""
    page = None
    for start, number in offsets:
        if start <= offset:
            page = number
        else:
            break
    return page


def _is_toc(title: str) -> bool:
    """Contents-page entries carry dot leaders; real headings do not."""
    return ". . ." in title or ".." in title


def _sentences(text: str) -> list[tuple[int, str]]:
    """(offset, sentence) pairs.

    The offset is carried because each piece of a split section has to work out
    which PAGE it starts on. Section 4.9 spans four PDF pages; citing page 43
    for all of it sends the reader to the wrong page.
    """
    starts = [0] + [m.end() for m in SENTENCE_END.finditer(text)]
    out: list[tuple[int, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        sentence = text[start:end].strip()
        if not sentence:
            continue
        if out and ABBREV.search(out[-1][1]):
            out[-1] = (out[-1][0], out[-1][1] + " " + sentence)
        else:
            out.append((start, sentence))
    return out


def _split_long(offset: int, sentence: str, limit: int) -> list[tuple[int, str]]:
    """One "sentence" longer than a whole chunk — in practice a TABLE, which has
    no sentence boundaries at all.

    Cut on whitespace, never inside a word. This is the floor of the design:
    whatever else goes wrong, a chunk can no longer begin "atus change events".
    """
    if len(sentence) <= limit:
        return [(offset, sentence)]

    out: list[tuple[int, str]] = []
    current: list[str] = []
    size = 0
    start = offset
    cursor = offset
    for word in sentence.split():
        if size + len(word) + 1 > limit and current:
            out.append((start, " ".join(current)))
            current, size, start = [], 0, cursor
        current.append(word)
        size += len(word) + 1
        cursor += len(word) + 1
    if current:
        out.append((start, " ".join(current)))
    return out


def _pack(units: list[tuple[int, str]], limit: int, overlap: int) -> list[tuple[int, str]]:
    """Fill each piece with WHOLE sentences, up to the limit.

    The overlap is now whole trailing sentences rather than the last 120
    characters, so the repeated text reads as English instead of starting
    mid-word.
    """
    pieces: list[tuple[int, str]] = []
    current: list[tuple[int, str]] = []
    size = 0

    for unit in units:
        if size + len(unit[1]) + 1 > limit and current:
            pieces.append((current[0][0], " ".join(u[1] for u in current)))
            # Carry trailing sentences into the next piece as the overlap.
            carry: list[tuple[int, str]] = []
            carried = 0
            for previous in reversed(current):
                if carried + len(previous[1]) > overlap:
                    break
                carry.insert(0, previous)
                carried += len(previous[1]) + 1
            # If the carried overlap plus the next sentence would itself
            # overflow, drop the overlap. Otherwise the cap is a suggestion:
            # the first version produced a 2,503-character piece against a
            # 2,400 limit, because the carry was added after the check.
            if carried + len(unit[1]) + 1 > limit:
                carry, carried = [], 0
            current, size = carry, carried
        current.append(unit)
        size += len(unit[1]) + 1

    if current:
        pieces.append((current[0][0], " ".join(u[1] for u in current)))
    return pieces


def chunk_structured(pdf: Path) -> list[Chunk]:
    """Chunk one PDF on its section headings."""
    pages = read_pages(pdf)
    if not pages:
        return []

    text, offsets = _page_offsets(pages)

    headings = [
        (m.start(), m.group(1), m.group(2).strip())
        for m in HEADING.finditer(text)
        if not _is_toc(m.group(2))
    ]
    if not headings:
        # No detectable structure — fall back rather than return nothing.
        # The errata sheets and the JSON/SOAP specs may land here.
        return chunk_pdf(pdf)

    chunks: list[Chunk] = []
    bounds = [h[0] for h in headings] + [len(text)]

    for (start, number, title), end in zip(headings, bounds[1:]):
        body = text[start:end].strip()
        if len(body) < MIN_SECTION_CHARS:
            continue

        label = f"{number}. {title}"

        if len(body) <= MAX_SECTION_CHARS:
            chunks.append(
                Chunk(text=body, source=pdf.name, page=_page_at(start, offsets),
                      section=label)
            )
            continue

        # Too long: split it ON SENTENCE BOUNDARIES, repeating the heading on
        # every piece.
        #
        # The previous version cut every 2,280 characters. On section 4.9 that
        # produced one chunk starting "atus change events" and another ending
        # "...and that do not inform the Central System of Charge" — a rule
        # truncated mid-clause. The model, handed that fragment, tidied it into
        # a finished sentence by dropping the word "not", inverting it. The
        # fabrication was ours to prevent.
        #
        # Room is reserved for the heading prepended below, so a piece cannot
        # exceed the limit once labelled.
        room = MAX_SECTION_CHARS - len(label) - 2

        units: list[tuple[int, str]] = []
        for offset, sentence in _sentences(body):
            units.extend(_split_long(offset, sentence, room))

        pieces = _pack(units, room, OVERLAP_CHARS)
        total = len(pieces)
        for index, (offset, piece) in enumerate(pieces, start=1):
            # Each piece gets the page it ACTUALLY starts on...
            piece_page = _page_at(start + offset, offsets)
            # ...and its own label, so two citations are never identical.
            # Day 17 returned "4.9. Status Notification · p.43" twice, from
            # different chunks, and nothing distinguished them.
            piece_label = f"{label} ({index}/{total})"
            prefixed = piece if index == 1 else f"{label}\n\n{piece}"
            chunks.append(
                Chunk(text=prefixed, source=pdf.name, page=piece_page,
                      section=piece_label)
            )

    return chunks


def chunk_corpus_structured(folder: Path) -> list[Chunk]:
    chunks = []
    for pdf in sorted(folder.glob("*.pdf")):
        chunks.extend(chunk_structured(pdf))
    return chunks
