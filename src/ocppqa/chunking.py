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
