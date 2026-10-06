"""Text in, coordinates out.

THE ONLY FILE THAT KNOWS WHERE EMBEDDINGS COME FROM. Swapping OpenAI for a
local model (sentence-transformers, running on this machine, nothing leaving
it) is a change to this file and nothing else.

That seam is not hypothetical: "the documents never leave your machine" is a
real selling point to a client with confidential contracts, and it should
not require touching the pipeline to offer it.
"""

from openai import OpenAI

from .settings import settings

client = OpenAI(api_key=settings.openai_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a BATCH of texts. One API call, many vectors.

    Batched on purpose: embedding 300 chunks one at a time is 300 round
    trips. The API accepts a list and returns them in the same order.
    """
    if not texts:
        return []
    response = client.embeddings.create(
        model=settings.embedding_model, input=texts
    )
    # The API does not guarantee ordering in the response object, so sort by
    # the index it returns rather than trusting the sequence.
    ordered = sorted(response.data, key=lambda d: d.index)
    return [item.embedding for item in ordered]


def embed_one(text: str) -> list[float]:
    """Embed a single text — a question, usually."""
    return embed([text])[0]
