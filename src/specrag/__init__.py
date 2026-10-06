"""Grounded question answering over the OCPP 1.6 specification.

    chunking    cut the spec on its own section headings
    embeddings  the provider seam — the only file that knows whose model
    store       pgvector: write chunks, search by cosine distance
    db          connection handling
    settings    environment-bound config
"""
