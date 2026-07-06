"""
Text Chunking Module
---------------------
Splits long documents into overlapping, word-based sliding-window
chunks. Overlap keeps context from being severed mid-idea at a
chunk boundary, which noticeably improves retrieval quality.
"""

from typing import List, Dict


def sliding_window_chunk(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[str]:
    """
    Split `text` into word-based chunks of `chunk_size` words each,
    stepping forward by (chunk_size - overlap) words every iteration
    so consecutive chunks share `overlap` words of context.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Dict]:
    """
    Turn a list of loaded documents into a flat list of chunk records.
    Each chunk keeps a back-reference to its source document and its
    position, which is what lets the UI show "grounded in doc X, chunk Y".
    """
    all_chunks = []
    for doc in documents:
        pieces = sliding_window_chunk(doc["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['source']}::chunk_{i}",
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": piece,
                }
            )
    return all_chunks
