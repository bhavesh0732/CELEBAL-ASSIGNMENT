"""
Retrieval Module
-----------------
Combines dense vector search (semantic meaning) with a lightweight
keyword overlap score (exact term matches) into a single hybrid
score, then re-ranks the merged candidate pool. This catches cases
where a query uses exact wording ("invoice number") that a purely
semantic search sometimes under-ranks.
"""

import re
from typing import List, Dict
from .embeddings import EmbeddingModel
from .vectorstore import FaissVectorStore

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _keyword_score(query_tokens: set, chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


class HybridRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FaissVectorStore,
        vector_weight: float = 0.75,
        keyword_weight: float = 0.25,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(self, query: str, top_k: int = 4, candidate_pool: int = 12) -> List[Dict]:
        """
        1. Pull a larger candidate pool from FAISS (semantic search).
        2. Score each candidate with a keyword-overlap signal too.
        3. Blend the two scores and return the top_k re-ranked chunks.
        """
        query_embedding = self.embedding_model.encode_query(query)
        candidates = self.vector_store.search(query_embedding, top_k=candidate_pool)

        if not candidates:
            return []

        query_tokens = _tokenize(query)
        # Normalize vector scores (cosine sim already in [-1, 1], typically [0,1] for similar text)
        rescored = []
        for chunk, vec_score in candidates:
            kw_score = _keyword_score(query_tokens, chunk["text"])
            blended = self.vector_weight * vec_score + self.keyword_weight * kw_score
            rescored.append({**chunk, "vector_score": vec_score, "keyword_score": kw_score, "final_score": blended})

        rescored.sort(key=lambda r: r["final_score"], reverse=True)
        return rescored[:top_k]
