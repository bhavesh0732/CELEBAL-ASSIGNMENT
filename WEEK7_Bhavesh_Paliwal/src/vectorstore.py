"""
Vector Store Module
---------------------
Thin wrapper around a FAISS flat inner-product index. Flat search is
exact (no approximation error) and is plenty fast for the document
sizes this project targets (a handful of PDFs / notes).
"""

from typing import List, Dict, Tuple
import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        assert embeddings.shape[0] == len(chunks), "embeddings/chunks length mismatch"
        self.index.add(embeddings)
        self.chunk_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        query_vec = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunk_metadata[idx], float(score)))
        return results

    def __len__(self):
        return self.index.ntotal
