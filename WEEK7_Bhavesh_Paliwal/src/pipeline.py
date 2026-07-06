"""
RAG Pipeline Orchestrator
---------------------------
Wires ingestion -> chunking -> embedding -> vector store -> retrieval
-> generation into one object with a simple `.ask()` interface.
Fully local. No API keys, no network calls at query time.
"""

import time
from typing import List, Dict

from .ingestion import load_documents
from .chunking import chunk_documents
from .embeddings import EmbeddingModel
from .vectorstore import FaissVectorStore
from .retriever import HybridRetriever
from .generator import LocalAnswerGenerator


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name: str = "google/flan-t5-base",
        chunk_size: int = 200,
        overlap: int = 40,
        top_k: int = 4,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.embedding_model = EmbeddingModel(embedding_model_name)
        # NOTE: in TF-IDF fallback mode the embedding dimension is only known
        # after the vectorizer is fit on the corpus, so the vector store and
        # retriever are created lazily inside ingest() instead of here.
        self.vector_store = None
        self.retriever = None
        self.generator = LocalAnswerGenerator(generator_model_name)

        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []
        self.metrics: Dict = {}

    def ingest(self, file_paths: List[str]) -> Dict:
        """Load documents, chunk them, embed them, and index them."""
        t0 = time.time()
        self.documents = load_documents(file_paths)
        t1 = time.time()

        self.chunks = chunk_documents(self.documents, self.chunk_size, self.overlap)
        t2 = time.time()

        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            embeddings = self.embedding_model.encode(texts)  # fits TF-IDF here in fallback mode
            self.vector_store = FaissVectorStore(self.embedding_model.dimension)
            self.retriever = HybridRetriever(self.embedding_model, self.vector_store)
            self.vector_store.add(embeddings, self.chunks)
        t3 = time.time()

        self.metrics = {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "chunk_size_words": self.chunk_size,
            "chunk_overlap_words": self.overlap,
            "embedding_model": self.embedding_model.model_name,
            "embedding_mode": self.embedding_model.mode,
            "embedding_dimension": self.embedding_model.dimension,
            "generator_model": self.generator.model_name,
            "generator_mode": self.generator.mode,
            "vector_store": "FAISS IndexFlatIP",
            "ingestion_time_sec": round(t1 - t0, 3),
            "chunking_time_sec": round(t2 - t1, 3),
            "embedding_index_time_sec": round(t3 - t2, 3),
            "total_setup_time_sec": round(t3 - t0, 3),
        }
        return self.metrics

    def ask(self, query: str, top_k: int = None) -> Dict:
        """Run one query through retrieve -> generate and return a full trace."""
        if self.retriever is None:
            return {
                "query": query,
                "answer": "No documents have been ingested yet. Please ingest documents first.",
                "retrieved_chunks": [],
                "retrieval_time_sec": 0.0,
                "generation_time_sec": 0.0,
                "total_time_sec": 0.0,
            }
        k = top_k or self.top_k
        t0 = time.time()
        retrieved = self.retriever.retrieve(query, top_k=k)
        t1 = time.time()
        answer = self.generator.generate(query, retrieved)
        t2 = time.time()

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "retrieval_time_sec": round(t1 - t0, 3),
            "generation_time_sec": round(t2 - t1, 3),
            "total_time_sec": round(t2 - t0, 3),
        }
