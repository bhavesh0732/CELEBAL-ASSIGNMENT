"""
Embedding Module
-----------------
Wraps a local sentence-transformers model. Everything runs on-device;
no network call is made once the model weights are cached locally.

Resilience: if the sentence-transformers weights cannot be downloaded
(no internet / firewalled environment), this module automatically
falls back to a pure scikit-learn TF-IDF vectorizer so the whole
pipeline still runs end-to-end with zero external downloads. Which
mode is active is always recorded in `self.mode` and surfaced in the
system metrics so it's never silently hidden.
"""

from typing import List
import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, strong quality/speed tradeoff


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.mode = "sentence-transformer"
        self._tfidf = None  # populated only in fallback mode

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as exc:
            print(
                f"[embeddings] Could not load '{model_name}' ({exc}). "
                "Falling back to local TF-IDF embeddings (no download required)."
            )
            self.mode = "tfidf-fallback"
            self.model_name = "tfidf (scikit-learn, local fallback)"
            self.model = None
            self.dimension = None  # fixed once fit() runs on the corpus

    def _fit_tfidf(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._tfidf = TfidfVectorizer(stop_words="english", max_features=2048)
        matrix = self._tfidf.fit_transform(texts).toarray()
        matrix = normalize(matrix)  # so inner product == cosine similarity
        self.dimension = matrix.shape[1]
        return matrix.astype("float32")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of strings into L2-normalized embedding vectors."""
        if self.mode == "tfidf-fallback":
            # First call (corpus ingestion) fits the vectorizer; later calls
            # (queries) reuse the already-fit vocabulary.
            if self._tfidf is None:
                return self._fit_tfidf(texts)
            from sklearn.preprocessing import normalize
            matrix = self._tfidf.transform(texts).toarray()
            matrix = normalize(matrix)
            return matrix.astype("float32")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
