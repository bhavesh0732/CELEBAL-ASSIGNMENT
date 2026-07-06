"""
Answer Generation Module
--------------------------
Runs a small local sequence-to-sequence model (Flan-T5) to produce a
grounded answer from the retrieved context. Everything executes
on-device via the `transformers` library -- there is no call to
OpenAI, Anthropic, or any other hosted API.

Resilience: if the Flan-T5 weights cannot be downloaded (no internet /
firewalled environment), this module automatically falls back to a
pure Python extractive answerer -- it scores every sentence across the
retrieved chunks against the question's keywords and returns the
best-matching sentences. This keeps the pipeline fully runnable with
zero external downloads. The active mode is always recorded in
`self.mode` and surfaced in the system metrics.
"""

import re
from typing import List, Dict

DEFAULT_MODEL_NAME = "google/flan-t5-base"
_WORD_RE = re.compile(r"[a-z0-9]+")

# Small stopword list so generic words (the, is, what, does) don't inflate
# overlap scores and cause irrelevant sentences to look like good matches.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "why", "how", "who", "when", "where", "which", "does", "do",
    "did", "of", "in", "on", "at", "to", "for", "with", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "about", "into", "than", "then", "there", "their", "can", "will",
}

# Minimum blended relevance score a sentence must reach before it's trusted
# as an answer. Below this, the question is treated as not answerable from
# the provided documents -- this is what keeps the fallback grounded.
_MIN_RELEVANCE_SCORE = 0.18


def _tokenize(text: str) -> set:
    tokens = _WORD_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _split_sentences(text: str) -> List[str]:
    # Simple, dependency-free sentence splitter (good enough for chunk-sized text)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class LocalAnswerGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_new_tokens: int = 200):
        self.mode = "flan-t5"
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.pipe = None

        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline("text2text-generation", model=model_name, device=device)
        except Exception as exc:
            print(
                f"[generator] Could not load '{model_name}' ({exc}). "
                "Falling back to local extractive answering (no download required)."
            )
            self.mode = "extractive-fallback"
            self.model_name = "extractive sentence-ranking (local fallback, no LM)"

    @staticmethod
    def build_prompt(query: str, context_chunks: List[Dict]) -> str:
        context_block = "\n\n".join(
            f"[Source: {c['source']} | chunk {c['chunk_index']}]\n{c['text']}"
            for c in context_chunks
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not contained in the context, say "
            "\"I could not find this in the provided documents.\"\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return prompt

    def _generate_extractive(self, query: str, context_chunks: List[Dict]) -> str:
        """Rank every sentence in the retrieved chunks by keyword overlap
        with the question (stopwords excluded) and return the best matching
        sentence(s), grounded and cited by source. No language model
        involved -- pure lexical scoring. A minimum relevance threshold
        prevents the fallback from confidently answering questions that
        aren't actually covered by the documents."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return "I could not find this in the provided documents."

        scored_sentences = []
        for chunk in context_chunks:
            for sentence in _split_sentences(chunk["text"]):
                sent_tokens = _tokenize(sentence)
                if not sent_tokens:
                    continue
                overlap = query_tokens & sent_tokens
                if not overlap:
                    continue
                # Recall against the question's meaningful terms, weighted
                # down for very long sentences that pick up words by chance.
                score = len(overlap) / len(query_tokens)
                length_penalty = min(1.0, 12 / max(len(sent_tokens), 1))
                score *= 0.7 + 0.3 * length_penalty
                scored_sentences.append((score, sentence, chunk["source"]))

        if not scored_sentences:
            return "I could not find this in the provided documents."

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_score = scored_sentences[0][0]

        if best_score < _MIN_RELEVANCE_SCORE:
            return "I could not find this in the provided documents."

        # Return only the single best-matching sentence. Adding a "runner-up"
        # sentence was tried and reverted: with pure lexical scoring the
        # second-best sentence is too often a false-positive match on a
        # generic shared word, which hurts precision more than the extra
        # context helps -- one clean, well-grounded sentence beats two
        # sentences where one might be off-topic.
        best_score, best_sentence, best_source = scored_sentences[0]
        return f"{best_sentence} (Source: {best_source})"

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I could not find this in the provided documents."

        if self.mode == "extractive-fallback":
            return self._generate_extractive(query, context_chunks)

        prompt = self.build_prompt(query, context_chunks)
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return result[0]["generated_text"].strip()
