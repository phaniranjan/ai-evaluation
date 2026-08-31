"""BM25 Chunk Retriever for lightweight text retrieval without external vector databases."""

import math
import re
from typing import Any, Dict, List, Optional


class BM25Retriever:
    """A pure-Python BM25 ranker over document text chunks with metadata tracking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_frequencies: Dict[str, int] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Convert text into lowercase tokens, stripping punctuation."""
        return re.findall(r"\b\w+\b", text.lower())

    def add_articles(self, articles: List[Dict[str, Any]]) -> None:
        """Load and index article chunks with full document metadata.

        Args:
            articles: List of article dicts containing metadata and 'chunks'.
        """
        for article in articles:
            article_id = article.get("id", "")
            title = article.get("title", "")
            source = article.get("source", "")
            published_date = article.get("published_date", "")

            for chunk in article.get("chunks", []):
                chunk_id = chunk.get("chunk_id", "")
                text = chunk.get("text", "")
                tokens = self._tokenize(text)

                chunk_entry = {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "article_title": title,
                    "source": source,
                    "published_date": published_date,
                    "text": text,
                    "tokens": tokens,
                }
                self.chunks.append(chunk_entry)

        self._build_index()

    def _build_index(self) -> None:
        """Calculate document lengths, term frequencies, and inverse document frequencies."""
        self.corpus_size = len(self.chunks)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        self.doc_lengths = [len(c["tokens"]) for c in self.chunks]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size

        self.doc_term_freqs = []
        self.doc_frequencies = {}

        for chunk in self.chunks:
            tf: Dict[str, int] = {}
            for token in chunk["tokens"]:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            for token in tf.keys():
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1

    def _calc_idf(self, word: str) -> float:
        """Calculate Robertson-Spärck Jones IDF for a query term."""
        df = self.doc_frequencies.get(word, 0)
        if df == 0:
            return 0.0
        # Standard BM25 IDF formula with smoothing
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def retrieve_chunks(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve top_k matching chunks complete with internal metadata and BM25 scores.

        Args:
            query: The user query string.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            List of chunk dicts including score and document metadata.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for token in set(query_tokens):
            idf = self._calc_idf(token)
            if idf <= 0.0:
                continue

            for idx, tf_dict in enumerate(self.doc_term_freqs):
                freq = tf_dict.get(token, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                )
                scores[idx] += idf * (numerator / denominator)

        # Pair chunks with scores
        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["score"] = round(score, 4)
            scored_chunks.append(chunk_copy)

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieve raw text strings for top_k relevant chunks (for DeepEval retrieval_context).

        Args:
            query: The user query string.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            List of chunk text strings.
        """
        top_chunks = self.retrieve_chunks(query, top_k=top_k)
        # Filter out chunks with zero score if query is completely unrelated
        relevant_chunks = [c["text"] for c in top_chunks if c["score"] > 0.0]
        return relevant_chunks
