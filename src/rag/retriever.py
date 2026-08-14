"""Retriever: embedder + vector store → ranked dict results + context formatting."""

from typing import List, Dict
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """End-to-end retriever: query → embedding → vector search → ranked results.

    All search methods return List[dict] with keys: {text, score, metadata}.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "topk",
        mmr_lambda: float = 0.5,
        lang: str = "zh",
    ) -> List[Dict]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User question.
            top_k: Number of results.
            method: "topk" or "mmr".
            mmr_lambda: Relevance-diversity tradeoff for MMR.
            lang: Language for BGE query prefix ("zh" or "en").

        Returns:
            List of {text, score, metadata} dicts.
        """
        query_emb = self.embedder.encode_query(query, lang=lang)

        if method == "mmr":
            return self.vector_store.mmr_search(
                query_emb, top_k=top_k, lambda_mult=mmr_lambda
            )
        return self.vector_store.search(query_emb, top_k=top_k)

    @staticmethod
    def format_context(results: List[Dict], max_length: int = 2000) -> str:
        """Format retrieved results into a context string for the LLM.

        Each entry: ``[i] (source: xxx, score: 0.xxx)\\nchunk_text``
        Total length is capped at *max_length* characters.
        """
        parts = []
        total = 0
        for i, r in enumerate(results):
            source = r.get("metadata", {}).get("source", "unknown")
            score = r.get("score", 0.0)
            text = r.get("text", "")
            entry = f"[{i+1}] (source: {source}, score: {score:.3f})\n{text}"
            if total + len(entry) > max_length:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)
