"""Hybrid retrieval — RRF / weighted fusion of vector + KG results."""

from .graph_store import KnowledgeGraph
from .retriever import GraphRetriever
from ..rag.embedder import Embedder
from ..rag.vector_store import VectorStore


class HybridRetriever:
    """Fuse vector search and KG retrieval via RRF or weighted scoring.

    Returns List[dict] with keys: {text, score, source}.
    """

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_store: VectorStore,
        embedder: Embedder,
        fusion: str = "rrf",
        rrf_k: int = 60,
        vector_weight: float = 0.4,
        graph_weight: float = 0.6,
    ):
        self.graph_retriever = graph_retriever
        self.vector_store = vector_store
        self.embedder = embedder
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self._text_cache = {}  # key → full text for RRF fusion

    def retrieve(self, query, top_k=5, graph_hops=2, lang="zh"):
        self._text_cache.clear()
        vector_results = self._vector_search(query, top_k=top_k * 2, lang=lang)
        graph_results = self.graph_retriever.retrieve(
            query, hops=graph_hops, max_results=top_k * 2
        )

        if self.fusion == "rrf":
            return self._rrf_fusion(vector_results, graph_results, top_k)
        return self._weighted_fusion(vector_results, graph_results, top_k)

    def _vector_search(self, query, top_k=10, lang="zh"):
        query_emb = self.embedder.encode_query(query, lang=lang)
        return self.vector_store.search(query_emb, top_k=top_k)

    def _rrf_fusion(self, vector_results, graph_results, top_k):
        scores = {}
        sources = {}

        for rank, r in enumerate(vector_results):
            key = r["text"][:200]  # truncated key for better cross-source matching
            scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            sources[key] = sources.get(key, set())
            sources[key].add("vector")
            if key not in self._text_cache:
                self._text_cache[key] = r["text"]

        for rank, r in enumerate(graph_results):
            key = r["text"][:200]
            scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            sources[key] = sources.get(key, set())
            sources[key].add("graph")
            if key not in self._text_cache:
                self._text_cache[key] = r["text"]

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "text": self._text_cache[doc],
                "score": score,
                "source": "+".join(sorted(sources[doc])),
            }
            for doc, score in sorted_docs[:top_k]
        ]

    def _weighted_fusion(self, vector_results, graph_results, top_k):
        scores = {}
        sources = {}
        text_map = {}

        if vector_results:
            raw_scores = [r["score"] for r in vector_results]
            min_v = min(raw_scores)
            # Shift to non-negative if needed, then normalize
            shifted = [max(s - min(min_v, 0), 0) for s in raw_scores]
            max_v = max(shifted) if shifted else 1.0
            if max_v == 0:
                max_v = 1.0
            for r, s in zip(vector_results, shifted):
                doc = r["text"]
                text_map[doc] = doc
                scores[doc] = self.vector_weight * (s / max_v)
                sources[doc] = sources.get(doc, set())
                sources[doc].add("vector")

        if graph_results:
            raw_scores = [r["score"] for r in graph_results]
            min_g = min(raw_scores)
            shifted = [max(s - min(min_g, 0), 0) for s in raw_scores]
            max_g = max(shifted) if shifted else 1.0
            if max_g == 0:
                max_g = 1.0
            for r, s in zip(graph_results, shifted):
                doc = r["text"]
                text_map[doc] = doc
                existing = scores.get(doc, 0)
                scores[doc] = existing + self.graph_weight * (s / max_g)
                sources[doc] = sources.get(doc, set())
                sources[doc].add("graph")

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "text": text_map[doc],
                "score": score,
                "source": "+".join(sorted(sources[doc])),
            }
            for doc, score in sorted_docs[:top_k]
        ]

    def get_context(self, query, top_k=5, graph_hops=2, lang="zh"):
        results = self.retrieve(query, top_k=top_k, graph_hops=graph_hops, lang=lang)
        if not results:
            return "No relevant context found."

        lines = ["Retrieved Context:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  [{i}] (score: {r['score']:.4f}) {r['text']}")
        return "\n".join(lines)


# ---------- convenience functions ----------

def vector_only_retrieve(vector_store, embedder, query, top_k=5, lang="zh"):
    """Quick vector-only retrieval returning formatted text."""
    query_emb = embedder.encode_query(query, lang=lang)
    results = vector_store.search(query_emb, top_k=top_k)
    if not results:
        return "No relevant context found."
    lines = ["Retrieved Context:"]
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] (score: {r['score']:.3f}) {r['text']}")
    return "\n".join(lines)


def graph_only_retrieve(graph_retriever, query, hops=2, max_results=5):
    """Quick graph-only retrieval returning formatted text."""
    return graph_retriever.get_context(query, hops=hops, max_results=max_results)
