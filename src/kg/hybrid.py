"""Hybrid retrieval — RRF / weighted fusion of vector + KG results."""

import hashlib

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

    @staticmethod
    def _evidence_id(result, backend):
        metadata = result.get("metadata", {})
        source = metadata.get("source")
        chunk_id = metadata.get("chunk_id")
        if source is not None and chunk_id is not None:
            key = f"chunk\x1f{source}\x1f{chunk_id}"
            return "evidence:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        evidence_id = result.get("evidence_id")
        if evidence_id and not evidence_id.startswith(("kg:", "vector:", "graph:")):
            return evidence_id
        text_hash = hashlib.sha1(result.get("text", "").encode("utf-8")).hexdigest()[:16]
        return "text:" + text_hash

    @staticmethod
    def _result_metadata(result):
        return dict(result.get("metadata", {}))

    def _format_fused(self, items, top_k):
        ranked = sorted(items.items(), key=lambda item: item[1]["score"], reverse=True)
        output = []
        for evidence_id, item in ranked[:top_k]:
            output.append({
                "evidence_id": evidence_id,
                "text": item["text"],
                "score": item["score"],
                "source": "+".join(sorted(item["sources"])),
                "metadata": item["metadata"],
            })
        return output

    def _rrf_fusion(self, vector_results, graph_results, top_k):
        items = {}
        for backend, results in (("vector", vector_results), ("graph", graph_results)):
            for rank, result in enumerate(results):
                evidence_id = self._evidence_id(result, backend)
                item = items.setdefault(evidence_id, {
                    "text": result.get("text", ""),
                    "metadata": self._result_metadata(result),
                    "sources": set(),
                    "score": 0.0,
                })
                item["score"] += 1.0 / (self.rrf_k + rank + 1)
                item["sources"].add(backend)
        return self._format_fused(items, top_k)

    def _weighted_fusion(self, vector_results, graph_results, top_k):
        items = {}
        for backend, results, weight in (
            ("vector", vector_results, self.vector_weight),
            ("graph", graph_results, self.graph_weight),
        ):
            if not results:
                continue
            raw_scores = [result.get("score", 0.0) for result in results]
            min_score = min(raw_scores)
            shifted = [max(score - min(min_score, 0), 0) for score in raw_scores]
            max_score = max(shifted) or 1.0
            for result, normalized_score in zip(results, shifted):
                evidence_id = self._evidence_id(result, backend)
                item = items.setdefault(evidence_id, {
                    "text": result.get("text", ""),
                    "metadata": self._result_metadata(result),
                    "sources": set(),
                    "score": 0.0,
                })
                item["score"] += weight * normalized_score / max_score
                item["sources"].add(backend)
        return self._format_fused(items, top_k)

    def get_context(self, query, top_k=5, graph_hops=2, lang="zh"):
        results = self.retrieve(query, top_k=top_k, graph_hops=graph_hops, lang=lang)
        if not results:
            return "No relevant context found."

        lines = ["Retrieved Context:"]
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "unknown")
            lines.append(f"  [{i}] (source: {source}, evidence_id: {r['evidence_id']}, score: {r['score']:.4f}) {r['text']}")
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
