"""FAISS vector store — Top-K + MMR search + save/load persistence."""

import json
import numpy as np
import faiss
from typing import List, Tuple, Dict


class VectorStore:
    """FAISS IndexFlatIP store with MMR and persistence.

    Merges smallrag's MMR implementation with kg-agent's save/load.
    All search methods return List[dict] with keys: {text, score, metadata}.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.texts: List[str] = []
        self.metadata: List[dict] = []

    # ---- add / size ----

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[dict] = None,
    ):
        """Add embeddings with associated texts and metadata."""
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]
        assert len(embeddings) == len(texts) == len(metadata)
        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    @property
    def size(self) -> int:
        return self.index.ntotal

    # ---- search ----

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict]:
        """Standard Top-K inner-product search.

        Returns list of {text, score, metadata} dicts.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.texts[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    def mmr_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        candidates: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Dict]:
        """Maximal Marginal Relevance search — balances relevance + diversity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            candidates: Number of candidates to fetch from FAISS.
            lambda_mult: 0 = max diversity, 1 = max relevance.

        Returns list of {text, score, metadata} dicts.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        n_cand = min(candidates, self.index.ntotal)
        if n_cand == 0:
            return []
        scores, indices = self.index.search(query, n_cand)

        cand_ids = [int(i) for i in indices[0] if i >= 0]
        if not cand_ids:
            return []

        cand_embs = np.array(
            [self.index.reconstruct(i) for i in cand_ids], dtype=np.float32
        )
        cand_scores = {
            i: float(s) for i, s in zip(cand_ids, scores[0]) if i >= 0
        }

        selected: List[int] = []
        remaining = list(range(len(cand_ids)))

        for _ in range(min(top_k, len(cand_ids))):
            best_score = -float("inf")
            best_local = -1

            for i in remaining:
                relevance = cand_scores[cand_ids[i]]
                if selected:
                    sel_embs = cand_embs[selected]
                    diversity = max(float(cand_embs[i] @ e) for e in sel_embs)
                else:
                    diversity = 0.0

                mmr = lambda_mult * relevance - (1 - lambda_mult) * diversity
                if mmr > best_score:
                    best_score = mmr
                    best_local = i

            if best_local < 0:
                break
            selected.append(best_local)
            remaining.remove(best_local)

        results = []
        for i in selected:
            idx = cand_ids[i]
            results.append({
                "text": self.texts[idx],
                "score": cand_scores[idx],
                "metadata": self.metadata[idx],
            })
        return results

    # ---- persistence ----

    def save(self, path: str):
        """Save index + texts + metadata to a directory."""
        import os
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "store.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"texts": self.texts, "metadata": self.metadata, "dimension": self.dimension},
                f,
                ensure_ascii=False,
            )

    def load(self, path: str):
        """Load index + texts + metadata from a directory."""
        import os
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "store.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        self.dimension = data.get("dimension", self.dimension)
