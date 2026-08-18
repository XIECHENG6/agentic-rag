"""Embedding model wrapper — configurable query prefix for BGE / non-BGE models."""

import numpy as np
from typing import List, Optional

import yaml
import os


def _load_embedding_config():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f).get("embedding", {})
    return {}


class Embedder:
    """Wrapper around sentence-transformers for text embedding.

    Supports configurable query prefix for BGE-style models.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        query_prefix_zh: str = None,
        query_prefix_en: str = None,
    ):
        from sentence_transformers import SentenceTransformer

        config = _load_embedding_config()
        self.model_name = model_name or config.get("model", "BAAI/bge-small-zh-v1.5")
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Configurable query prefixes
        self.query_prefix_zh = query_prefix_zh or config.get(
            "query_prefix_zh", "为这个句子生成表示以用于检索相关文章："
        )
        self.query_prefix_en = query_prefix_en or config.get(
            "query_prefix_en",
            "Represent this sentence for searching relevant passages: ",
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into normalized dense vectors (dot product = cosine sim)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str, lang: str = "zh") -> np.ndarray:
        """Encode a single query with the appropriate BGE prefix.

        For BGE models, prepending the retrieval prefix significantly
        improves retrieval quality.  For non-BGE models, the query is
        passed through unchanged.
        """
        if "bge" in self.model_name.lower():
            prefix = self.query_prefix_zh if lang == "zh" else self.query_prefix_en
            query = prefix + query
        return self.encode([query])[0]

    def token_count(self, text: str) -> int:
        """Return the model tokenizer length without adding special tokens."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))
