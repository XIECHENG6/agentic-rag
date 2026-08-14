"""KG retriever — entity matching + BFS graph traversal + triple scoring."""

import re

from .graph_store import KnowledgeGraph
from .extractor import TripleExtractor


class GraphRetriever:
    """Retrieve relevant KG triples for a given query.

    Pipeline: keyword match → LLM entity extraction → fuzzy match → BFS → score.
    """

    def __init__(self, kg: KnowledgeGraph, extractor: TripleExtractor = None):
        self.kg = kg
        self.extractor = extractor

    def retrieve(self, query, hops=2, max_results=10):
        entities = self._find_query_entities(query)
        if not entities:
            return []

        all_triples = []
        seen = set()

        for entity in entities:
            _, triples = self.kg.get_neighbors(entity, hops=hops)
            for t in triples:
                key = (t[0].lower(), t[1].lower(), t[2].lower())
                if key not in seen:
                    seen.add(key)
                    all_triples.append(t)

        scored = self._score_triples(all_triples, query, entities)
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {"triple": t, "score": s, "text": f"{t[0]} --[{t[1]}]--> {t[2]}"}
            for t, s in scored[:max_results]
        ]

    @staticmethod
    def _clean_query(query):
        return re.sub(r"[^\w\s]", "", query)

    @staticmethod
    def _is_cjk(text: str) -> bool:
        """Check if text contains CJK characters."""
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _find_query_entities(self, query):
        found = []

        # 1. Direct keyword match in graph
        if self._is_cjk(query):
            # Chinese: extract character n-grams and match against entities
            clean = self._clean_query(query).replace(" ", "")
            all_entities_lower = {n.lower(): n for n in self.kg.graph.nodes()}
            # Try full query first
            if clean.lower() in all_entities_lower:
                found.append(all_entities_lower[clean.lower()])
            # Then character n-grams (length 2..min(len,6))
            if not found:
                for n in range(min(len(clean), 6), 1, -1):
                    for i in range(len(clean) - n + 1):
                        ngram = clean[i : i + n].lower()
                        if ngram in all_entities_lower:
                            name = all_entities_lower[ngram]
                            if name not in found:
                                found.append(name)
        else:
            # English: space-split word n-grams
            words = self._clean_query(query).lower().split()
            for n in range(len(words), 0, -1):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    if self.kg.has_entity(phrase):
                        found.append(phrase)

        # 2. LLM entity extraction fallback
        if not found and self.extractor:
            extracted = self.extractor.extract_entities_from_query(query)
            for entity in extracted:
                matches = self.kg.search_entities(entity, threshold=0.7)
                for match_name, score in matches[:2]:
                    if match_name.lower() not in [f.lower() for f in found]:
                        found.append(match_name)

        # 3. Fuzzy match fallback
        if not found:
            matches = self.kg.search_entities(query, threshold=0.5)
            found = [name for name, _ in matches[:3]]

        return found

    @staticmethod
    def _tokenize_for_scoring(text: str) -> set:
        """Tokenize text: space-split for English, character bigrams for CJK."""
        if any('\u4e00' <= ch <= '\u9fff' for ch in text):
            # Chinese: use character bigrams
            clean = text.replace(" ", "").lower()
            tokens = set(clean)  # unigrams
            for i in range(len(clean) - 1):
                tokens.add(clean[i : i + 2])  # bigrams
            return tokens
        return set(text.lower().split())

    def _score_triples(self, triples, query, query_entities):
        query_tokens = self._tokenize_for_scoring(query)
        entity_set = {e.lower() for e in query_entities}

        scored = []
        for triple in triples:
            s, r, o = triple
            score = 0.0

            # Boost if subject/object is a query entity
            if s.lower() in entity_set or o.lower() in entity_set:
                score += 2.0

            # Token overlap with query
            triple_tokens = self._tokenize_for_scoring(f"{s} {r} {o}")
            overlap = len(query_tokens & triple_tokens)
            score += overlap * 0.5

            # Boost if relation matches query tokens
            relation_tokens = self._tokenize_for_scoring(r)
            if query_tokens & relation_tokens:
                score += 1.0

            scored.append((triple, score))

        return scored

    def get_context(self, query, hops=2, max_results=10):
        """Return formatted text context from KG for a query."""
        results = self.retrieve(query, hops=hops, max_results=max_results)
        if not results:
            return "No relevant knowledge graph context found."

        lines = ["Knowledge Graph Context:"]
        for r in results:
            lines.append(f"  - {r['text']}")
        return "\n".join(lines)
