"""Knowledge Graph store — NetworkX MultiDiGraph with BFS traversal + visualization."""

import json
from collections import deque
from difflib import SequenceMatcher

import networkx as nx


class KnowledgeGraph:
    """Lightweight in-memory knowledge graph backed by NetworkX."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    # ---- properties ----

    @property
    def num_entities(self):
        return self.graph.number_of_nodes()

    @property
    def num_relations(self):
        return self.graph.number_of_edges()

    @property
    def relation_types(self):
        return sorted(set(k for _, _, k in self.graph.edges(keys=True)))

    # ---- normalization helpers ----

    def _normalize(self, name):
        return name.strip().lower()

    def _display(self, name):
        norm = self._normalize(name)
        if self.graph.has_node(norm):
            return self.graph.nodes[norm].get("display", name)
        return name

    # ---- add / query ----

    def add_triple(self, subject, relation, obj, **metadata):
        s_norm = self._normalize(subject)
        o_norm = self._normalize(obj)
        r_norm = relation.strip().lower()

        if not self.graph.has_node(s_norm):
            self.graph.add_node(s_norm, display=subject.strip())
        if not self.graph.has_node(o_norm):
            self.graph.add_node(o_norm, display=obj.strip())

        if not self.graph.has_edge(s_norm, o_norm, key=r_norm):
            self.graph.add_edge(s_norm, o_norm, key=r_norm, **metadata)
            return True
        return False

    def add_triples(self, triples):
        added = 0
        for t in triples:
            if isinstance(t, dict):
                s = t.get("subject", "")
                r = t.get("relation", "")
                o = t.get("object", "")
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                s, r, o = t[0], t[1], t[2]
            else:
                continue
            if s and r and o:
                added += int(self.add_triple(s, r, o))
        return added

    def has_entity(self, name):
        return self.graph.has_node(self._normalize(name))

    def get_entity_edges(self, entity):
        norm = self._normalize(entity)
        if not self.graph.has_node(norm):
            return []
        edges = []
        for _, target, key in self.graph.out_edges(norm, keys=True):
            edges.append((self._display(norm), key, self._display(target)))
        for source, _, key in self.graph.in_edges(norm, keys=True):
            edges.append((self._display(source), key, self._display(norm)))
        return edges

    def get_neighbors(self, entity, hops=1):
        norm = self._normalize(entity)
        if not self.graph.has_node(norm):
            return set(), []

        visited = {norm}
        queue = deque([(norm, 0)])
        triples = []

        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for _, target, key in self.graph.out_edges(node, keys=True):
                triples.append((self._display(node), key, self._display(target)))
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))
            for source, _, key in self.graph.in_edges(node, keys=True):
                triples.append((self._display(source), key, self._display(node)))
                if source not in visited:
                    visited.add(source)
                    queue.append((source, depth + 1))

        entities = {self._display(n) for n in visited}
        unique_triples = list({(s, r, o) for s, r, o in triples})
        return entities, unique_triples

    def query(self, subject=None, relation=None, obj=None):
        results = []
        for s, o, k in self.graph.edges(keys=True):
            match = True
            if subject and self._normalize(subject) != s:
                match = False
            if obj and self._normalize(obj) != o:
                match = False
            if relation and self._normalize(relation) != k:
                match = False
            if match:
                results.append((self._display(s), k, self._display(o)))
        return results

    def search_entities(self, keyword, threshold=0.6):
        keyword_lower = keyword.lower()
        matches = []
        for node in self.graph.nodes():
            display = self.graph.nodes[node].get("display", node)
            if keyword_lower in node:
                matches.append((display, 1.0))
            else:
                ratio = SequenceMatcher(None, keyword_lower, node).ratio()
                if ratio >= threshold:
                    matches.append((display, ratio))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_subgraph_text(self, entity, hops=2):
        entities, triples = self.get_neighbors(entity, hops)
        if not triples:
            return f"No information found about '{entity}'."
        lines = [f"({s}) --[{r}]--> ({o})" for s, r, o in triples]
        return "\n".join(lines)

    def get_all_triples(self):
        triples = []
        for s, o, k in self.graph.edges(keys=True):
            triples.append((self._display(s), k, self._display(o)))
        return triples

    def merge(self, other):
        added = 0
        for s, o, k in other.graph.edges(keys=True):
            s_disp = other._display(s)
            o_disp = other._display(o)
            added += int(self.add_triple(s_disp, k, o_disp))
        return added

    def stats(self):
        return {
            "entities": self.num_entities,
            "relations": self.num_relations,
            "relation_types": len(self.relation_types),
            "top_entities": self._top_entities(10),
        }

    def _top_entities(self, n=10):
        degree = dict(self.graph.degree())
        sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(self._display(node), deg) for node, deg in sorted_nodes]

    # ---- persistence ----

    def save(self, path):
        data = {
            "triples": [
                {"subject": self._display(s), "relation": k, "object": self._display(o)}
                for s, o, k in self.graph.edges(keys=True)
            ],
            "stats": self.stats(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.MultiDiGraph()
        self.add_triples(data.get("triples", []))

    # ---- visualization ----

    def to_pyvis(self, height="600px"):
        """Return a pyvis Network for notebook visualization."""
        try:
            from pyvis.network import Network
        except ImportError:
            return None
        net = Network(height=height, directed=True, notebook=True, cdn_resources="in_line")
        for node in self.graph.nodes():
            display = self._display(node)
            deg = self.graph.degree(node)
            net.add_node(node, label=display, size=10 + deg * 3)
        for s, o, k in self.graph.edges(keys=True):
            net.add_edge(s, o, title=k, label=k)
        return net

    def to_matplotlib(self, figsize=(14, 10)):
        """Draw the graph with matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        labels = {n: self._display(n) for n in self.graph.nodes()}
        nx.draw_networkx_nodes(self.graph, pos, node_color="#4ECDC4", node_size=800, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edge_color="#999", arrows=True, arrowsize=15, ax=ax)

        edge_labels = {}
        for s, o, k in self.graph.edges(keys=True):
            key = (s, o)
            if key in edge_labels:
                edge_labels[key] += f"\n{k}"
            else:
                edge_labels[key] = k
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6, font_color="#e74c3c", ax=ax)

        ax.set_title(f"Knowledge Graph ({self.num_entities} entities, {self.num_relations} relations)")
        ax.axis("off")
        plt.tight_layout()
        return fig
