import tempfile
import unittest
from pathlib import Path

from src.kg.graph_store import KnowledgeGraph


class GraphStoreTests(unittest.TestCase):
    def test_edge_metadata_survives_round_trip(self):
        graph = KnowledgeGraph()
        graph.add_triple("A", "relates to", "B", source="doc.md", evidence_id="e1")

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "graph.json")
            graph.save(path)
            restored = KnowledgeGraph()
            restored.load(path)

        self.assertEqual(
            restored.get_triple_metadata("A", "relates to", "B"),
            {"source": "doc.md", "evidence_id": "e1"},
        )


if __name__ == "__main__":
    unittest.main()
