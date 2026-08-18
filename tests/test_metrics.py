import unittest

from src.evaluation.metrics import classify_failure, compute_source_recall


class MetricsTests(unittest.TestCase):
    def test_source_recall_uses_source_markers(self):
        self.assertEqual(
            compute_source_recall("(source: doc.md)", ["doc.md"]), 1.0
        )

    def test_budget_exhaustion_is_a_failure(self):
        result = {"status": "llm_budget_exceeded", "answer": ""}
        self.assertEqual(classify_failure(result), "llm_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
