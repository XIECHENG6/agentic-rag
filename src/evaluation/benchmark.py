"""Benchmark runner — compare multiple RAG systems on the same QA dataset.

Supports: no_retrieval, simple_rag, hybrid_rag, agentic_rag (+ ablation variants).
"""

import json
import time
from typing import List, Dict, Optional

from src.pipeline import AgenticRAGPipeline
from src.evaluation.metrics import evaluate_single


class BenchmarkRunner:
    """Run multiple systems on the same QA dataset and compare results.

    Usage::

        runner = BenchmarkRunner(pipeline)
        runner.load_benchmark("data/benchmark.json")
        results = runner.run_all(systems=["simple_rag", "hybrid_rag", "agentic_rag"])
        runner.print_summary(results)
    """

    def __init__(self, pipeline: AgenticRAGPipeline, verbose: bool = False):
        self.pipeline = pipeline
        self.benchmark: List[Dict] = []
        self.verbose = verbose

    def load_benchmark(self, path: str):
        """Load benchmark QA pairs from JSON.

        Expected format: list of {
            "question": str,
            "answer": str,
            "type": "simple"|"bridge"|"comparison",
            "difficulty": "easy"|"medium"|"hard",
            "source_docs": [str],
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            self.benchmark = json.load(f)
        if self.verbose:
            types = {}
            for q in self.benchmark:
                t = q.get("type", "unknown")
                types[t] = types.get(t, 0) + 1
            print(f"Loaded {len(self.benchmark)} QA pairs: {types}")

    def run_system(self, system: str, questions: List[str], verbose: bool = False) -> List[Dict]:
        """Run a single system on a list of questions.

        Args:
            system: One of "no_retrieval", "simple_rag", "hybrid_rag", "agentic_rag".
            questions: List of question strings.

        Returns:
            List of result dicts with keys: answer, time, trace (if agent).
        """
        results = []
        for i, q in enumerate(questions):
            if verbose:
                print(f"  [{i+1}/{len(questions)}] {q[:60]}...")

            start = time.time()
            try:
                if system == "no_retrieval":
                    out = self.pipeline.no_retrieval(q)
                elif system == "simple_rag":
                    out = self.pipeline.simple_rag(q)
                elif system == "hybrid_rag":
                    out = self.pipeline.hybrid_rag(q)
                elif system == "agentic_rag":
                    out = self.pipeline.ask(q, verbose=False)
                else:
                    out = {"answer": "", "error": f"Unknown system: {system}"}
            except Exception as e:
                out = {"answer": "", "error": str(e)}

            elapsed = time.time() - start

            result = {
                "answer": out.get("answer", ""),
                "context": out.get("context", ""),
                "time": round(elapsed, 2),
                "error": out.get("error", None),
            }
            # Agent-specific fields
            if system == "agentic_rag":
                result["trace"] = out.get("trace", [])
                result["question_type"] = out.get("question_type", "")
                result["strategy"] = out.get("strategy", "")
                result["reformulations"] = out.get("reformulations", 0)

            results.append(result)

            if verbose:
                print(f"    → {result['answer'][:80]}... ({elapsed:.1f}s)")

        return results

    def run_all(
        self,
        systems: List[str] = None,
        questions: List[Dict] = None,
        verbose: bool = False,
    ) -> Dict[str, Dict]:
        """Run all systems on the benchmark and compute metrics.

        Args:
            systems: List of system names. Default: all four.
            questions: Override benchmark questions (useful for subset testing).
            verbose: Print progress.

        Returns:
            Dict of system → {metrics, results, per_type_metrics}.
        """
        if systems is None:
            systems = ["no_retrieval", "simple_rag", "hybrid_rag", "agentic_rag"]

        qa_pairs = questions or self.benchmark
        if not qa_pairs:
            print("No benchmark loaded. Call load_benchmark() first.")
            return {}

        q_list = [q["question"] for q in qa_pairs]
        ref_list = [q["answer"] for q in qa_pairs]
        type_list = [q.get("type", "unknown") for q in qa_pairs]

        all_results = {}
        for system in systems:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running: {system}")
                print(f"{'='*60}")

            results = self.run_system(system, q_list, verbose=verbose)

            # Separate successful results from errors
            success_results = [r for r in results if not r.get("error")]
            error_results = [r for r in results if r.get("error")]
            error_count = len(error_results)

            # Compute overall metrics on successful results only
            from src.evaluation.metrics import evaluate_dataset
            if success_results:
                success_indices = [i for i, r in enumerate(results) if not r.get("error")]
                predictions = [results[i]["answer"] for i in success_indices]
                refs = [ref_list[i] for i in success_indices]
                qs = [q_list[i] for i in success_indices]
                ctxs = [results[i].get("context", "") for i in success_indices]
                metrics = evaluate_dataset(predictions, refs, qs, ctxs)
            else:
                metrics = {"rouge_l": 0, "exact_match": 0, "f1": 0, "num_samples": 0}
            metrics["avg_time"] = sum(r["time"] for r in success_results) / max(len(success_results), 1)
            metrics["error_count"] = error_count
            metrics["error_rate"] = error_count / len(results) if results else 0

            # Per-type metrics (also skip errors)
            per_type = {}
            for qtype in set(type_list):
                type_indices = [
                    i for i, t in enumerate(type_list) if t == qtype and not results[i].get("error")
                ]
                if not type_indices:
                    continue
                type_preds = [results[i]["answer"] for i in type_indices]
                type_refs = [ref_list[i] for i in type_indices]
                type_qs = [q_list[i] for i in type_indices]
                type_ctxs = [results[i].get("context", "") for i in type_indices]
                type_metrics = evaluate_dataset(type_preds, type_refs, type_qs, type_ctxs)
                per_type[qtype] = type_metrics

            all_results[system] = {
                "metrics": metrics,
                "results": results,
                "per_type_metrics": per_type,
            }

        return all_results

    @staticmethod
    def print_summary(all_results: Dict[str, Dict]):
        """Print a comparison table of all systems."""
        print(f"\n{'='*90}")
        print(f"{'System':<16} {'ROUGE-L':>8} {'EM':>8} {'F1':>8} {'Faith':>8} {'Avg Time':>10} {'Errors':>8}")
        print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

        for system, data in all_results.items():
            m = data["metrics"]
            err = m.get("error_count", 0)
            faith = f"{m['faithfulness']:>8.3f}" if "faithfulness" in m else "     N/A"
            print(
                f"{system:<16} "
                f"{m.get('rouge_l', 0):>8.3f} "
                f"{m.get('exact_match', 0):>8.1%} "
                f"{m.get('f1', 0):>8.3f} "
                f"{faith}"
                f"{m.get('avg_time', 0):>10.1f}s"
                f"{err:>8}"
            )

        # Per-type breakdown for agentic_rag
        if "agentic_rag" in all_results:
            print(f"\n{'='*80}")
            print("Agentic RAG per-type breakdown:")
            print(f"{'Type':<16} {'ROUGE-L':>8} {'EM':>8} {'F1':>8} {'N':>5}")
            print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

            for qtype, m in all_results["agentic_rag"]["per_type_metrics"].items():
                print(
                    f"{qtype:<16} "
                    f"{m.get('rouge_l', 0):>8.3f} "
                    f"{m.get('exact_match', 0):>8.1%} "
                    f"{m.get('f1', 0):>8.3f} "
                    f"{m.get('num_samples', 0):>5}"
                )

    @staticmethod
    def save_results(all_results: Dict[str, Dict], path: str):
        """Save benchmark results to JSON."""
        serializable = {}
        for system, data in all_results.items():
            serializable[system] = {
                "metrics": data["metrics"],
                "per_type_metrics": data["per_type_metrics"],
                "results": [
                    {k: v for k, v in r.items() if k != "trace"}
                    for r in data["results"]
                ],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
