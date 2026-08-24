"""Benchmark runner — compare multiple RAG systems on the same QA dataset.

Supports: no_retrieval, simple_rag, hybrid_rag, agentic_rag (+ ablation variants).
"""

import json
import time
from typing import List, Dict, Optional

from src.pipeline import AgenticRAGPipeline
from src.evaluation.metrics import classify_failure, compute_source_recall, evaluate_single, compute_llm_judge


class BenchmarkRunner:
    """Run multiple systems on the same QA dataset and compare results.

    Usage::

        runner = BenchmarkRunner(pipeline)
        runner.load_benchmark("data/benchmark.json")
        results = runner.run_all(systems=["simple_rag", "hybrid_rag", "agentic_rag"])
        runner.print_summary(results)
    """

    def __init__(
        self,
        pipeline: AgenticRAGPipeline,
        verbose: bool = False,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        use_llm_judge: bool = True,
    ):
        self.pipeline = pipeline
        self.benchmark: List[Dict] = []
        self.verbose = verbose
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.use_llm_judge = use_llm_judge

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

            usage_before = self.pipeline.llm.snapshot_usage()
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
            usage = self.pipeline.llm.usage_delta(usage_before)
            usage["estimated_cost_usd"] = (
                usage["prompt_tokens"] * self.input_cost_per_1m / 1_000_000
                + usage["completion_tokens"] * self.output_cost_per_1m / 1_000_000
            )

            result = {
                "answer": out.get("answer", ""),
                "context": out.get("context", ""),
                "time": round(elapsed, 2),
                "error": out.get("error", None),
                "status": out.get("status", "completed"),
                "llm_usage": usage,
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
        source_list = [q.get("source_docs", []) for q in qa_pairs]

        all_results = {}
        for system in systems:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running: {system}")
                print(f"{'='*60}")

            results = self.run_system(system, q_list, verbose=verbose)
            for index, result in enumerate(results):
                result["source_recall"] = compute_source_recall(
                    result.get("context", ""), source_list[index]
                )
                result["failure_type"] = classify_failure(
                    result, source_list[index] if system != "no_retrieval" else [], ref_list[index]
                )

            # Only completed runs belong in quality averages. A forced answer
            # after max_steps is useful for failure analysis, but is not a
            # successful Agent run.
            success_results = [
                r for r in results
                if not r.get("error") and r.get("status") == "completed"
            ]
            error_results = [r for r in results if r.get("error")]
            incomplete_results = [
                r for r in results
                if not r.get("error") and r.get("status") != "completed"
            ]
            error_count = len(error_results)

            # Compute overall metrics on successful results only
            from src.evaluation.metrics import evaluate_dataset
            if success_results:
                success_indices = [
                    i for i, r in enumerate(results)
                    if not r.get("error") and r.get("status") == "completed"
                ]
                predictions = [results[i]["answer"] for i in success_indices]
                refs = [ref_list[i] for i in success_indices]
                qs = [q_list[i] for i in success_indices]
                ctxs = [results[i].get("context", "") for i in success_indices]
                expected = [source_list[i] for i in success_indices]
                metrics = evaluate_dataset(predictions, refs, qs, ctxs, expected)
            else:
                metrics = {"rouge_l": 0, "exact_match": 0, "f1": 0, "num_samples": 0}
            metrics["avg_time"] = sum(r["time"] for r in success_results) / max(len(success_results), 1)
            metrics["error_count"] = error_count
            metrics["error_rate"] = error_count / len(results) if results else 0
            metrics["incomplete_count"] = len(incomplete_results)
            usage_fields = ["calls", "errors", "prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds", "estimated_cost_usd"]
            for field in usage_fields:
                metrics[f"llm_{field}"] = sum(
                    result.get("llm_usage", {}).get(field, 0) for result in results
                )
            metrics["failure_count"] = sum(
                1 for result in results if result.get("failure_type")
            )

            # Per-type metrics use the same completed-only population.
            per_type = {}
            for qtype in set(type_list):
                type_indices = [
                    i for i, t in enumerate(type_list)
                    if t == qtype
                    and not results[i].get("error")
                    and results[i].get("status") == "completed"
                ]
                if not type_indices:
                    continue
                type_preds = [results[i]["answer"] for i in type_indices]
                type_refs = [ref_list[i] for i in type_indices]
                type_qs = [q_list[i] for i in type_indices]
                type_ctxs = [results[i].get("context", "") for i in type_indices]
                type_sources = [source_list[i] for i in type_indices]
                type_metrics = evaluate_dataset(type_preds, type_refs, type_qs, type_ctxs, type_sources)
                per_type[qtype] = type_metrics

            # LLM-as-Judge: semantic quality scoring (one extra API call per question)
            if self.use_llm_judge:
                judge_scores = []
                for idx in success_indices:
                    score = compute_llm_judge(
                        self.pipeline.llm,
                        q_list[idx],
                        ref_list[idx],
                        results[idx]["answer"],
                    )
                    results[idx]["llm_judge"] = score
                    if score is not None:
                        judge_scores.append(score)
                    if self.verbose and score is not None:
                        print(f"    [Judge] Q{idx+1}: {score:.2f}")
                if judge_scores:
                    metrics["llm_judge"] = sum(judge_scores) / len(judge_scores)

                # Per-type LLM judge scores
                for qtype in per_type:
                    type_indices = [
                        i for i, t in enumerate(type_list)
                        if t == qtype
                        and not results[i].get("error")
                        and results[i].get("status") == "completed"
                    ]
                    type_judge = [
                        results[i]["llm_judge"]
                        for i in type_indices
                        if results[i].get("llm_judge") is not None
                    ]
                    if type_judge:
                        per_type[qtype]["llm_judge"] = sum(type_judge) / len(type_judge)
            else:
                for idx in success_indices:
                    results[idx]["llm_judge"] = None

            all_results[system] = {
                "metrics": metrics,
                "results": results,
                "per_type_metrics": per_type,
                "failure_cases": [
                    {"index": i, "question": q_list[i], "reference": ref_list[i], **result}
                    for i, result in enumerate(results)
                    if result.get("failure_type")
                ],
            }

        return all_results

    @staticmethod
    def print_summary(all_results: Dict[str, Dict]):
        """Print a comparison table of all systems."""
        print(f"\n{'='*110}")
        print(f"{'System':<16} {'ROUGE-L':>8} {'Judge':>8} {'F1':>8} {'Recall':>8} {'Calls':>8} {'Cost $':>10} {'Avg Time':>10} {'Failures':>9}")
        print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*9}")

        for system, data in all_results.items():
            metrics = data["metrics"]
            judge_str = f"{metrics.get('llm_judge', 0):>8.3f}" if 'llm_judge' in metrics else f"{'N/A':>8}"
            print(
                f"{system:<16} "
                f"{metrics.get('rouge_l', 0):>8.3f} "
                f"{judge_str} "
                f"{metrics.get('f1', 0):>8.3f} "
                f"{metrics.get('source_recall', 0):>8.3f} "
                f"{metrics.get('llm_calls', 0):>8.0f} "
                f"{metrics.get('llm_estimated_cost_usd', 0):>10.4f} "
                f"{metrics.get('avg_time', 0):>9.1f}s "
                f"{metrics.get('failure_count', 0):>9}"
            )

        if "agentic_rag" in all_results:
            print(f"\n{'='*90}")
            print("Agentic RAG per-type breakdown:")
            print(f"{'Type':<16} {'ROUGE-L':>8} {'Judge':>8} {'Recall':>8} {'F1':>8} {'N':>5}")
            print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
            for qtype, metrics in all_results["agentic_rag"]["per_type_metrics"].items():
                judge_str = f"{metrics.get('llm_judge', 0):>8.3f}" if 'llm_judge' in metrics else f"{'N/A':>8}"
                print(
                    f"{qtype:<16} "
                    f"{metrics.get('rouge_l', 0):>8.3f} "
                    f"{judge_str} "
                    f"{metrics.get('source_recall', 0):>8.3f} "
                    f"{metrics.get('f1', 0):>8.3f} "
                    f"{metrics.get('num_samples', 0):>5}"
                )

    @staticmethod
    def save_results(all_results: Dict[str, Dict], path: str):
        """Save benchmark results to JSON."""
        serializable = {}
        for system, data in all_results.items():
            serializable[system] = {
                "metrics": data["metrics"],
                "per_type_metrics": data["per_type_metrics"],
                "failure_cases": data.get("failure_cases", []),
                "results": [
                    {key: value for key, value in result.items() if key != "trace"}
                    for result in data["results"]
                ],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
