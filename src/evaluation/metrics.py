"""Evaluation metrics — merged from smallrag + kg-agent + agent-specific metrics.

Rule-based (no LLM needed):
  - ROUGE-L F1 (character-level for Chinese)
  - Exact Match (normalized)
  - Token F1
  - Faithfulness (character n-gram overlap with context)
  - Answer Relevancy (keyword overlap with question)

Agent-specific:
  - Reformulation success rate
  - Step efficiency
"""

import re
import string
from collections import Counter
from typing import List, Dict, Optional


# ============================================================
# Answer quality metrics
# ============================================================

def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 — character-level LCS, designed for Chinese text."""
    pred_chars = list(prediction)
    ref_chars = list(reference)
    if not pred_chars or not ref_chars:
        return 0.0

    m, n = len(pred_chars), len(ref_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_chars[i - 1] == ref_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_answer(s: str) -> str:
    """Lowercase + strip articles + strip punctuation (ASCII + Chinese) + collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # ASCII punctuation
    s = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", s)
    # Chinese punctuation
    s = re.sub(r"[，。！？、；：""''（）【】《》「」『』—…·～]", "", s)
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction: str, reference: str) -> float:
    """Normalized exact match (1.0 or 0.0)."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(reference).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_faithfulness(answer: str, context: str, n: int = 3) -> float:
    """Faithfulness — character n-gram overlap of answer with context.

    Higher = more grounded in the retrieved context.
    """
    if not answer or not context:
        return 0.0

    def get_ngrams(text, n):
        text = text.replace(" ", "").replace("\n", "")
        return Counter([text[i : i + n] for i in range(len(text) - n + 1)])

    answer_ngrams = get_ngrams(answer, n)
    context_ngrams = get_ngrams(context, n)
    if not answer_ngrams:
        return 0.0
    overlap = sum((answer_ngrams & context_ngrams).values())
    total = sum(answer_ngrams.values())
    return overlap / total


def compute_answer_relevancy(answer: str, question: str) -> float:
    """Answer relevancy — character bigram overlap between answer and question.

    Works for both Chinese (no spaces) and English text.
    """
    def char_bigrams(text):
        text = text.replace(" ", "").replace("\n", "").lower()
        return Counter([text[i:i+2] for i in range(len(text) - 1)]) if len(text) >= 2 else Counter([text])

    q_ngrams = char_bigrams(question)
    a_ngrams = char_bigrams(answer)
    if not q_ngrams or not a_ngrams:
        return 0.0
    overlap = sum((q_ngrams & a_ngrams).values())
    return overlap / sum(q_ngrams.values())


# ============================================================
# Agent-specific metrics
# ============================================================

def compute_reformulation_success(trace: List[Dict]) -> Optional[float]:
    """Among reformulation steps, what fraction improved context?

    Returns None if no reformulations occurred.
    """
    reformulations = [t for t in trace if t.get("state") == "REFORMULATING"]
    reflections = [t for t in trace if t.get("state") == "REFLECTING"]
    if not reformulations or len(reflections) < 2:
        return None

    # Compare pre- and post-reformulation sufficiency scores
    improvements = 0
    for i in range(len(reformulations)):
        if i + 1 < len(reflections) and i < len(reflections):
            pre = reflections[i].get("sufficiency", 0)
            post = reflections[i + 1].get("sufficiency", 0) if i + 1 < len(reflections) else pre
            if post > pre:
                improvements += 1

    return improvements / len(reformulations) if reformulations else None


def compute_step_efficiency(trace: List[Dict]) -> int:
    """Count the total number of state transitions (tool calls)."""
    return len(trace)


# ============================================================
# Aggregate evaluation
# ============================================================

def evaluate_single(
    prediction: str,
    reference: str,
    question: str = "",
    context: str = "",
) -> Dict[str, float]:
    """Compute all metrics for a single prediction.

    Returns dict with keys: rouge_l, exact_match, f1, faithfulness, answer_relevancy.
    """
    result = {
        "rouge_l": compute_rouge_l(prediction, reference),
        "exact_match": compute_exact_match(prediction, reference),
        "f1": compute_f1(prediction, reference),
    }
    if context:
        result["faithfulness"] = compute_faithfulness(prediction, context)
    if question:
        result["answer_relevancy"] = compute_answer_relevancy(prediction, question)
    return result


def evaluate_dataset(
    predictions: List[str],
    references: List[str],
    questions: List[str] = None,
    contexts: List[str] = None,
) -> Dict[str, float]:
    """Aggregate metrics across a dataset.

    Returns dict of averaged metrics.
    """
    n = len(predictions)
    assert n == len(references)
    if questions is not None:
        assert len(questions) >= n, f"questions ({len(questions)}) shorter than predictions ({n})"
    if contexts is not None:
        assert len(contexts) >= n, f"contexts ({len(contexts)}) shorter than predictions ({n})"

    rouge_scores = []
    em_scores = []
    f1_scores = []
    faith_scores = []
    relevancy_scores = []

    for i in range(n):
        q = questions[i] if questions else ""
        ctx = contexts[i] if contexts else ""
        m = evaluate_single(predictions[i], references[i], q, ctx)

        rouge_scores.append(m["rouge_l"])
        em_scores.append(m["exact_match"])
        f1_scores.append(m["f1"])
        if "faithfulness" in m:
            faith_scores.append(m["faithfulness"])
        if "answer_relevancy" in m:
            relevancy_scores.append(m["answer_relevancy"])

    result = {
        "rouge_l": sum(rouge_scores) / n,
        "exact_match": sum(em_scores) / n,
        "f1": sum(f1_scores) / n,
        "num_samples": n,
    }
    if faith_scores:
        result["faithfulness"] = sum(faith_scores) / len(faith_scores)
    if relevancy_scores:
        result["answer_relevancy"] = sum(relevancy_scores) / len(relevancy_scores)

    return result
