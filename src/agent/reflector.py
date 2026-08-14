"""Reflector — context quality scoring + query reformulation."""

from src.core.llm_client import LLMClient

# ---------- Reflection ----------

REFLECT_SYSTEM = """You are a context quality evaluator for a RAG system.
Given a question and retrieved context, score the context on three dimensions (0.0-1.0):
- relevance: Does the context contain information about the question's topic?
- coverage: Does the context contain enough information to fully answer the question?
- sufficiency: Can you confidently answer the question using ONLY this context?

Also provide an overall judgment:
- "sufficient": all three scores >= 0.6
- "reformulate": any score < 0.6 (suggest a better query)
- "give_up": retrieval has been tried multiple times without success

If judgment is "reformulate", also provide:
- new_query: an improved query string
- reformulation_strategy: one of "expand" (add synonyms), "focus" (narrow scope), "perspective_shift" (rephrase from different angle)

Output ONLY a JSON object, no other text."""

REFLECT_PROMPT = """Evaluate the retrieved context for answering this question.

Question: {question}

Retrieved Context:
{context}

Score relevance, coverage, sufficiency (0.0-1.0 each) and provide judgment.
Output JSON:"""

# ---------- Reformulation ----------

REFORMULATE_SYSTEM = """You are a query reformulation agent for a RAG system.
Given the original question, the current query, the retrieved (insufficient) context, and a reformulation strategy, generate an improved search query.

Strategies:
- "expand": Add synonyms, related terms, or broader concepts. E.g., "BERT模型" → "BERT预训练语言模型 双向Transformer 掩码语言模型"
- "focus": Narrow to a specific aspect. E.g., "Transformer架构" → "Transformer自注意力机制的计算过程和数学公式"
- "perspective_shift": Rephrase from a different angle. E.g., "谁发明了X?" → "X的发明者是谁 X的起源"

Output ONLY the new query string, nothing else."""

REFORMULATE_PROMPT = """Reformulate this search query using the "{strategy}" strategy.

Original question: {question}
Current query: {current_query}
Insufficient context summary: {context_summary}

Generate an improved search query:"""


class Reflector:
    """Evaluate context quality and reformulate queries when needed."""

    def __init__(self, llm: LLMClient, threshold: float = 0.6):
        self.llm = llm
        self.threshold = threshold

    @staticmethod
    def _safe_float(val, default: float = 0.5) -> float:
        """Safely convert a value to float, clamp to [0, 1], return default on failure."""
        if val is None:
            return default
        try:
            f = float(val)
        except (ValueError, TypeError):
            # Try extracting a number from strings like "0.8分", "high" etc.
            import re
            m = re.search(r"[-+]?\d*\.?\d+", str(val))
            if m:
                try:
                    f = float(m.group())
                except (ValueError, TypeError):
                    return default
            else:
                return default
        return max(0.0, min(1.0, f))

    def _build_reflect_system(self) -> str:
        """Build reflect system prompt with the current threshold."""
        return f"""You are a context quality evaluator for a RAG system.
Given a question and retrieved context, score the context on three dimensions (0.0-1.0):
- relevance: Does the context contain information about the question's topic?
- coverage: Does the context contain enough information to fully answer the question?
- sufficiency: Can you confidently answer the question using ONLY this context?

IMPORTANT: Output scores as plain numbers like 0.8, not "0.8分" or "high".

Also provide an overall judgment:
- "sufficient": all three scores >= {self.threshold}
- "reformulate": any score < {self.threshold} (suggest a better query)
- "give_up": retrieval has been tried multiple times without success

If judgment is "reformulate", also provide:
- new_query: an improved query string
- reformulation_strategy: one of "expand" (add synonyms), "focus" (narrow scope), "perspective_shift" (rephrase from different angle)

Output ONLY a JSON object, no other text."""

    def evaluate(self, question: str, context: str) -> dict:
        """Score retrieved context quality.

        Returns:
            {
                "relevance": float,
                "coverage": float,
                "sufficiency": float,
                "judgment": "sufficient"|"reformulate"|"give_up",
                "new_query": str|None,
                "strategy": str|None,
                "reasoning": str,
            }
        """
        response = self.llm.generate_text(
            REFLECT_PROMPT.format(question=question, context=context[:3000]),
            system=self._build_reflect_system(),
            temperature=0.0,
        )

        parsed = self.llm.extract_json(response)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            return {
                "relevance": 0.5,
                "coverage": 0.5,
                "sufficiency": 0.5,
                "judgment": "reformulate",
                "new_query": question,
                "strategy": "expand",
                "reasoning": "Reflector parse failed, attempting reformulation.",
            }

        relevance = self._safe_float(parsed.get("relevance", 0.5))
        coverage = self._safe_float(parsed.get("coverage", 0.5))
        sufficiency = self._safe_float(parsed.get("sufficiency", 0.5))

        # Override judgment based on threshold
        judgment = parsed.get("judgment", "sufficient")
        if all(s >= self.threshold for s in [relevance, coverage, sufficiency]):
            judgment = "sufficient"
        elif judgment == "sufficient":
            judgment = "reformulate"

        return {
            "relevance": relevance,
            "coverage": coverage,
            "sufficiency": sufficiency,
            "judgment": judgment,
            "new_query": parsed.get("new_query"),
            "strategy": parsed.get("reformulation_strategy", "expand"),
            "reasoning": parsed.get("reasoning", ""),
        }

    def reformulate(
        self,
        question: str,
        current_query: str,
        context: str,
        strategy: str = "expand",
    ) -> str:
        """Generate an improved search query.

        Args:
            question: Original question.
            current_query: The query that produced insufficient results.
            context: The insufficient context (for context_summary).
            strategy: "expand", "focus", or "perspective_shift".

        Returns:
            New query string.
        """
        context_summary = context[:500] if context else "No useful context retrieved."

        response = self.llm.generate_text(
            REFORMULATE_PROMPT.format(
                question=question,
                current_query=current_query,
                context_summary=context_summary,
                strategy=strategy,
            ),
            system=REFORMULATE_SYSTEM,
            temperature=0.0,
        )

        # Clean up — take first non-empty line
        new_query = response.strip().split("\n")[0].strip().strip('"').strip("'")
        return new_query if new_query else current_query
