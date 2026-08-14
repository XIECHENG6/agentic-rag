"""Planner — question type analysis + problem decomposition (single LLM call)."""

from src.core.llm_client import LLMClient

PLANNER_SYSTEM = """You are a question analysis agent for an Agentic RAG system.
Your job is to analyze the user's question and decide the best retrieval strategy.

Analyze the question and output JSON with these fields:
- question_type: one of "simple" (single-fact lookup), "multi_hop" (requires chaining 2+ facts), "comparison" (compare entities/concepts), "multi_part" (multiple independent sub-questions)
- strategy: one of "retrieve" (single retrieval suffices), "decompose" (split into sub-questions), "direct" (answerable from LLM knowledge, no retrieval needed)
- suggested_tool: which retrieval tool to use first — "vector_search", "kg_query", "kg_search", or "hybrid_search"
- sub_problems: list of 2-4 atomic sub-questions (empty if strategy != "decompose")
- reasoning: brief explanation of your decision

Output ONLY a JSON object, no other text."""

PLANNER_PROMPT = """Analyze this question and decide the retrieval strategy.

Question: {question}

Available tools:
- vector_search: semantic similarity search over document chunks
- kg_query: structured lookup by entity in knowledge graph (best for named entities)
- kg_search: fuzzy keyword search for entities in knowledge graph
- hybrid_search: combined vector + KG retrieval (best for complex questions)

Rules:
- "simple" questions ask about one fact → strategy="retrieve", suggested_tool based on whether it involves named entities (kg_query/hybrid_search) or general concepts (vector_search)
- "multi_hop" questions require chaining information → strategy="retrieve" with hybrid_search, or "decompose" if too complex for single retrieval
- "comparison" questions compare two things → strategy="decompose" into sub-questions about each thing
- "multi_part" questions have multiple independent parts → strategy="decompose"
- If the question is about well-known general knowledge that doesn't need retrieval → strategy="direct"

Output JSON:"""


class Planner:
    """Analyze questions and plan retrieval strategy."""

    VALID_TYPES = {"simple", "multi_hop", "comparison", "multi_part"}
    VALID_STRATEGIES = {"retrieve", "decompose", "direct"}
    VALID_TOOLS = {"vector_search", "kg_query", "kg_search", "hybrid_search"}

    def __init__(self, llm: LLMClient, max_sub_problems: int = 4):
        self.llm = llm
        self.max_sub_problems = max_sub_problems

    def plan(self, question: str) -> dict:
        """Analyze question and return a plan dict.

        Returns:
            {
                "question_type": "simple"|"multi_hop"|"comparison"|"multi_part",
                "strategy": "retrieve"|"decompose"|"direct",
                "suggested_tool": "vector_search"|"kg_query"|"kg_search"|"hybrid_search",
                "sub_problems": [...],
                "reasoning": "...",
            }
        """
        response = self.llm.generate_text(
            PLANNER_PROMPT.format(question=question),
            system=PLANNER_SYSTEM,
            temperature=0.0,
        )

        parsed = self.llm.extract_json(response)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            # Fallback: treat as simple retrieval
            return {
                "question_type": "simple",
                "strategy": "retrieve",
                "suggested_tool": "hybrid_search",
                "sub_problems": [],
                "reasoning": "Planner parse failed, defaulting to hybrid_search.",
            }

        # Validate and clamp sub_problems
        sub_problems = parsed.get("sub_problems", [])
        if not isinstance(sub_problems, list):
            sub_problems = []
        sub_problems = [
            s for s in sub_problems if isinstance(s, str) and s.strip()
        ][: self.max_sub_problems]

        # Whitelist-validate strategy
        strategy = parsed.get("strategy", "retrieve")
        if not isinstance(strategy, str) or strategy not in self.VALID_STRATEGIES:
            strategy = "retrieve"

        # If strategy says decompose but no sub_problems generated, fall back
        if strategy == "decompose" and not sub_problems:
            strategy = "retrieve"

        # Whitelist-validate question_type
        question_type = parsed.get("question_type", "simple")
        if not isinstance(question_type, str) or question_type not in self.VALID_TYPES:
            question_type = "simple"

        # Whitelist-validate suggested_tool
        suggested_tool = parsed.get("suggested_tool", "hybrid_search")
        if not isinstance(suggested_tool, str) or suggested_tool not in self.VALID_TOOLS:
            suggested_tool = "hybrid_search"

        return {
            "question_type": question_type,
            "strategy": strategy,
            "suggested_tool": suggested_tool,
            "sub_problems": sub_problems,
            "reasoning": parsed.get("reasoning", ""),
        }
