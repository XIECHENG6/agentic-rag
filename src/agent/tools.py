"""Tool definitions + ToolExecutor for the Agentic RAG agent."""

import json

from src.core.llm_client import LLMClient
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.kg.graph_store import KnowledgeGraph
from src.kg.extractor import TripleExtractor
from src.kg.retriever import GraphRetriever
from src.kg.hybrid import HybridRetriever
from src.agent.planner import Planner
from src.agent.reflector import Reflector
from src.agent.synthesizer import Synthesizer


# ---------- Tool schema (for documentation / future LLM tool-use) ----------

TOOL_DEFINITIONS = [
    {
        "name": "vector_search",
        "description": "Semantic similarity search over document chunks. Best for general concept questions.",
        "parameters": {"query": "str", "top_k": "int=5"},
    },
    {
        "name": "kg_query",
        "description": "Structured KG lookup by entity name. Best for questions about named entities.",
        "parameters": {"entity": "str", "hops": "int=2"},
    },
    {
        "name": "kg_search",
        "description": "Fuzzy keyword search for entities in the knowledge graph.",
        "parameters": {"keyword": "str"},
    },
    {
        "name": "hybrid_search",
        "description": "Combined vector + KG retrieval via RRF fusion. Best for complex questions.",
        "parameters": {"query": "str", "top_k": "int=5"},
    },
    {
        "name": "direct_answer",
        "description": "LLM answers from parametric knowledge without retrieval.",
        "parameters": {"question": "str"},
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression safely.",
        "parameters": {"expression": "str"},
    },
]


class ToolExecutor:
    """Dispatch tool calls to the appropriate handler.

    Wires together all retrieval backends + LLM-powered reasoning tools.
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        graph_retriever: GraphRetriever,
        hybrid_retriever: HybridRetriever,
        kg: KnowledgeGraph,
        planner: Planner,
        reflector: Reflector,
        synthesizer: Synthesizer,
    ):
        self.llm = llm
        self.retriever = retriever
        self.graph_retriever = graph_retriever
        self.hybrid_retriever = hybrid_retriever
        self.kg = kg
        self.planner = planner
        self.reflector = reflector
        self.synthesizer = synthesizer

        self.dispatch = {
            "vector_search": self._vector_search,
            "kg_query": self._kg_query,
            "kg_search": self._kg_search,
            "hybrid_search": self._hybrid_search,
            "direct_answer": self._direct_answer,
            "calculate": self._calculate,
        }

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool by name with the given arguments.

        Returns a formatted string result.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments: {arguments}"

        handler = self.dispatch.get(tool_name)
        if not handler:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            return handler(**arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    # ---------- Retrieval tools ----------

    def _vector_search(self, query: str, top_k: int = 5, lang: str = "zh") -> str:
        results = self.retriever.retrieve(query, top_k=top_k, method="topk", lang=lang)
        if not results:
            return "No relevant passages found via vector search."
        lines = ["Vector search results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  [{i}] (score: {r['score']:.3f}) {r['text']}")
        return "\n".join(lines)

    def _kg_query(self, entity: str, hops: int = 2) -> str:
        if not self.kg.has_entity(entity):
            matches = self.kg.search_entities(entity, threshold=0.6)
            if matches:
                suggestions = ", ".join(f"'{m[0]}'" for m in matches[:5])
                return f"Entity '{entity}' not found. Did you mean: {suggestions}?"
            return f"Entity '{entity}' not found in knowledge graph."
        return self.kg.get_subgraph_text(entity, hops=hops)

    def _kg_search(self, keyword: str) -> str:
        matches = self.kg.search_entities(keyword)
        if not matches:
            return f"No entities matching '{keyword}' found."
        lines = [f"Entities matching '{keyword}':"]
        for name, score in matches[:10]:
            edges = self.kg.get_entity_edges(name)
            lines.append(f"  - {name} (relevance: {score:.2f}, {len(edges)} connections)")
        return "\n".join(lines)

    def _hybrid_search(self, query: str, top_k: int = 5, lang: str = "zh") -> str:
        results = self.hybrid_retriever.retrieve(query, top_k=top_k, lang=lang)
        if not results:
            return "No relevant context found via hybrid search."
        lines = ["Hybrid search results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  [{i}] (score: {r['score']:.4f}) {r['text']}")
        return "\n".join(lines)

    # ---------- Reasoning tools ----------

    def _direct_answer(self, question: str) -> str:
        """LLM answers from parametric knowledge (no retrieval)."""
        response = self.llm.generate_text(
            f"Answer this question concisely:\n\n{question}",
            system="You are a knowledgeable assistant. Answer concisely and directly.",
            temperature=0.0,
        )
        return response.strip()

    def _calculate(self, expression: str) -> str:
        """Safely evaluate a math expression using AST whitelist."""
        import ast
        if len(expression) > 200:
            return f"Error: Expression too long ({len(expression)} chars, max 200)"
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return f"Error: Invalid expression '{expression}': {e}"

        # Validate AST: only allow numbers and basic arithmetic
        _ALLOWED_NODES = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.FloorDiv, ast.UAdd, ast.USub,
        )
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return f"Error: Disallowed operation in '{expression}'"
            # Reject non-numeric constants (strings, booleans, None, etc.)
            if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
                return f"Error: Non-numeric constant in '{expression}'"
            # Cap Pow exponent to prevent resource exhaustion
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exp_val = self._ast_num_value(node.right)
                if exp_val is None or abs(exp_val) > 100:
                    return f"Error: Exponent {exp_val} exceeds limit (max 100)"

        try:
            code = compile(tree, "<calc>", "eval")
            result = eval(code)  # noqa: S307 — AST validated
            if isinstance(result, (int, float)) and abs(result) > 1e18:
                return f"Error: Result magnitude {result:.2e} exceeds limit (1e18)"
            return str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"

    @staticmethod
    def _ast_num_value(node) -> float:
        """Extract numeric value from an AST node, or None if not a literal."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        # Python 3.7 compat
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = ToolExecutor._ast_num_value(node.operand)
            return -inner if inner is not None else None
        return None

    # ---------- Utility ----------

    # Map each retrieval tool to the parameter names it expects
    _RETRIEVAL_PARAM_MAP = {
        "vector_search":  {"query_key": "query",  "supports_top_k": True,  "supports_lang": True},
        "kg_query":       {"query_key": "entity", "supports_top_k": False, "extra": {"hops": 2}},
        "kg_search":      {"query_key": "keyword", "supports_top_k": False},
        "hybrid_search":  {"query_key": "query",  "supports_top_k": True,  "supports_lang": True},
    }

    def get_retrieval_result_text(self, tool_name: str, query: str, top_k: int = 5, lang: str = "zh") -> str:
        """Execute a retrieval tool and return the raw text result.

        Maps the generic (query, top_k) call to each tool's specific signature.
        Used by the state machine's RETRIEVING state.
        """
        spec = self._RETRIEVAL_PARAM_MAP.get(tool_name)
        if spec is None:
            # Fallback: pass query/top_k as-is
            return self.execute(tool_name, {"query": query, "top_k": top_k})

        args = {spec["query_key"]: query}
        if spec.get("supports_top_k"):
            args["top_k"] = top_k
        # Pass lang to tools that support it (vector_search, hybrid_search)
        if spec.get("supports_lang"):
            args["lang"] = lang
        args.update(spec.get("extra", {}))
        return self.execute(tool_name, args)

    def get_available_tools(self):
        return TOOL_DEFINITIONS
