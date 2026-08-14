"""Agentic RAG state machine — 8-state FSM with dict-based transitions.

States:
    INIT → PLANNING → RETRIEVING → REFLECTING → (REFORMULATING)* → GENERATING
                    ↘ SOLVING_SUB* → SYNTHESIZING → GENERATING

The structure is fixed (state transitions), but decisions within each
state are made by LLM-powered tools (planner, reflector, synthesizer).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.core.llm_client import LLMClient
from src.agent.planner import Planner
from src.agent.reflector import Reflector
from src.agent.synthesizer import Synthesizer
from src.agent.tools import ToolExecutor


# ---------- Shared state object ----------

@dataclass
class AgentContext:
    """Mutable state shared across all state handlers."""

    question: str
    current_query: str = ""
    question_type: str = ""          # simple | multi_hop | comparison | multi_part
    strategy: str = ""               # retrieve | decompose | direct
    suggested_tool: str = "hybrid_search"
    lang: str = "zh"                 # query language for BGE prefix
    sub_problems: List[str] = field(default_factory=list)
    sub_answers: List[str] = field(default_factory=list)
    current_sub_index: int = 0
    retrieved_contexts: List[str] = field(default_factory=list)
    reflection_scores: Dict[str, float] = field(default_factory=dict)
    reformulation_count: int = 0
    max_reformulations: int = 2
    final_answer: str = ""
    trace: List[Dict] = field(default_factory=list)


# ---------- Transition map ----------

TRANSITIONS = {
    "INIT":          {"next": "PLANNING"},
    "PLANNING":      {"simple": "RETRIEVING", "decompose": "SOLVING_SUB", "direct": "GENERATING"},
    "RETRIEVING":    {"next": "REFLECTING"},
    "REFLECTING":    {"sufficient": "GENERATING", "reformulate": "REFORMULATING", "give_up": "GENERATING"},
    "REFORMULATING": {"next": "RETRIEVING"},
    "SOLVING_SUB":   {"next_sub": "SOLVING_SUB", "all_done": "SYNTHESIZING"},
    "SYNTHESIZING":  {"next": "GENERATING"},
    "GENERATING":    {},  # terminal
}


# ---------- State machine ----------

class AgenticRAG:
    """The main Agentic RAG agent.

    Usage::

        agent = AgenticRAG(llm, tools, planner, reflector, synthesizer)
        result = agent.run("你的问题")
        print(result["answer"])
        print(result["trace"])  # full execution trace
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolExecutor,
        planner: Planner,
        reflector: Reflector,
        synthesizer: Synthesizer,
        max_steps: int = 12,
        max_reformulations: int = 2,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = tools
        self.planner = planner
        self.reflector = reflector
        self.synthesizer = synthesizer
        self.max_steps = max_steps
        self.max_reformulations = max_reformulations
        self.verbose = verbose

    def run(self, question: str, verbose: bool = None, lang: str = "zh") -> dict:
        """Run the agent on a question.

        Returns:
            {
                "answer": str,
                "trace": list[dict],
                "question_type": str,
                "strategy": str,
                "sub_problems": list[str],
                "reformulations": int,
                "reflection_scores": dict,
                "status": "completed"|"max_steps_exceeded",
            }
        """
        verbose = verbose if verbose is not None else self.verbose
        ctx = AgentContext(
            question=question,
            current_query=question,
            max_reformulations=self.max_reformulations,
            lang=lang,
        )
        self._log(ctx, "INIT", {"question": question})

        state = "PLANNING"
        status = "completed"
        handlers = {
            "PLANNING":      self._handle_planning,
            "RETRIEVING":    self._handle_retrieving,
            "REFLECTING":    self._handle_reflecting,
            "REFORMULATING": self._handle_reformulating,
            "SOLVING_SUB":   self._handle_solving_sub,
            "SYNTHESIZING":  self._handle_synthesizing,
            "GENERATING":    self._handle_generating,
        }

        for step in range(self.max_steps):
            handler = handlers.get(state)
            if handler is None:
                break

            if verbose:
                print(f"\n{'='*50}")
                print(f"[Step {step+1}] State: {state}")
                print(f"{'='*50}")

            next_state = handler(ctx, verbose)
            state = next_state

            if state == "GENERATING":
                # Run the generating handler one last time
                self._handle_generating(ctx, verbose)
                break
        else:
            # max_steps exhausted without reaching GENERATING
            status = "max_steps_exceeded"
            if not ctx.final_answer:
                # Force a final answer from whatever context we have
                self._handle_generating(ctx, verbose)

        return {
            "answer": ctx.final_answer,
            "trace": ctx.trace,
            "question_type": ctx.question_type,
            "strategy": ctx.strategy,
            "sub_problems": ctx.sub_problems,
            "reformulations": ctx.reformulation_count,
            "reflection_scores": ctx.reflection_scores,
            "status": status,
            "context": "\n\n---\n\n".join(ctx.retrieved_contexts) if ctx.retrieved_contexts else "",
        }

    # ---------- State handlers ----------

    def _handle_planning(self, ctx: AgentContext, verbose: bool) -> str:
        plan = self.planner.plan(ctx.question)
        ctx.question_type = plan["question_type"]
        ctx.strategy = plan["strategy"]
        ctx.suggested_tool = plan["suggested_tool"]
        ctx.sub_problems = plan["sub_problems"]

        self._log(ctx, "PLANNING", plan)

        if verbose:
            print(f"  Type: {ctx.question_type} | Strategy: {ctx.strategy}")
            print(f"  Tool: {ctx.suggested_tool}")
            if ctx.sub_problems:
                for i, sp in enumerate(ctx.sub_problems, 1):
                    print(f"  Sub-{i}: {sp}")

        return TRANSITIONS["PLANNING"][ctx.strategy]

    def _handle_retrieving(self, ctx: AgentContext, verbose: bool) -> str:
        tool_name = ctx.suggested_tool
        query = ctx.current_query

        result = self.tools.get_retrieval_result_text(tool_name, query, top_k=5, lang=ctx.lang)
        ctx.retrieved_contexts.append(result)

        self._log(ctx, "RETRIEVING", {
            "tool": tool_name,
            "query": query,
            "result_length": len(result),
        })

        if verbose:
            print(f"  Tool: {tool_name} | Query: {query}")
            print(f"  Retrieved: {len(result)} chars")

        return TRANSITIONS["RETRIEVING"]["next"]

    def _handle_reflecting(self, ctx: AgentContext, verbose: bool) -> str:
        context = ctx.retrieved_contexts[-1] if ctx.retrieved_contexts else ""

        scores = self.reflector.evaluate(ctx.question, context)
        ctx.reflection_scores = {
            "relevance": scores["relevance"],
            "coverage": scores["coverage"],
            "sufficiency": scores["sufficiency"],
        }

        self._log(ctx, "REFLECTING", scores)

        if verbose:
            print(f"  Scores: rel={scores['relevance']:.2f} "
                  f"cov={scores['coverage']:.2f} suf={scores['sufficiency']:.2f}")
            print(f"  Judgment: {scores['judgment']}")

        judgment = scores["judgment"]

        if judgment == "sufficient":
            return TRANSITIONS["REFLECTING"]["sufficient"]

        # Override give_up: only allow if we've exhausted reformulation budget
        if ctx.reformulation_count >= ctx.max_reformulations:
            return TRANSITIONS["REFLECTING"]["give_up"]

        # Store reformulation info for the next state
        ctx._pending_strategy = scores.get("strategy", "expand")
        ctx._pending_new_query = scores.get("new_query")

        return TRANSITIONS["REFLECTING"]["reformulate"]

    def _handle_reformulating(self, ctx: AgentContext, verbose: bool) -> str:
        context = ctx.retrieved_contexts[-1] if ctx.retrieved_contexts else ""
        strategy = getattr(ctx, "_pending_strategy", "expand")
        pending_query = getattr(ctx, "_pending_new_query", None)

        if pending_query:
            # Use the reflector's suggested query directly (saves one LLM call)
            new_query = pending_query
        else:
            new_query = self.reflector.reformulate(
                ctx.question, ctx.current_query, context, strategy
            )

        ctx.current_query = new_query
        ctx.reformulation_count += 1

        self._log(ctx, "REFORMULATING", {
            "new_query": new_query,
            "strategy": strategy,
            "count": ctx.reformulation_count,
        })

        if verbose:
            print(f"  Strategy: {strategy}")
            print(f"  New query: {new_query}")
            print(f"  Reformulation #{ctx.reformulation_count}")

        return TRANSITIONS["REFORMULATING"]["next"]

    def _handle_solving_sub(self, ctx: AgentContext, verbose: bool) -> str:
        """Solve one sub-problem: retrieve → (optional reformulate) → generate sub-answer."""
        idx = ctx.current_sub_index
        sub_q = ctx.sub_problems[idx]

        if verbose:
            print(f"  Solving sub-problem {idx+1}/{len(ctx.sub_problems)}: {sub_q}")

        # Retrieve for this sub-problem
        result = self.tools.get_retrieval_result_text("hybrid_search", sub_q, top_k=5, lang=ctx.lang)

        # Quick reflect (1 reformulation max for sub-problems)
        scores = self.reflector.evaluate(sub_q, result)
        if scores["judgment"] == "reformulate":
            new_q = self.reflector.reformulate(sub_q, sub_q, result, "expand")
            result = self.tools.get_retrieval_result_text("hybrid_search", new_q, top_k=5, lang=ctx.lang)

        # Generate sub-answer
        sub_answer = self.synthesizer.generate_answer(sub_q, result)
        ctx.sub_answers.append(sub_answer)

        self._log(ctx, "SOLVING_SUB", {
            "sub_index": idx,
            "sub_question": sub_q,
            "sub_answer_length": len(sub_answer),
        })

        if verbose:
            print(f"  Sub-answer: {sub_answer[:100]}...")

        ctx.current_sub_index += 1

        if ctx.current_sub_index >= len(ctx.sub_problems):
            return TRANSITIONS["SOLVING_SUB"]["all_done"]
        return TRANSITIONS["SOLVING_SUB"]["next_sub"]

    def _handle_synthesizing(self, ctx: AgentContext, verbose: bool) -> str:
        answer = self.synthesizer.synthesize(
            ctx.question, ctx.sub_problems, ctx.sub_answers
        )
        ctx.final_answer = answer

        self._log(ctx, "SYNTHESIZING", {
            "num_sub_answers": len(ctx.sub_answers),
            "answer_length": len(answer),
        })

        if verbose:
            print(f"  Synthesized answer: {answer[:150]}...")

        return TRANSITIONS["SYNTHESIZING"]["next"]

    def _handle_generating(self, ctx: AgentContext, verbose: bool) -> str:
        """Generate final answer from accumulated context (or for direct strategy)."""
        if ctx.strategy == "direct":
            answer = self.tools.execute("direct_answer", {"question": ctx.question})
        elif ctx.final_answer:
            # Already have an answer from synthesis
            answer = ctx.final_answer
        else:
            # Combine all retrieved contexts
            all_context = "\n\n---\n\n".join(ctx.retrieved_contexts)
            answer = self.synthesizer.generate_answer(ctx.question, all_context)

        ctx.final_answer = answer

        self._log(ctx, "GENERATING", {
            "answer_length": len(answer),
        })

        if verbose:
            print(f"  Final answer: {answer[:200]}...")

        return "DONE"

    # ---------- Logging ----------

    @staticmethod
    def _log(ctx: AgentContext, state: str, data: dict):
        ctx.trace.append({"state": state, **data})
