"""Synthesizer — combine sub-answers into a coherent final answer."""

from src.core.llm_client import LLMClient

SYNTHESIZE_SYSTEM = """You are an answer synthesis agent. Your task is to combine multiple sub-answers into a single, coherent, comprehensive answer to the original question.

Rules:
- Merge information logically — do NOT just concatenate sub-answers
- Resolve any contradictions between sub-answers by preferring more specific information
- Ensure the final answer directly addresses the original question
- Be concise but complete — include all relevant facts from the sub-answers
- Write in the same language as the question (Chinese → Chinese, English → English)"""

SYNTHESIZE_PROMPT = """Combine these sub-answers into a unified answer.

Original question: {question}

Sub-questions and their answers:
{sub_qa_text}

Write a single, coherent answer to the original question:"""

# ---------- Final generation (with context) ----------

GENERATE_SYSTEM = """You are a knowledgeable assistant. Answer the question based on the provided context.

Rules:
- Base your answer on the retrieved context
- If the context doesn't contain enough information, say so honestly
- Be concise and direct — give the answer, not a lecture
- Write in the same language as the question"""

GENERATE_PROMPT = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""


class Synthesizer:
    """Combine sub-answers and generate final answers."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, question: str, sub_problems: list, sub_answers: list) -> str:
        """Combine sub-answers into a unified answer.

        Args:
            question: Original question.
            sub_problems: List of sub-question strings.
            sub_answers: List of sub-answer strings (same length).

        Returns:
            Synthesized answer string.
        """
        sub_qa_parts = []
        for i, (q, a) in enumerate(zip(sub_problems, sub_answers), 1):
            sub_qa_parts.append(f"  {i}. Q: {q}\n     A: {a}")
        sub_qa_text = "\n".join(sub_qa_parts)

        response = self.llm.generate_text(
            SYNTHESIZE_PROMPT.format(question=question, sub_qa_text=sub_qa_text),
            system=SYNTHESIZE_SYSTEM,
            temperature=0.0,
        )
        return response.strip()

    def generate_answer(self, question: str, context: str) -> str:
        """Generate final answer from accumulated context.

        Args:
            question: The question to answer.
            context: Formatted context string from retrieval.

        Returns:
            Answer string.
        """
        response = self.llm.generate_text(
            GENERATE_PROMPT.format(context=context[:4000], question=question),
            system=GENERATE_SYSTEM,
            temperature=0.0,
        )
        return response.strip()
