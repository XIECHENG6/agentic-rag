"""Gradio demo for Agentic RAG — interactive comparison of 3 retrieval systems."""

import gradio as gr


def create_demo(pipeline):
    """Create a Gradio Interface for the Agentic RAG pipeline.

    Args:
        pipeline: An initialized AgenticRAGPipeline.

    Returns:
        gr.Blocks app (call .launch() on it).
    """

    def ask_simple_rag(question):
        result = pipeline.simple_rag(question)
        return result["answer"]

    def ask_hybrid_rag(question):
        result = pipeline.hybrid_rag(question)
        return result["answer"]

    def ask_agentic_rag(question):
        result = pipeline.ask(question, verbose=False)
        answer = result["answer"]

        # Build trace summary
        trace_lines = []
        for i, step in enumerate(result["trace"], 1):
            state = step.get("state", "?")
            if state == "PLANNING":
                trace_lines.append(f"[Plan] Type={step.get('question_type', '?')}, "
                                   f"Strategy={step.get('strategy', '?')}")
            elif state == "RETRIEVING":
                trace_lines.append(f"[Retrieve] Tool={step.get('tool', '?')}, "
                                   f"Query=\"{step.get('query', '?')[:50]}\"")
            elif state == "REFLECTING":
                trace_lines.append(f"[Reflect] rel={step.get('relevance', 0):.2f}, "
                                   f"cov={step.get('coverage', 0):.2f}, "
                                   f"suf={step.get('sufficiency', 0):.2f} → "
                                   f"{step.get('judgment', '?')}")
            elif state == "REFORMULATING":
                trace_lines.append(f"[Reformulate] New query: "
                                   f"\"{step.get('new_query', '?')[:60]}\"")
            elif state == "SOLVING_SUB":
                trace_lines.append(f"[Solve Sub-{step.get('sub_index', '?')+1}] "
                                   f"\"{step.get('sub_question', '?')[:50]}\"")
            elif state == "SYNTHESIZING":
                trace_lines.append("[Synthesize] Combined sub-answers")
            elif state == "GENERATING":
                trace_lines.append("[Generate] Final answer produced")

        trace_text = "\n".join(trace_lines)

        meta = (
            f"Question Type: {result.get('question_type', '?')}\n"
            f"Strategy: {result.get('strategy', '?')}\n"
            f"Reformulations: {result.get('reformulations', 0)}\n"
            f"Sub-problems: {len(result.get('sub_problems', []))}\n"
            f"Steps: {len(result['trace'])}"
        )

        return answer, trace_text, meta

    def compare_all(question):
        """Run all 3 systems and return results side by side."""
        simple = ask_simple_rag(question)
        hybrid = ask_hybrid_rag(question)
        agent_ans, agent_trace, agent_meta = ask_agentic_rag(question)
        return simple, hybrid, agent_ans, agent_trace, agent_meta

    # ---- Build UI ----

    with gr.Blocks(title="Agentic RAG Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🤖 Agentic RAG Demo\n"
            "对比 Simple RAG / Hybrid RAG / Agentic RAG 的回答质量。\n\n"
            "**Agentic RAG** 会自适应检索、自我反思、问题分解，展示完整执行轨迹。"
        )

        with gr.Tab("三系统对比"):
            question_input = gr.Textbox(
                label="输入问题",
                placeholder="例如：QLoRA中的NF4量化和传统INT4量化有什么区别？",
                lines=2,
            )
            compare_btn = gr.Button("对比回答", variant="primary")

            with gr.Row():
                simple_out = gr.Textbox(label="📎 Simple RAG", lines=6)
                hybrid_out = gr.Textbox(label="🔗 Hybrid RAG", lines=6)
            with gr.Row():
                agent_out = gr.Textbox(label="🤖 Agentic RAG", lines=6)
                agent_meta = gr.Textbox(label="Agent 元信息", lines=6)

            trace_out = gr.Textbox(label="Agent 执行轨迹", lines=8)

            compare_btn.click(
                compare_all,
                inputs=[question_input],
                outputs=[simple_out, hybrid_out, agent_out, trace_out, agent_meta],
            )

        with gr.Tab("Agentic RAG 单独"):
            q_input2 = gr.Textbox(label="输入问题", lines=2)
            ask_btn = gr.Button("提问", variant="primary")
            ans_out = gr.Textbox(label="回答", lines=6)
            trace_out2 = gr.Textbox(label="执行轨迹", lines=8)
            meta_out = gr.Textbox(label="元信息", lines=4)

            ask_btn.click(
                ask_agentic_rag,
                inputs=[q_input2],
                outputs=[ans_out, trace_out2, meta_out],
            )

        gr.Markdown(
            "---\n"
            "**架构**: 自研8状态FSM + 自适应检索 + 问题分解 | "
            "**项目系列**: smallrag → kg-agent → smallllm-dpo → **agentic-rag**\n\n"
            "[GitHub](https://github.com/XIECHENG6/agentic-rag)"
        )

        # Example questions
        gr.Examples(
            examples=[
                ["QLoRA中使用什么量化方法来减少显存占用？"],
                ["ReAct框架和Chain-of-Thought的主要区别是什么？"],
                ["对比RAG中的Top-K检索和MMR检索的优缺点"],
                ["QLoRA的双重量化技术额外节省了多少显存？这个技术和NF4量化分别优化了什么？"],
                ["小语言模型在Function Calling任务上的表现如何？和RAG技术有什么关联？"],
            ],
            inputs=[question_input],
        )

    return app
