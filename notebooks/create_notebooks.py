"""Generate two Agentic RAG Colab notebooks as valid .ipynb files."""
import json
import os

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": source.splitlines(keepends=True)}

def build_notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": METADATA, "cells": cells}

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1: Setup & Quick Start
# ─────────────────────────────────────────────────────────────────────────────

n01_cells = [
    md(
        "# Agentic RAG — Setup & Quick Start\n"
        "\n"
        "本 Notebook 是 Agentic RAG 项目的入口点，带你完成：\n"
        "1. 环境准备（GPU 验证 + 依赖安装）\n"
        "2. 知识库文档加载\n"
        "3. 向量索引 & 知识图谱构建\n"
        "4. 三种 RAG 模式 Quick Demo\n"
        "5. Agent 执行轨迹可视化\n"
        "6. 知识图谱可视化\n"
    ),
    md("## 第一步：环境准备"),
    code(
        "# GPU 验证\n"
        "import torch\n"
        "print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\")\n"
        "print(f\"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\" if torch.cuda.is_available() else \"\")\n"
        "\n"
        "# 安装依赖\n"
        "!pip install -q openai sentence-transformers faiss-cpu networkx numpy pandas matplotlib gradio pyyaml tqdm pymupdf\n"
    ),
    code(
        "import os\n"
        "import sys\n"
        "\n"
        "# 克隆项目（Colab中）\n"
        "if not os.path.exists(\"agentic-rag\"):\n"
        "    !git clone https://github.com/XIECHENG6/agentic-rag.git\n"
        "    # 或者直接上传项目文件\n"
        "\n"
        "os.chdir(\"agentic-rag\")\n"
        "sys.path.insert(0, \".\")\n"
        "\n"
        "# API Key 配置\n"
        "from google.colab import userdata\n"
        "os.environ[\"OPENAI_API_KEY\"] = userdata.get(\"DEEPSEEK_API_KEY\")\n"
        "os.environ[\"OPENAI_API_BASE\"] = \"https://api.deepseek.com/v1\"\n"
        "print(\"API key configured \\u2713\")\n"
    ),
    md("## 第二步：加载知识库文档"),
    code(
        "from data.documents import DOCUMENTS\n"
        "\n"
        "print(f\"知识库文档数: {len(DOCUMENTS)}\")\n"
        "for title, content in DOCUMENTS:\n"
        "    print(f\"  \\U0001f4c4 {title}: {len(content)} 字符\")\n"
    ),
    md("## 第三步：构建向量索引 + 知识图谱"),
    code(
        "from src.pipeline import AgenticRAGPipeline\n"
        "\n"
        "pipeline = AgenticRAGPipeline(verbose=True)\n"
        "\n"
        "# 将文档注入pipeline\n"
        "pipeline.ingest_texts(DOCUMENTS)\n"
        "\n"
        "# 查看统计\n"
        "stats = pipeline.stats()\n"
        "print(f\"\\n\\U0001f4ca Pipeline统计:\")\n"
        "for k, v in stats.items():\n"
        "    print(f\"  {k}: {v}\")\n"
    ),
    md("## 第四步：Quick Demo — 三种模式对比"),
    code(
        "demo_questions = [\n"
        "    \"QLoRA中使用的NF4量化和传统INT4量化有什么区别？\",\n"
        "    \"ReAct框架和Chain-of-Thought的主要区别是什么？\",\n"
        "    \"对比RAG中的Top-K检索和MMR检索的优缺点\",\n"
        "]\n"
        "\n"
        "print(\"=\" * 70)\n"
        "for q in demo_questions:\n"
        "    print(f\"\\n\\u2753 问题: {q}\")\n"
        "    print(\"-\" * 50)\n"
        "    \n"
        "    # Simple RAG\n"
        "    simple = pipeline.simple_rag(q)\n"
        "    print(f\"\\n\\U0001f4ce Simple RAG:\\n{simple['answer'][:200]}\")\n"
        "    \n"
        "    # Hybrid RAG\n"
        "    hybrid = pipeline.hybrid_rag(q)\n"
        "    print(f\"\\n\\U0001f517 Hybrid RAG:\\n{hybrid['answer'][:200]}\")\n"
        "    \n"
        "    # Agentic RAG (with trace)\n"
        "    result = pipeline.ask(q, verbose=True)\n"
        "    print(f\"\\n\\U0001f916 Agentic RAG:\\n{result['answer'][:200]}\")\n"
        "    print(f\"   Type: {result['question_type']} | Strategy: {result['strategy']}\")\n"
        "    print(f\"   Reformulations: {result['reformulations']}\")\n"
        "    print(\"=\" * 70)\n"
    ),
    md("## 第五步：可视化 Agent 执行轨迹"),
    code(
        "import json\n"
        "\n"
        "# 选一个多跳问题展示完整trace\n"
        "complex_q = \"QLoRA的双重量化技术额外节省了多少显存？这个技术和NF4量化分别优化了模型权重的哪个方面？\"\n"
        "result = pipeline.ask(complex_q, verbose=False)\n"
        "\n"
        "print(f\"\\u2753 问题: {complex_q}\")\n"
        "print(f\"\\n\\U0001f916 答案:\\n{result['answer']}\")\n"
        "print(f\"\\n\\U0001f4ca 执行轨迹 ({len(result['trace'])} 步):\")\n"
        "for i, step in enumerate(result['trace'], 1):\n"
        "    state = step.get('state', '?')\n"
        "    print(f\"\\n  Step {i} [{state}]:\")\n"
        "    for k, v in step.items():\n"
        "        if k != 'state' and isinstance(v, str) and len(v) > 200:\n"
        "            print(f\"    {k}: {v[:200]}...\")\n"
        "        elif k != 'state':\n"
        "            print(f\"    {k}: {v}\")\n"
    ),
    md("## 第六步：知识图谱可视化"),
    code(
        "import os\n"
        "fig = pipeline.kg.to_matplotlib(figsize=(16, 12))\n"
        "os.makedirs(\"results/figures\", exist_ok=True)\n"
        "fig.savefig(\"results/figures/knowledge_graph.png\", dpi=150, bbox_inches=\"tight\")\n"
        "print(\"知识图谱已保存\")\n"
    ),
    md(
        "## 完成！下一步\n"
        "\n"
        "- [02_Agent_vs_Simple_RAG.ipynb](02_Agent_vs_Simple_RAG.ipynb) — 完整基准测试（80题 × 4系统）\n"
        "- [demo/app.py](../demo/app.py) — 交互式 Gradio 界面对比三种 RAG 系统\n"
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2: Agent vs Simple RAG Benchmark
# ─────────────────────────────────────────────────────────────────────────────

SETUP_CELL = code(
    "# GPU 验证\n"
    "import torch\n"
    "print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\")\n"
    "print(f\"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\" if torch.cuda.is_available() else \"\")\n"
    "\n"
    "# 安装依赖\n"
    "!pip install -q openai sentence-transformers faiss-cpu networkx numpy pandas matplotlib gradio pyyaml tqdm pymupdf\n"
    "\n"
    "import os\n"
    "import sys\n"
    "\n"
    "# 克隆项目（Colab中）\n"
    "if not os.path.exists(\"agentic-rag\"):\n"
    "    !git clone https://github.com/XIECHENG6/agentic-rag.git\n"
    "\n"
    "os.chdir(\"agentic-rag\")\n"
    "sys.path.insert(0, \".\")\n"
    "\n"
    "# API Key 配置\n"
    "from google.colab import userdata\n"
    "os.environ[\"OPENAI_API_KEY\"] = userdata.get(\"DEEPSEEK_API_KEY\")\n"
    "os.environ[\"OPENAI_API_BASE\"] = \"https://api.deepseek.com/v1\"\n"
    "print(\"API key configured \\u2713\")\n"
)

PIPELINE_CELL = code(
    "from data.documents import DOCUMENTS\n"
    "from src.pipeline import AgenticRAGPipeline\n"
    "\n"
    "pipeline = AgenticRAGPipeline(verbose=True)\n"
    "pipeline.ingest_texts(DOCUMENTS)\n"
    "\n"
    "stats = pipeline.stats()\n"
    "print(f\"Pipeline ready — {stats}\")\n"
)

n02_cells = [
    md(
        "# Agentic RAG — Agent vs Simple RAG Benchmark\n"
        "\n"
        "本 Notebook 对比四种 RAG 系统在 80 道基准题上的表现：\n"
        "- **No Retrieval** — 纯 LLM，无检索增强\n"
        "- **Simple RAG** — Top-K 向量检索 + LLM\n"
        "- **Hybrid RAG** — 向量检索 + 知识图谱子图增强\n"
        "- **Agentic RAG** — 自主 Agent 动态规划检索策略\n"
    ),
    md("## 第一步：环境准备（同 Notebook 01）"),
    SETUP_CELL,
    PIPELINE_CELL,
    md("## 第二步：加载 Benchmark 数据集"),
    code(
        "import json\n"
        "\n"
        "with open(\"data/benchmark.json\", \"r\", encoding=\"utf-8\") as f:\n"
        "    benchmark = json.load(f)\n"
        "\n"
        "types = {}\n"
        "for q in benchmark:\n"
        "    t = q.get(\"type\", \"?\")\n"
        "    types[t] = types.get(t, 0) + 1\n"
        "\n"
        "print(f\"Benchmark: {len(benchmark)} QA pairs\")\n"
        "for t, n in sorted(types.items()):\n"
        "    print(f\"  {t}: {n}\")\n"
        "\n"
        "# Show 2 examples from each type\n"
        "for qtype in [\"simple\", \"bridge\", \"comparison\"]:\n"
        "    examples = [q for q in benchmark if q[\"type\"] == qtype][:2]\n"
        "    print(f\"\\n{'='*50}\")\n"
        "    print(f\"  {qtype} 示例:\")\n"
        "    for ex in examples:\n"
        "        print(f\"    Q: {ex['question']}\")\n"
        "        print(f\"    A: {ex['answer'][:100]}\")\n"
    ),
    md("## 第三步：运行基准测试"),
    code(
        "from src.evaluation.benchmark import BenchmarkRunner\n"
        "\n"
        "import os\n"
        "input_cost_per_1m = float(os.getenv(\"LLM_INPUT_COST_PER_1M\", \"0\"))\n"
        "output_cost_per_1m = float(os.getenv(\"LLM_OUTPUT_COST_PER_1M\", \"0\"))\n"
        "runner = BenchmarkRunner(pipeline, verbose=True, input_cost_per_1m=input_cost_per_1m, output_cost_per_1m=output_cost_per_1m)\n"
        "runner.load_benchmark(\"data/benchmark.json\")\n"
        "\n"
        "# 运行所有系统\n"
        "results = runner.run_all(\n"
        "    systems=[\"no_retrieval\", \"simple_rag\", \"hybrid_rag\", \"agentic_rag\"],\n"
        "    verbose=True,\n"
        ")\n"
    ),
    md("## 第四步：结果汇总"),
    code("runner.print_summary(results)"),
    code(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "systems = list(results.keys())\n"
        "rouge_scores = [results[s][\"metrics\"][\"rouge_l\"] for s in systems]\n"
        "em_scores = [results[s][\"metrics\"][\"exact_match\"] for s in systems]\n"
        "f1_scores = [results[s][\"metrics\"][\"f1\"] for s in systems]\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "for ax, scores, title in zip(axes, [rouge_scores, em_scores, f1_scores],\n"
        "                              [\"ROUGE-L\", \"Exact Match\", \"Token F1\"]):\n"
        "    bars = ax.bar(range(len(systems)), scores, color=[\"#e74c3c\", \"#3498db\", \"#2ecc71\", \"#9b59b6\"])\n"
        "    ax.set_xticks(range(len(systems)))\n"
        "    ax.set_xticklabels([s.replace(\"_\", \"\\n\") for s in systems], fontsize=8)\n"
        "    ax.set_title(title)\n"
        "    ax.set_ylim(0, 1)\n"
        "    for bar, score in zip(bars, scores):\n"
        "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n"
        "                f\"{score:.3f}\", ha=\"center\", va=\"bottom\", fontsize=9)\n"
        "\n"
        "plt.tight_layout()\n"
        "import os; os.makedirs(\"results/figures\", exist_ok=True)\n"
        "plt.savefig(\"results/figures/benchmark_comparison.png\", dpi=150, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    code(
        "types = [\"simple\", \"bridge\", \"comparison\"]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "colors = [\"#e74c3c\", \"#3498db\", \"#2ecc71\", \"#9b59b6\"]\n"
        "\n"
        "for ax, qtype in zip(axes, types):\n"
        "    scores = []\n"
        "    for system in systems:\n"
        "        per_type = results[system].get(\"per_type_metrics\", {})\n"
        "        if qtype in per_type:\n"
        "            scores.append(per_type[qtype][\"rouge_l\"])\n"
        "        else:\n"
        "            scores.append(0)\n"
        "    \n"
        "    bars = ax.bar(range(len(systems)), scores, color=colors)\n"
        "    ax.set_xticks(range(len(systems)))\n"
        "    ax.set_xticklabels([s.replace(\"_\", \"\\n\") for s in systems], fontsize=8)\n"
        "    ax.set_title(f\"{qtype.title()} Questions (ROUGE-L)\")\n"
        "    ax.set_ylim(0, 1)\n"
        "    for bar, score in zip(bars, scores):\n"
        "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n"
        "                f\"{score:.3f}\", ha=\"center\", va=\"bottom\", fontsize=9)\n"
        "\n"
        "plt.suptitle(\"Agentic RAG: Per-Type Performance\", fontsize=14)\n"
        "plt.tight_layout()\n"
        "import os; os.makedirs(\"results/figures\", exist_ok=True)\n"
        "plt.savefig(\"results/figures/per_type_comparison.png\", dpi=150, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md("## 第五步：Case Study — Agent 优势案例"),
    code(
        "# 找出Agent比Simple RAG好最多的问题\n"
        "agent_results = results[\"agentic_rag\"][\"results\"]\n"
        "simple_results = results[\"simple_rag\"][\"results\"]\n"
        "\n"
        "improvements = []\n"
        "for i in range(len(benchmark)):\n"
        "    from src.evaluation.metrics import compute_rouge_l\n"
        "    agent_score = compute_rouge_l(agent_results[i][\"answer\"], benchmark[i][\"answer\"])\n"
        "    simple_score = compute_rouge_l(simple_results[i][\"answer\"], benchmark[i][\"answer\"])\n"
        "    diff = agent_score - simple_score\n"
        "    improvements.append((i, diff, agent_score, simple_score))\n"
        "\n"
        "improvements.sort(key=lambda x: x[1], reverse=True)\n"
        "\n"
        "# Show top 3 improvements\n"
        "print(\"\\U0001f3c6 Agent 优势最大的 3 个问题:\\n\")\n"
        "for idx, diff, a_score, s_score in improvements[:3]:\n"
        "    q = benchmark[idx]\n"
        "    print(f\"\\U0001f4cc [{q['type']}] {q['question']}\")\n"
        "    print(f\"   Simple RAG (ROUGE-L: {s_score:.3f}): {simple_results[idx]['answer'][:100]}...\")\n"
        "    print(f\"   Agentic RAG (ROUGE-L: {a_score:.3f}): {agent_results[idx]['answer'][:100]}...\")\n"
        "    print(f\"   参考答案: {q['answer'][:100]}\")\n"
        "    print(f\"   提升: +{diff:.3f}\")\n"
        "    print()\n"
    ),
    code(
        "avg_times = {s: results[s][\"metrics\"][\"avg_time\"] for s in systems}\n"
        "print(\"\\u23f1\\ufe0f 平均响应时间:\")\n"
        "for s, t in avg_times.items():\n"
        "    print(f\"  {s}: {t:.1f}s\")\n"
    ),
    md("## 第六步：调用成本、来源召回率与失败案例"),
    code(
        "from collections import Counter\n"
        "\n"
        "for system, data in results.items():\n"
        "    metrics = data[\"metrics\"]\n"
        "    print()\n"
        "    print(f\"{system}\")\n"
        "    print(f\"  source_recall={metrics.get(\'source_recall\', 0):.3f}\")\n"
        "    print(f\"  llm_calls={metrics.get(\'llm_calls\', 0):.0f}\")\n"
        "    print(f\"  prompt_tokens={metrics.get(\'llm_prompt_tokens\', 0):.0f}\")\n"
        "    print(f\"  completion_tokens={metrics.get(\'llm_completion_tokens\', 0):.0f}\")\n"
        "    print(f\"  estimated_cost_usd={metrics.get(\'llm_estimated_cost_usd\', 0):.4f}\")\n"
        "    failures = Counter(item[\'failure_type\'] for item in data.get(\'failure_cases\', []))\n"
        "    print(f\"  failures={dict(failures)}\")\n"
        "\n"
        "print()\n"
        "print(\"Agentic RAG failure cases (max 5):\")\n"
        "for case in results.get(\'agentic_rag\', {}).get(\'failure_cases\', [])[:5]:\n"
        "    print(f\"- [{case[\'failure_type\']}] {case[\'question\']}\")\n"
        "    print(f\"  answer={case[\'answer\'][:160]}\")\n"
    ),
    md("## 第七步：保存结果"),

    code(
        "BenchmarkRunner.save_results(results, \"results/benchmark_results.json\")\n"
        "print(\"结果已保存到 results/benchmark_results.json\")\n"
        "\n"
        "# 可选：保存到Google Drive\n"
        "from google.colab import drive\n"
        "drive.mount(\"/content/drive\")\n"
        "import shutil\n"
        "shutil.copy(\"results/benchmark_results.json\", \"/content/drive/MyDrive/agentic_rag_results.json\")\n"
        "print(\"已复制到 Google Drive \\u2713\")\n"
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Write to disk
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

for name, cells in [
    ("01_Setup_and_Quick_Start.ipynb", n01_cells),
    ("02_Agent_vs_Simple_RAG.ipynb",   n02_cells),
]:
    path = os.path.join(OUT_DIR, name)
    nb = build_notebook(cells)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}  ({len(cells)} cells)")

print("Done.")
