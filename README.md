<div align="center">

# Agentic RAG

**自适应检索 + 问题分解 + 知识图谱增强的智能问答系统**

*Agent 自主决定何时检索、用什么方式检索、结果是否充分，不够则自动改进。*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()

</div>

---

## TL;DR

一个轻量级 Agentic RAG 系统，整合了向量检索（FAISS）和知识图谱（NetworkX），通过 **8 状态有限状态机** 实现自适应检索和问题分解。Agent 自主判断检索结果是否充分，不够则自动 reformulate query 重新检索；复杂问题会被分解为子问题独立求解后综合。

> **核心发现**：Agent 在简单问题上开销极小（+1-2 次 LLM 调用），但在多跳和对比类问题上显著优于 Simple RAG。

---

## 系统架构

```
                        用户问题
                           │
                           ▼
                    ┌─────────────┐
                    │   PLANNING  │  LLM: 问题类型 + 策略 + 工具选择
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         [retrieve]   [decompose]   [direct]
              │            │            │
              ▼            ▼            │
       ┌───────────┐ ┌──────────┐       │
       │RETRIEVING │ │SOLVING_  │       │
       └─────┬─────┘ │SUB (循环)│       │
             │       └────┬─────┘       │
             ▼            │             │
       ┌───────────┐      ▼             │
       │REFLECTING │ ┌──────────┐       │
       └─────┬─────┘ │SYNTHESIZE│       │
             │       └────┬─────┘       │
      ┌──────┴──────┐     │             │
      ▼             ▼     ▼             ▼
[sufficient]  [reformulate]      ┌───────────┐
      │             │            │GENERATING │
      │             ▼            └───────────┘
      │        回到RETRIEVING
      │        (最多2轮)
      ▼
 ┌───────────┐
 │GENERATING │
 └───────────┘
```

### 三个核心能力

**自适应检索（Adaptive Retrieval）**
Agent 通过 Reflector 对检索结果打三个分（relevance / coverage / sufficiency），低于阈值时自动生成改进的 query 重新检索，最多 2 轮 reformulation。

**问题分解（Problem Decomposition）**
复杂问题在 Planning 阶段被拆分为 2-4 个原子子问题，每个子问题走简化的检索循环（最多 1 轮 reformulation），最后由 Synthesizer 综合子答案。

**工具箱选择（Tool Selection）**
Agent 有 vector_search / kg_query / kg_search / hybrid_search / direct_answer / calculate 共 6 个工具，由 Planner 和状态机按需选择。

---

## 项目结构

```
agentic-rag/
├── config/
│   └── settings.yaml              # DeepSeek API + 模型 + Agent 参数
├── data/
│   ├── documents.py               # 4 篇中文技术文档（知识库）
│   └── benchmark.json             # 60 道 QA 评测集 (20 simple + 20 bridge + 20 comparison)
├── src/
│   ├── core/
│   │   └── llm_client.py          # DeepSeek API + 4层 fallback JSON 解析器
│   ├── document/
│   │   ├── loader.py              # PDF/Markdown/TXT 文档加载
│   │   └── chunker.py             # 固定大小 + 递归分隔符分块
│   ├── rag/
│   │   ├── embedder.py            # BGE-small-zh 嵌入（可配置 query prefix）
│   │   ├── vector_store.py        # FAISS IndexFlatIP + MMR + save/load
│   │   └── retriever.py           # 端到端检索 + 上下文格式化
│   ├── kg/
│   │   ├── graph_store.py         # NetworkX MultiDiGraph 知识图谱
│   │   ├── extractor.py           # LLM 三元组提取（中英文双语 prompt）
│   │   ├── retriever.py           # 实体匹配 + BFS 图遍历 + 评分
│   │   └── hybrid.py              # RRF / 加权融合 (vector + KG)
│   ├── agent/
│   │   ├── planner.py             # 问题分析 + 策略决策 + 子问题生成
│   │   ├── reflector.py           # 上下文质量评分 + query reformulation
│   │   ├── synthesizer.py         # 子答案综合 + 最终生成
│   │   ├── tools.py               # 6 工具 ToolExecutor
│   │   └── state_machine.py       # 8 状态 FSM 运行器
│   ├── evaluation/
│   │   ├── metrics.py             # ROUGE-L / EM / F1 / Faithfulness / Agent 指标
│   │   └── benchmark.py           # 多系统对比评测运行器
│   └── pipeline.py                # AgenticRAGPipeline — 单入口编排器
├── notebooks/
│   ├── 01_Setup_and_Quick_Start   # 环境搭建 + 数据注入 + Quick Demo
│   └── 02_Agent_vs_Simple_RAG     # 60 题 Benchmark + 图表 + Case Study
├── demo/
│   └── app.py                     # Gradio 三系统对比 Demo
└── requirements.txt
```

---

## 快速开始

### Google Colab（推荐）

1. 打开 `notebooks/01_Setup_and_Quick_Start.ipynb`
2. 设置 `DEEPSEEK_API_KEY` 环境变量
3. 按顺序运行所有 cell

### 本地运行

```bash
git clone https://github.com/XIECHENG6/agentic-rag.git
cd agentic-rag
pip install -r requirements.txt
export OPENAI_API_KEY="your-deepseek-key"
export OPENAI_API_BASE="https://api.deepseek.com/v1"

# Python 中使用
from src.pipeline import AgenticRAGPipeline
pipeline = AgenticRAGPipeline(verbose=True)
pipeline.ingest_texts([("title", "content...")])
result = pipeline.ask("你的问题")
print(result["answer"])
```

---

## 评估方法

### Benchmark 数据集

60 道中文 QA，覆盖 4 个技术领域（QLoRA / ReAct / RAG / 小模型），分三类：

| 类型 | 数量 | 描述 | 预期 Agent 优势 |
|------|------|------|----------------|
| Simple | 20 | 单事实查询 | ≈ Simple RAG（开销小） |
| Bridge | 20 | 多跳推理（2-3 跳） | >> Simple RAG（+20-30pp） |
| Comparison | 20 | 概念对比 | >> Simple RAG（+20-35pp） |

### 对比系统

| 系统 | 描述 |
|------|------|
| `no_retrieval` | DeepSeek 直接回答（无检索） |
| `simple_rag` | 向量 Top-5 检索 + LLM 生成 |
| `hybrid_rag` | Vector + KG RRF 融合检索 + LLM 生成 |
| **`agentic_rag`** | **完整 Agent：自适应检索 + 问题分解** |

### 评估指标

ROUGE-L F1（字符级 LCS）、Exact Match、Token F1、Faithfulness、Answer Relevancy

---

## 与前期项目的关系

| 阶段 | 项目 | 方向 | 核心能力 |
|:----:|:-----|:-----|:---------|
| 1 | [small-llms-tool-use](https://github.com/XIECHENG6/small-llms-tool-use) | SFT Function Calling | 小模型工具调用 |
| 2 | [agenttune](https://github.com/XIECHENG6/agenttune) | SFT ReAct Agent | 多步推理 |
| 3 | [smallrag](https://github.com/XIECHENG6/smallrag) | RAG Pipeline | 端到端检索 |
| 4 | [CodeAgent-MCP](https://github.com/XIECHENG6/CodeAgent-MCP) | Multi-Agent Code Gen | MCP 集成 |
| 5 | [kg-agent](https://github.com/XIECHENG6/kg-agent) | KG-Enhanced Agent | 知识增强 |
| 6 | [smallllm-dpo](https://github.com/XIECHENG6/smallllm-dpo) | DPO 偏好对齐 | 负结果分析 |
| **7** | **agentic-rag** | **Agentic RAG** | **自适应检索 + 分解** |

**递进路线**：SFT (P1-2) → 检索增强 (P3) → 系统设计 (P4-5) → 对齐 (P6) → **编排 (P7)**

---

## 技术亮点

- **全自研编排**：8 状态 FSM 约 200 行代码，不依赖 LangChain/LangGraph
- **双检索后端**：FAISS 向量检索 + NetworkX 知识图谱，RRF 融合
- **自我反思循环**：Agent 评估上下文质量并自动改进查询
- **可解释执行轨迹**：每个状态转移都有详细日志
- **中文原生**：BGE 嵌入、中文 prompt、中文评测

---

<div align="center">

**License**: Apache-2.0

</div>
