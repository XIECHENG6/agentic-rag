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

默认分块长度按 embedding 模型的 token 计数（`chunking.unit: tokens`），也可切换为字符计数；每次问答还受 `agent.max_llm_calls` 的硬调用预算限制。

> **核心发现（80 题 benchmark）**：在模型参数化知识答不准的 hard 题上，Agentic RAG 的语义评分（LLM-as-Judge）降幅仅 0.020，而纯 LLM 降 0.091；Source Recall 0.988、Answer Relevancy 0.715，均为四个对比系统最高。完整数据见 `results/`。

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
│   └── benchmark.json             # 80 道 QA 评测集 (27 simple + 27 bridge + 26 comparison；easy 8 / medium 34 / hard 38)
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
│   └── 02_Agent_vs_Simple_RAG     # 80 题 Benchmark + 图表 + Case Study
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

80 道中文 QA，覆盖 4 个技术领域（QLoRA / ReAct / RAG / 小模型）：

| 题型 | 数量 | 描述 |
|------|------|------|
| Simple | 27 | 单事实查询 |
| Bridge | 27 | 多跳推理（2-3 跳） |
| Comparison | 26 | 概念对比 |

难度标注为 easy 8 / medium 34 / hard 38。其中 20 道新增题全为 hard，聚焦精确数字（如具体评测得分）与跨文档推理——模型参数化知识答不准的区间，用于检验检索系统的真实价值。

### 对比系统

| 系统 | 描述 |
|------|------|
| `no_retrieval` | DeepSeek 直接回答（无检索） |
| `simple_rag` | 向量 Top-5 检索 + LLM 生成 |
| `hybrid_rag` | Vector + KG RRF 融合检索 + LLM 生成 |
| **`agentic_rag`** | **完整 Agent：自适应检索 + 问题分解** |

### 评估指标

ROUGE-L F1（字符级 LCS，输入已归一化）、Exact Match、Token F1、Faithfulness、Answer Relevancy、Source Recall、LLM-as-Judge（1-5 分映射到 [0,1]）

### 运行成本、来源召回与失败案例

BenchmarkRunner 现在会为每道题记录：

- LLM 调用次数、Prompt/Completion/Total tokens、请求耗时
- 可配置单价后的估算美元成本
- 基于 benchmark 的 `source_docs` 计算来源级召回率
- `api_or_runtime_error`、`max_steps_exceeded`、`empty_answer`、`retrieval_miss`、`answer_mismatch` 失败类型

在 Colab 中运行 Benchmark 前设置当前供应商的价格（单位：美元/百万 tokens）：

```python
import os
os.environ["LLM_INPUT_COST_PER_1M"] = "your-input-price"
os.environ["LLM_OUTPUT_COST_PER_1M"] = "your-output-price"
```

如果价格未设置，token 和调用次数仍会统计，但估算成本显示为 0；不要把 0 解释为真实免费。

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
