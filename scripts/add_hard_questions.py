"""Add 20 hard benchmark questions that require knowledge base retrieval."""
import json
import os

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark.json")

NEW_QUESTIONS = [
    # === Simple-Hard (exact numbers, hard to guess) ===
    {
        "question": "QLoRA的双重量化技术对一个3B模型具体节省了多少显存？请给出每参数节省位数和总节省量。",
        "answer": "QLoRA的双重量化技术将NF4量化的FP32缩放因子进一步量化为8位，每个参数额外节省约0.37位。对于一个3B模型（约4700万个FP32缩放因子），总共额外节省约140MB显存。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["qlora_intro.md"]
    },
    {
        "question": "NF4量化和INT4量化在MMLU基准上的具体性能差距是多少？困惑度差距呢？",
        "answer": "根据消融实验，NF4量化在MMLU基准上比INT4量化高约1.5个百分点，困惑度降低0.3到0.5个点。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["qlora_intro.md"]
    },
    {
        "question": "bge-large-zh-v1.5的参数量和嵌入维度分别是多少？",
        "answer": "bge-large-zh-v1.5的参数量约为3.26亿（326M），嵌入维度为1024维。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["rag_overview.md"]
    },
    {
        "question": "QLoRA论文的第一作者是谁？发表在哪个会议？",
        "answer": "QLoRA论文由Tim Dettmers等人撰写，发表在NeurIPS 2023会议上。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["qlora_intro.md"]
    },
    {
        "question": "Paged Optimizer在GPU显存使用率达到多少时触发？对训练速度的影响有多大？",
        "answer": "Paged Optimizer在GPU显存使用率达到95%以上时触发，将优化器状态分页卸载到CPU内存，对训练速度的影响不到5%。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["qlora_intro.md"]
    },
    {
        "question": "Qwen2.5-7B-Instruct在MMLU和HumanEval上的具体得分是多少？",
        "answer": "Qwen2.5-7B-Instruct在MMLU上得分约70.3，在HumanEval上得分约79.4。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["small_lm_trends.md"]
    },
    {
        "question": "使用QLoRA微调一个33B模型需要多少显存？相比全参数微调节省了多少百分比？",
        "answer": "使用QLoRA微调33B模型约需48GB显存，而全参数微调约需320GB显存，节省约85%。",
        "type": "simple", "difficulty": "hard", "hops": 1,
        "source_docs": ["qlora_intro.md"]
    },

    # === Bridge-Hard (cross-document reasoning) ===
    {
        "question": "如果我要用Qwen2.5-3B搭建一个RAG系统，同时使用bge-small-zh-v1.5作为嵌入模型，推理时总共大约需要多少显存？",
        "answer": "Qwen2.5-3B在FP16下约需6GB显存（INT4量化后1.5-2GB），bge-small-zh-v1.5约需200MB显存，RAG系统推理时总共约需2-6GB显存（取决于是否量化）。",
        "type": "bridge", "difficulty": "hard", "hops": 2,
        "source_docs": ["small_lm_trends.md", "rag_overview.md"]
    },
    {
        "question": "3B规模的小模型做多步Function Calling的准确率是多少？相比GPT-4有多大差距？",
        "answer": "3B规模模型的单工具调用准确率可达85%以上，但4步以上的多工具链准确率降至60-70%，而GPT-4在相同多工具任务上可达90%以上，差距约20-30个百分点。",
        "type": "bridge", "difficulty": "hard", "hops": 2,
        "source_docs": ["react_agent.md", "small_lm_trends.md"]
    },
    {
        "question": "QLoRA微调3B模型需要8GB显存，那么用同一张卡能同时运行bge-large-zh-v1.5做RAG检索吗？",
        "answer": "bge-large-zh-v1.5有约3.26亿参数，FP16下约需650MB显存。加上QLoRA微调3B模型约需8GB显存，总共约需8.65GB，在8GB显卡上无法同时运行，需要16GB以上的显卡或使用bge-small-zh（约200MB）替代。",
        "type": "bridge", "difficulty": "hard", "hops": 2,
        "source_docs": ["qlora_intro.md", "rag_overview.md"]
    },
    {
        "question": "ReAct框架是在哪一年提出的？和Function Calling的推出时间相比哪个更早？",
        "answer": "ReAct框架由Shunyu Yao等人在2022年提出，Function Calling由OpenAI在2023年6月随GPT-3.5-turbo和GPT-4 API首次引入。因此ReAct比Function Calling早约半年到一年。",
        "type": "bridge", "difficulty": "hard", "hops": 2,
        "source_docs": ["react_agent.md"]
    },
    {
        "question": "RAG系统中FAISS的IndexFlatIP适合多大规模的数据？超过这个规模应该用什么索引？",
        "answer": "FAISS的IndexFlatIP适合10万（100K）以下的向量规模。超过1000万（10M）向量时，推荐使用IndexIVFPQ索引，可实现30倍以上的压缩比。",
        "type": "bridge", "difficulty": "hard", "hops": 1,
        "source_docs": ["rag_overview.md"]
    },
    {
        "question": "Qwen2.5系列支持128K上下文，那为什么还需要RAG而不直接把所有文档塞进上下文？",
        "answer": "虽然Qwen2.5支持128K上下文，但直接塞入所有文档存在四个问题：一是存在Lost in the Middle问题（中间位置信息容易被忽略）；二是成本高出2-3个数量级；三是推理速度随上下文增长而变慢；四是存在注意力稀释问题。RAG通过精准检索只取5-10个chunk（约1000-3000 tokens），效率远高于上下文填充。",
        "type": "bridge", "difficulty": "hard", "hops": 1,
        "source_docs": ["rag_overview.md"]
    },
    {
        "question": "如果Qwen2.5-3B使用LoRA微调（r=16，alpha=32），训练超参数应该怎么设置？",
        "answer": "LoRA使用r=16、alpha=32（alpha/r=2的缩放比）时，建议学习率设为1e-5到5e-5，warmup设为总步数的3%-5%，梯度累积步数8-16（有效batch size 32-64），训练2-5个epoch。仅注意力层添加LoRA约增加0.3%-0.5%的额外参数（3B模型约1000万-1500万可训练参数）。",
        "type": "bridge", "difficulty": "hard", "hops": 1,
        "source_docs": ["qlora_intro.md"]
    },

    # === Comparison-Hard (precise comparisons) ===
    {
        "question": "bge-small-zh-v1.5、bge-base-zh-v1.5和bge-large-zh-v1.5在参数量、嵌入维度和显存占用上各有什么差异？",
        "answer": "bge-small-zh-v1.5约3300万参数、512维、约200MB显存，速度最快；bge-base-zh-v1.5约1.1亿参数、768维，均衡型；bge-large-zh-v1.5约3.26亿参数、1024维，精度最高。三者都推荐使用查询前缀\"为这个句子生成表示以检索中文文档：\"。",
        "type": "comparison", "difficulty": "hard", "hops": 1,
        "source_docs": ["rag_overview.md"]
    },
    {
        "question": "AWQ和GPTQ两种4位量化方法的核心区别是什么？各自基于什么原理？",
        "answer": "AWQ（Activation-Aware Weight Quantization）是激活感知的量化方法，通过识别和保护重要权重来减少量化误差，能在4位精度下接近原始精度。GPTQ是基于Hessian矩阵的逐层量化方法，通过二阶信息优化每层的量化参数。AWQ侧重保护重要权重，GPTQ侧重全局最优的层内量化。",
        "type": "comparison", "difficulty": "hard", "hops": 1,
        "source_docs": ["small_lm_trends.md"]
    },
    {
        "question": "在评估RAG系统和ReAct Agent时，分别使用哪些核心指标？两者有什么不同？",
        "answer": "RAG系统的核心评估指标包括ROUGE-L（答案与参考答案的文本重叠度）、Faithfulness（答案与检索上下文的n-gram重叠度）、Answer Relevancy（答案与问题的关键词相关度）和Source Recall（是否正确检索到来源文档）。ReAct Agent的核心评估指标包括TCR（Task Completion Rate，任务完成率）、TSA（Tool Selection Accuracy，工具选择准确率）、参数正确率和步效率。RAG侧重检索质量和答案忠实度，Agent侧重工具使用正确性和任务完成效率。",
        "type": "comparison", "difficulty": "hard", "hops": 2,
        "source_docs": ["rag_overview.md", "react_agent.md"]
    },
    {
        "question": "INT8量化和INT4量化在精度损失上有什么具体差异？对数学推理的影响分别是多少？",
        "answer": "INT8量化将模型大小减半，精度损失不到1%，可通过bitsandbytes的LLM.int8()实现。INT4量化将模型压缩至约1/4大小，但数学推理能力会下降5-10个百分点。INT8适合精度敏感场景，INT4适合显存极度受限场景。",
        "type": "comparison", "difficulty": "hard", "hops": 1,
        "source_docs": ["small_lm_trends.md"]
    },
    {
        "question": "MMR检索中lambda参数的取值范围是多少？不同取值对检索结果有什么影响？",
        "answer": "MMR（最大边际相关性）的lambda参数在实践中通常取0.5到0.7。lambda=1时退化为纯Top-K检索（只考虑相关性），lambda=0时只考虑多样性（最大化结果间差异）。0.5-0.7的取值在相关性和多样性之间取得平衡，避免检索结果过于重复。",
        "type": "comparison", "difficulty": "hard", "hops": 1,
        "source_docs": ["rag_overview.md"]
    },
    {
        "question": "A100 80GB和H100 80GB的市场价格分别是多少？Qwen2.5-3B在RTX 4090上的推理速度是多少？",
        "answer": "A100 80GB约15万元人民币，H100 80GB约25万元人民币。Qwen2.5-3B在RTX 4090上的推理速度约100-150 tokens/s，而同硬件上70B模型只有约10-20 tokens/s。",
        "type": "comparison", "difficulty": "hard", "hops": 1,
        "source_docs": ["small_lm_trends.md"]
    }
]

# Load existing benchmark
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Before: {len(data)} questions")

data.extend(NEW_QUESTIONS)

with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Summary
types = {}
diffs = {}
for q in data:
    t = q.get("type", "?")
    d = q.get("difficulty", "?")
    types[t] = types.get(t, 0) + 1
    diffs[d] = diffs.get(d, 0) + 1

print(f"After: {len(data)} questions")
print(f"By type: {types}")
print(f"By difficulty: {diffs}")
