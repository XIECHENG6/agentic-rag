"""Agentic RAG Pipeline — single entry point that wires everything together.

Usage::

    from src.pipeline import AgenticRAGPipeline

    pipeline = AgenticRAGPipeline(api_key="sk-...")
    pipeline.ingest_documents(documents)       # or pipeline.ingest_directory("data/docs")
    result = pipeline.ask("What is BERT?")
    print(result["answer"])
"""

import os
from typing import List, Optional

from src.core.llm_client import LLMClient, load_config
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.document.chunker import chunk_documents, Chunk
from src.document.loader import Document, load_directory
from src.kg.graph_store import KnowledgeGraph
from src.kg.extractor import TripleExtractor
from src.kg.retriever import GraphRetriever
from src.kg.hybrid import HybridRetriever
from src.agent.planner import Planner
from src.agent.reflector import Reflector
from src.agent.synthesizer import Synthesizer
from src.agent.tools import ToolExecutor
from src.agent.state_machine import AgenticRAG


class AgenticRAGPipeline:
    """End-to-end Agentic RAG: ingest → build KG + vector index → ask questions.

    This is the single class users interact with.  It owns all sub-components
    and wires them together.
    """

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
        max_reformulations: int = None,
        reflection_threshold: float = None,
        max_steps: int = None,
        verbose: bool = False,
    ):
        config = load_config()
        agent_cfg = config.get("agent", {})
        chunk_cfg = config.get("chunking", {})

        # LLM
        self.llm = LLMClient(api_base=api_base, api_key=api_key, model=model)

        # Embedding + vector store
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.retriever = Retriever(self.embedder, self.vector_store)

        # KG
        self.kg = KnowledgeGraph()
        self.extractor = TripleExtractor(self.llm)
        self.graph_retriever = GraphRetriever(self.kg, self.extractor)

        # Hybrid
        self.hybrid_retriever = HybridRetriever(
            self.graph_retriever, self.vector_store, self.embedder
        )

        # Chunking config — param > settings.yaml > hardcoded default
        self.chunk_size = chunk_size or chunk_cfg.get("chunk_size", 512)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else chunk_cfg.get("chunk_overlap", 64)
        self.chunk_strategy = chunk_strategy or chunk_cfg.get("strategy", "recursive")
        self.all_chunks: List[Chunk] = []

        # Agent components
        self.planner = Planner(
            self.llm,
            max_sub_problems=agent_cfg.get("max_sub_problems", 4),
        )
        self.reflector = Reflector(
            self.llm,
            threshold=reflection_threshold or agent_cfg.get("reflection_threshold", 0.6),
        )
        self.synthesizer = Synthesizer(self.llm)

        # Tool executor
        self.tool_executor = ToolExecutor(
            llm=self.llm,
            retriever=self.retriever,
            graph_retriever=self.graph_retriever,
            hybrid_retriever=self.hybrid_retriever,
            kg=self.kg,
            planner=self.planner,
            reflector=self.reflector,
            synthesizer=self.synthesizer,
        )

        # The agent itself
        self.agent = AgenticRAG(
            llm=self.llm,
            tools=self.tool_executor,
            planner=self.planner,
            reflector=self.reflector,
            synthesizer=self.synthesizer,
            max_steps=max_steps or agent_cfg.get("max_steps", 12),
            max_reformulations=max_reformulations if max_reformulations is not None else agent_cfg.get("max_reformulations", 2),
            verbose=verbose,
        )

        self.verbose = verbose

    # ---------- Ingestion ----------

    def ingest_documents(self, documents: List[Document]):
        """Chunk, embed, index + extract KG triples from a list of Documents."""
        # Chunk
        chunks = chunk_documents(
            documents,
            strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.all_chunks.extend(chunks)

        if self.verbose:
            print(f"Chunked {len(documents)} documents → {len(chunks)} chunks")

        # Embed + index
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress=self.verbose)
        metas = [{"source": c.metadata.get("source", "unknown"), "chunk_id": c.chunk_id} for c in chunks]
        self.vector_store.add(embeddings, texts, metas)

        if self.verbose:
            print(f"Vector store: {self.vector_store.size} vectors")

        # KG extraction
        for doc in documents:
            text = doc.content
            triples = self.extractor.extract(text, max_triples=20)
            added = self.kg.add_triples(triples)
            if self.verbose:
                print(f"KG from '{doc.metadata.get('source', '?')}': "
                      f"{len(triples)} extracted, {added} added")

        if self.verbose:
            stats = self.kg.stats()
            print(f"KG total: {stats['entities']} entities, {stats['relations']} relations")

    def ingest_directory(self, dir_path: str, extensions: List[str] = None):
        """Load and ingest all documents from a directory."""
        documents = load_directory(dir_path, extensions)
        if not documents:
            print(f"No documents found in {dir_path}")
            return
        self.ingest_documents(documents)

    def ingest_texts(self, texts_and_titles: List[tuple]):
        """Ingest raw (title, text) tuples — convenient for notebook use."""
        documents = [
            Document(content=text, metadata={"source": title})
            for title, text in texts_and_titles
        ]
        self.ingest_documents(documents)

    # ---------- Query ----------

    def ask(self, question: str, verbose: bool = None, lang: str = "zh") -> dict:
        """Ask the agent a question.

        Returns:
            {
                "answer": str,
                "trace": list[dict],
                "question_type": str,
                "strategy": str,
                "sub_problems": list[str],
                "reformulations": int,
                "reflection_scores": dict,
            }
        """
        return self.agent.run(question, verbose=verbose, lang=lang)

    # ---------- Baseline queries (for benchmarking) ----------

    def simple_rag(self, question: str, lang: str = "zh") -> dict:
        """Baseline: single-pass vector retrieval + LLM generation."""
        results = self.retriever.retrieve(question, top_k=5, method="topk", lang=lang)
        context = self.retriever.format_context(results)
        answer = self.synthesizer.generate_answer(question, context)
        return {"answer": answer, "context": context}

    def hybrid_rag(self, question: str, lang: str = "zh") -> dict:
        """Baseline: hybrid (vector + KG) retrieval + LLM generation."""
        context = self.hybrid_retriever.get_context(question, top_k=5, lang=lang)
        answer = self.synthesizer.generate_answer(question, context)
        return {"answer": answer, "context": context}

    def no_retrieval(self, question: str) -> dict:
        """Baseline: LLM answers from parametric knowledge."""
        answer = self.tool_executor.execute("direct_answer", {"question": question})
        return {"answer": answer}

    # ---------- Stats ----------

    def stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "vector_store_size": self.vector_store.size,
            "kg_entities": self.kg.num_entities,
            "kg_relations": self.kg.num_relations,
            "kg_relation_types": len(self.kg.relation_types),
            "total_chunks": len(self.all_chunks),
        }
