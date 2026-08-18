"""Agentic RAG Pipeline — single entry point that wires everything together.

Usage::

    from src.pipeline import AgenticRAGPipeline

    pipeline = AgenticRAGPipeline(api_key="sk-...")
    pipeline.ingest_documents(documents)       # or pipeline.ingest_directory("data/docs")
    result = pipeline.ask("What is BERT?")
    print(result["answer"])
"""

import os
import json
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
        allow_direct: bool = False,
        chunk_unit: str = None,
        max_llm_calls: int = None,
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
        self.chunk_unit = chunk_unit or chunk_cfg.get("unit", "tokens")
        if self.chunk_unit not in {"chars", "tokens"}:
            raise ValueError("chunk_unit must be 'chars' or 'tokens'")
        self.all_chunks: List[Chunk] = []

        # Agent components
        self.planner = Planner(
            self.llm,
            max_sub_problems=agent_cfg.get("max_sub_problems", 4),
        )
        self.reflector = Reflector(
            self.llm,
            threshold=(reflection_threshold if reflection_threshold is not None
                       else agent_cfg.get("reflection_threshold", 0.6)),
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
            max_llm_calls=max_llm_calls if max_llm_calls is not None else agent_cfg.get("max_llm_calls", 24),
            verbose=verbose,
            allow_direct=allow_direct,
        )

        self.verbose = verbose

    # ---------- Ingestion ----------

    def ingest_documents(self, documents: List[Document]):
        """Chunk, embed, index + extract KG triples from a list of Documents."""
        if not documents:
            return

        # Chunk
        chunks = chunk_documents(
            documents,
            strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_fn=self.embedder.token_count if self.chunk_unit == "tokens" else None,
        )
        if self.verbose:
            print(f"Chunked {len(documents)} documents → {len(chunks)} chunks")
        if not chunks:
            return

        # Embed + index
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress=self.verbose)
        metas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.setdefault("source", "unknown")
            metadata["chunk_id"] = chunk.chunk_id
            metas.append(metadata)
        # Extract all triples before mutating either index. This keeps a failed
        # LLM extraction from leaving a partially ingested vector store.
        pending_triples = []
        for doc in documents:
            text = doc.content
            triples = self.extractor.extract(text, max_triples=20)
            source = doc.metadata.get("source", "unknown")
            triples = [dict(triple, source=source) for triple in triples]
            pending_triples.extend(triples)
            if self.verbose:
                print(f"KG from '{doc.metadata.get('source', '?')}': "
                      f"{len(triples)} extracted")

        self.vector_store.add(embeddings, texts, metas)
        self.all_chunks.extend(chunks)
        added = self.kg.add_triples(pending_triples)

        if self.verbose:
            print(f"Vector store: {self.vector_store.size} vectors")
            print(f"KG triples added: {added}")
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

    def save(self, path: str):
        """Persist the vector index, KG, and configuration as one artifact."""
        os.makedirs(path, exist_ok=True)
        vector_path = os.path.join(path, "vector_store")
        kg_path = os.path.join(path, "knowledge_graph.json")
        self.vector_store.save(vector_path)
        self.kg.save(kg_path)
        manifest = {
            "schema_version": 1,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy,
            "chunk_unit": self.chunk_unit,
            "vector_store": "vector_store",
            "knowledge_graph": "knowledge_graph.json",
        }
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Restore a pipeline artifact created by :meth:`save`."""
        with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        init_kwargs = dict(kwargs)
        init_kwargs.setdefault("embedding_model", manifest["embedding_model"])
        init_kwargs.setdefault("chunk_size", manifest["chunk_size"])
        init_kwargs.setdefault("chunk_overlap", manifest["chunk_overlap"])
        init_kwargs.setdefault("chunk_strategy", manifest["chunk_strategy"])
        init_kwargs.setdefault("chunk_unit", manifest.get("chunk_unit", "chars"))
        pipeline = cls(**init_kwargs)
        pipeline.vector_store.load(os.path.join(path, manifest["vector_store"]))
        if pipeline.vector_store.dimension != pipeline.embedder.dimension:
            raise ValueError(
                "Persisted vector dimension does not match the embedding model: "
                f"{pipeline.vector_store.dimension} != {pipeline.embedder.dimension}"
            )
        pipeline.kg.load(os.path.join(path, manifest["knowledge_graph"]))
        pipeline.all_chunks = [
            Chunk(text=text, metadata=dict(metadata),
                  chunk_id=metadata.get("chunk_id", i))
            for i, (text, metadata) in enumerate(
                zip(pipeline.vector_store.texts, pipeline.vector_store.metadata)
            )
        ]
        return pipeline

    def stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "vector_store_size": self.vector_store.size,
            "kg_entities": self.kg.num_entities,
            "kg_relations": self.kg.num_relations,
            "kg_relation_types": len(self.kg.relation_types),
            "total_chunks": len(self.all_chunks),
        }
