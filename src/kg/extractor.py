"""KG triple extractor — LLM-based with Chinese + English prompt support."""

from src.core.llm_client import LLMClient

# ---------- English prompts (from kg-agent) ----------

EXTRACTION_SYSTEM_EN = """You are a knowledge graph construction assistant. Your task is to extract factual relationships from text as (subject, relation, object) triples.

Rules:
1. Each triple must represent a clear, factual relationship stated in the text
2. Subject and object must be specific named entities or concrete concepts
3. Relations should be concise verb phrases (e.g., "is capital of", "was born in", "founded")
4. Normalize entity names: use full names, consistent capitalization
5. Extract ALL meaningful relationships, including implicit ones
6. Do NOT hallucinate — only extract facts supported by the text

Output format: JSON array of objects with "subject", "relation", "object" keys.
If no triples can be extracted, output an empty array: []"""

EXTRACTION_PROMPT_EN = """Extract all factual (subject, relation, object) triples from the following text.

Text:
{text}

Output only the JSON array, no other text."""

# ---------- Chinese prompts (new) ----------

EXTRACTION_SYSTEM_ZH = """你是一个知识图谱构建助手。你的任务是从文本中提取事实关系，输出(subject, relation, object)三元组。

规则：
1. 每个三元组必须表示文本中明确的事实关系
2. subject和object必须是具体的命名实体或概念
3. relation应简洁（如"是...的首都"、"出生于"、"发明了"）
4. 实体名称使用完整名称，保持一致的大小写
5. 提取所有有意义的关系，包括隐含关系
6. 不要虚构——只提取文本支持的事实

输出格式：JSON数组，每个对象包含"subject"、"relation"、"object"三个键。
如果没有可提取的三元组，输出空数组：[]"""

EXTRACTION_PROMPT_ZH = """从以下文本中提取所有事实(subject, relation, object)三元组。

文本：
{text}

只输出JSON数组，不要输出其他内容。"""

# ---------- Entity extraction prompts ----------

ENTITY_EXTRACTION_SYSTEM = """You are an entity recognition assistant. Extract the key entities (people, places, organizations, concepts, technical terms) mentioned in the query. Output as a JSON array of strings."""

ENTITY_EXTRACTION_PROMPT = """Extract the key entities from this query. Output only a JSON array of strings.

Query: {query}"""


class TripleExtractor:
    """LLM-based triple extractor with auto language detection."""

    def __init__(self, llm: LLMClient = None, lang: str = "auto"):
        """
        Args:
            llm: LLMClient instance.
            lang: "en", "zh", or "auto" (auto-detect per text).
        """
        self.llm = llm or LLMClient()
        self.lang = lang

    def _detect_lang(self, text):
        """Simple heuristic: if >30% characters are CJK, treat as Chinese."""
        if self.lang != "auto":
            return self.lang
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return "zh" if cjk / max(len(text), 1) > 0.3 else "en"

    def _get_prompts(self, lang):
        if lang == "zh":
            return EXTRACTION_SYSTEM_ZH, EXTRACTION_PROMPT_ZH
        return EXTRACTION_SYSTEM_EN, EXTRACTION_PROMPT_EN

    def extract(self, text, max_triples=20):
        if not text.strip():
            return []

        if len(text) > 8000:
            return self._extract_chunked(text, max_triples)

        lang = self._detect_lang(text)
        system, prompt_tpl = self._get_prompts(lang)

        response = self.llm.generate_text(
            prompt_tpl.format(text=text),
            system=system,
            temperature=0.0,
        )

        triples = self.llm.extract_json(response)
        if isinstance(triples, dict):
            triples = [triples]
        if not isinstance(triples, list):
            return []

        validated = []
        for t in triples[:max_triples]:
            if not isinstance(t, dict):
                continue
            s = t.get("subject", "").strip()
            r = t.get("relation", "").strip()
            o = t.get("object", "").strip()
            if s and r and o and s.lower() != o.lower():
                validated.append({"subject": s, "relation": r, "object": o})

        return validated

    def _extract_chunked(self, text, max_triples):
        chunks = self._split_text(text, chunk_size=4000, overlap=200)
        all_triples = []
        for chunk in chunks:
            triples = self.extract(chunk, max_triples=max_triples)
            all_triples.extend(triples)
        return self._deduplicate(all_triples)[:max_triples]

    def extract_entities_from_query(self, query):
        response = self.llm.generate_text(
            ENTITY_EXTRACTION_PROMPT.format(query=query),
            system=ENTITY_EXTRACTION_SYSTEM,
            temperature=0.0,
        )
        entities = self.llm.extract_json(response)
        if isinstance(entities, list):
            return [e for e in entities if isinstance(e, str) and e.strip()]
        return []

    @staticmethod
    def _split_text(text, chunk_size=4000, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                boundary = text.rfind(". ", start, end)
                if boundary > start:
                    end = boundary + 1
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    @staticmethod
    def _deduplicate(triples):
        seen = set()
        unique = []
        for t in triples:
            key = (t["subject"].lower(), t["relation"].lower(), t["object"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique
