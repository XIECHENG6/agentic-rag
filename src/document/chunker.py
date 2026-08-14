"""Text chunking strategies for RAG — fixed-size and recursive separator-based."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: int = 0


def fixed_size_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict = None,
) -> List[Chunk]:
    """Split text into fixed-size character-level chunks with overlap."""
    if metadata is None:
        metadata = {}
    # Guard: overlap >= size causes infinite loop
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size - 1, 0)
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id, "start": start, "end": end},
                chunk_id=chunk_id,
            ))
            chunk_id += 1
        start += chunk_size - chunk_overlap
    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: List[str] = None,
    metadata: dict = None,
) -> List[Chunk]:
    """Recursively split text using a hierarchy of separators.

    Tries paragraph breaks first, then sentences, then characters.
    Preserves semantic boundaries better than fixed-size chunking.
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ". ", " ", ""]
    if metadata is None:
        metadata = {}
    # Guard: overlap >= size causes infinite loop
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size - 1, 0)

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps:
            return [text]
        sep = seps[0]
        if sep == "":
            # Base case: character-level split with overlap
            result = []
            step = max(chunk_size - chunk_overlap, 1)
            for i in range(0, len(text), step):
                piece = text[i : i + chunk_size]
                if piece.strip():
                    result.append(piece)
            return result

        parts = text.split(sep) if sep else [text]
        merged = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(part) > chunk_size:
                    merged.extend(_split(part, seps[1:]))
                else:
                    current = part
                    continue
                current = ""
        if current:
            merged.append(current)
        return merged

    pieces = _split(text, separators)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk_text = piece.strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_id": i},
                chunk_id=i,
            ))
    return chunks


def chunk_documents(
    documents,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    """Chunk a list of Document objects.

    Args:
        documents: List of Document objects (from loader).
        strategy: "fixed" or "recursive".
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns globally re-indexed list of Chunk objects.
    """
    chunker = recursive_chunk if strategy == "recursive" else fixed_size_chunk
    all_chunks = []
    for doc in documents:
        doc_chunks = chunker(
            doc.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=doc.metadata,
        )
        all_chunks.extend(doc_chunks)
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = i
        chunk.metadata["chunk_id"] = i
    return all_chunks
