"""Text chunking strategies for RAG — fixed-size and recursive separator-based."""

from dataclasses import dataclass, field
from typing import List


def _measure(text: str, length_fn=None) -> int:
    return length_fn(text) if length_fn is not None else len(text)


def _max_end(text: str, start: int, limit: int, length_fn=None) -> int:
    """Find the furthest character boundary within a length budget."""
    if start >= len(text):
        return start
    if _measure(text[start:], length_fn) <= limit:
        return len(text)
    lo, hi = start + 1, len(text)
    best = start + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _measure(text[start:mid], length_fn) <= limit:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _overlap_start(text: str, start: int, end: int, overlap: int, length_fn=None) -> int:
    """Return the earliest boundary whose tail fits the overlap budget."""
    if overlap <= 0 or end <= start:
        return end
    if _measure(text[start:end], length_fn) <= overlap:
        return start
    lo, hi = start, end - 1
    best = end
    while lo <= hi:
        mid = (lo + hi) // 2
        if _measure(text[mid:end], length_fn) <= overlap:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def _tail_within(text: str, limit: int, length_fn=None) -> str:
    if not text or limit <= 0:
        return ""
    start = _overlap_start(text, 0, len(text), limit, length_fn)
    return text[start:]


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
    length_fn=None,
) -> List[Chunk]:
    """Split text into fixed-size chunks measured by characters or tokens."""
    if metadata is None:
        metadata = {}
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = _max_end(text, start, chunk_size, length_fn)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id, "start": start, "end": end},
                chunk_id=chunk_id,
            ))
            chunk_id += 1
        next_start = _overlap_start(text, start, end, chunk_overlap, length_fn)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: List[str] = None,
    metadata: dict = None,
    length_fn=None,
) -> List[Chunk]:
    """Recursively split text using a hierarchy of separators.

    Tries paragraph breaks first, then sentences, then characters.
    Preserves semantic boundaries better than fixed-size chunking.
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ". ", " ", ""]
    if metadata is None:
        metadata = {}
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps:
            return [text]
        sep = seps[0]
        if sep == "":
            # Base case: character-level split with overlap.
            result = []
            step = max(chunk_size - chunk_overlap, 1)
            i = 0
            while i < len(text):
                end = _max_end(text, i, chunk_size, length_fn)
                piece = text[i:end]
                if piece.strip():
                    result.append(piece)
                next_i = _overlap_start(text, i, end, chunk_overlap, length_fn)
                if next_i <= i:
                    next_i = end
                i = next_i
            return result

        parts = text.split(sep) if sep else [text]
        merged = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if _measure(candidate, length_fn) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if _measure(part, length_fn) > chunk_size:
                    merged.extend(_split(part, seps[1:]))
                else:
                    current = part
                    continue
                current = ""
        if current:
            merged.append(current)
        return merged

    pieces = _split(text, separators)

    # Preserve semantic boundaries first, then add bounded overlap where the
    # next semantic piece has room for it.
    if chunk_overlap > 0:
        overlapped = []
        for piece in pieces:
            if overlapped:
                available = max(chunk_size - _measure(piece, length_fn), 0)
                overlap_size = min(
                    chunk_overlap,
                    available,
                    _measure(overlapped[-1], length_fn),
                )
                if overlap_size:
                    candidate = _tail_within(overlapped[-1], overlap_size, length_fn) + piece
                    if _measure(candidate, length_fn) <= chunk_size:
                        piece = candidate
            overlapped.append(piece)
        pieces = overlapped

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
    length_fn=None,
) -> List[Chunk]:
    """Chunk a list of Document objects.

    Args:
        documents: List of Document objects (from loader).
        strategy: "fixed" or "recursive".
        chunk_size: Maximum value measured by ``length_fn`` (characters by default).
        chunk_overlap: Overlap in the same units as ``chunk_size``.
        length_fn: Optional text length function, such as a model tokenizer.

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
            length_fn=length_fn,
        )
        all_chunks.extend(doc_chunks)
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = i
        chunk.metadata["chunk_id"] = i
    return all_chunks
