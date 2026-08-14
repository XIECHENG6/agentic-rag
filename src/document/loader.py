"""Document loaders for PDF, Markdown, and plain text files."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """A loaded document with content and metadata."""
    content: str
    metadata: dict

    def __repr__(self):
        src = self.metadata.get("source", "unknown")
        return f"Document(source={src}, length={len(self.content)})"


def load_text(file_path: str) -> Document:
    """Load a plain text file."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return Document(content=content, metadata={"source": path.name, "type": "txt"})


def load_markdown(file_path: str) -> Document:
    """Load a Markdown file."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return Document(content=content, metadata={"source": path.name, "type": "markdown"})


def load_pdf(file_path: str) -> Document:
    """Load a PDF file using PyMuPDF."""
    import fitz

    path = Path(file_path)
    pages = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
    content = "\n\n".join(pages)
    return Document(
        content=content,
        metadata={"source": path.name, "type": "pdf", "pages": len(pages)},
    )


LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_document(file_path: str) -> Document:
    """Load a document based on file extension."""
    ext = Path(file_path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")
    return loader(file_path)


def load_directory(dir_path: str, extensions: List[str] = None) -> List[Document]:
    """Load all supported documents from a directory."""
    if extensions is None:
        extensions = list(LOADERS.keys())
    docs = []
    for root, _, files in os.walk(dir_path):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in extensions:
                fpath = os.path.join(root, fname)
                try:
                    docs.append(load_document(fpath))
                except Exception as e:
                    print(f"Warning: failed to load {fpath}: {e}")
    return docs
