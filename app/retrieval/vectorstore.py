"""
Retrieval layer — ingests census PDFs/markdowns into ChromaDB.
Run once: python -m app.retrieval.ingest
"""

from __future__ import annotations
import os
import hashlib
from pathlib import Path
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

DOCS_DIR = Path(os.environ.get("DOCS_DIR", "data/documents"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "workspace/chroma"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_embeddings = None
_vectorstore = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    """Return (or create) the persistent ChromaDB vector store."""
    global _vectorstore
    if _vectorstore is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_get_embeddings(),
            collection_name="census_docs",
        )
    return _vectorstore


def _chunk_id(source_file: str, page: int, idx: int) -> str:
    raw = f"{source_file}::{page}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def ingest_documents(docs_dir: Path = DOCS_DIR) -> int:
    """
    Ingest all .md and .txt files under docs_dir into ChromaDB.
    Returns the number of chunks added.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    vs = get_vectorstore()
    total_added = 0

    for path in sorted(docs_dir.rglob("*")):
        if path.suffix not in (".md", ".txt", ".markdown"):
            continue

        print(f"Ingesting: {path.name}")
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(raw_text)

        docs = []
        for i, chunk_text in enumerate(chunks):
            # Attempt to extract page number from markdown headers like "## Page 12"
            page_num = _extract_page_number(chunk_text, i)
            cid = _chunk_id(path.name, page_num, i)

            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": path.name,
                        "page_number": page_num,
                        "chunk_id": cid,
                        "chunk_index": i,
                    },
                )
            )

        if docs:
            vs.add_documents(docs)
            total_added += len(docs)
            print(f"  → {len(docs)} chunks added")

    print(f"\nIngestion complete. Total chunks: {total_added}")
    return total_added


def _extract_page_number(text: str, fallback: int) -> int:
    """Try to parse a page number from chunk text; fall back to chunk index."""
    import re
    match = re.search(r"page[:\s#]*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback


if __name__ == "__main__":
    ingest_documents()
