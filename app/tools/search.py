"""
Tool: search_documents
Semantic search over the ingested census document vector store.
Returns ranked chunks with source + page citations.
"""

from __future__ import annotations
import os
from langchain_core.tools import tool
from app.models.schemas import SearchInput, SearchOutput, Chunk

# Lazy import so the vector store is only loaded when actually used
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from app.retrieval.vectorstore import get_vectorstore
        _vectorstore = get_vectorstore()
    return _vectorstore


@tool("search_documents", args_schema=SearchInput)
def search_documents(query: str, top_k: int = 5) -> dict:
    """
    Search the census document corpus for chunks relevant to the query.
    Returns text chunks with source file, page number, and similarity score.
    Always use this before making any factual claim about the census data.
    """
    # Coerce top_k — LLM sometimes passes a string instead of int
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
 
    # Clamp to a sensible range
    top_k = max(1, min(top_k, 20))
    vs = _get_vectorstore()
    results = vs.similarity_search_with_score(query, k=top_k)

    chunks = []
    for i, (doc, score) in enumerate(results):
        chunks.append(
            Chunk(
                chunk_id=doc.metadata.get("chunk_id", f"chunk_{i}"),
                source_file=doc.metadata.get("source_file", "unknown"),
                page_number=int(doc.metadata.get("page_number", 0)),
                text=doc.page_content,
                score=float(score),
            ).model_dump()
        )

    return SearchOutput(chunks=chunks, query_used=query).model_dump()
