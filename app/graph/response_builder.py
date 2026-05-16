"""
Graph Nodes: Response Builder + Refuse nodes
"""

from __future__ import annotations
from app.models.schemas import AgentState


def _no_useful_chunks(state: AgentState) -> bool:
    """
    Return True only if zero chunks were retrieved.
    Chunks are stored as dicts (from model_dump()) or as a non-empty list.
    Trusts the LLM to say 'I don't know' when chunks are irrelevant.
    """
    chunks = state.get("retrieved_chunks", [])
    return len(chunks) == 0


def check_retrieval_result(state: AgentState) -> str:
    """
    Conditional edge after RAG / Summarizer / Artifact nodes.
    Routes to retrieval_refuse only if zero chunks came back.
    """
    if _no_useful_chunks(state):
        return "retrieval_refuse"
    return "response_builder"


def retrieval_refuse_node(state: AgentState) -> AgentState:
    """
    Fires when the agent searched but retrieved zero chunks.
    """
    query = state.get("user_query", "")
    reason = (
        f"I searched the available census documents for **\"{query}\"** "
        f"but could not find any relevant information.\n\n"
        f"This could mean:\n"
        f"- The specific data point is not present in the loaded documents\n"
        f"- Try rephrasing your question with different keywords\n"
        f"- The document covering this topic may not have been ingested yet"
    )
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": reason})
    return {
        **state,
        "final_response": reason,
        "refused": True,
        "refusal_reason": "retrieval_empty",
        "conversation_history": history,
    }


def refuse_node(state: AgentState) -> AgentState:
    """
    Fires when Supervisor classifies question as out of scope.
    """
    reason = (
        "This question cannot be answered from the available census documents. "
        "I only have access to government census data. Please ask something "
        "related to the census documents provided."
    )
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["user_query"]})
    history.append({"role": "assistant", "content": reason})
    return {
        **state,
        "final_response": reason,
        "refused": True,
        "refusal_reason": "out_of_scope",
        "conversation_history": history,
    }


def response_builder_node(state: AgentState) -> AgentState:
    """
    Final assembly — dedupes citations, appends artifact path, saves history.
    """
    response = state.get("final_response", "")

    artifact_path = state.get("artifact_path", "")
    if artifact_path:
        response += f"\n\n📊 **Artifact saved:** `{artifact_path}`"

    # Deduplicate citations by chunk_id
    seen_ids: set[str] = set()
    deduped_citations = []
    for c in state.get("citations", []):
        cid = c.get("chunk_id", "") if isinstance(c, dict) else ""
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped_citations.append(c)

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["user_query"]})
    history.append({
        "role": "assistant",
        "content": response,
        "citations": deduped_citations,
        "artifact_path": artifact_path,
    })

    return {
        **state,
        "final_response": response,
        "citations": deduped_citations,
        "conversation_history": history,
        "refused": False,
    }