"""
Graph Builder — wires all nodes and edges into the compiled LangGraph.

Graph topology:

                        ┌─────────────┐
                        │  supervisor │
                        └──────┬──────┘
          ┌──────────────┬─────┴──────────┬──────────────┐
       "query"      "summarize"        "artifact"      "refuse"
          │               │                │               │
          ▼               ▼                ▼               ▼
      rag_agent       summarizer     artifact_agent    refuse_node
          │               │                │               │
          └───────────────┴────────────────┘               │
                          │                                 │
               check_retrieval_result()                     │
                /                  \                        │
    "retrieval_refuse"       "response_builder"             │
          │                         │                       │
          ▼                         ▼                       │
  retrieval_refuse_node     response_builder_node ◄─────────┘
          │                         │
          └───────────┬─────────────┘
                      ▼
                     END
"""

from __future__ import annotations
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.memory import MemorySaver

from app.models.schemas import AgentState
from app.graph.supervisor import supervisor_node, route_by_intent
from app.graph.rag_agent import rag_agent_node
from app.graph.summarizer import summarizer_node
from app.graph.artifact_agent import artifact_agent_node
from app.graph.response_builder import (
    response_builder_node,
    refuse_node,
    retrieval_refuse_node,
    check_retrieval_result,
)


def build_graph():
    """Build and compile the LangGraph agent graph with in-memory checkpointing."""

    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("query",              rag_agent_node)
    graph.add_node("summarize",          summarizer_node)
    graph.add_node("artifact",           artifact_agent_node)
    graph.add_node("refuse",             refuse_node)
    graph.add_node("retrieval_refuse",   retrieval_refuse_node)
    graph.add_node("response_builder",   response_builder_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Supervisor routes by intent ─────────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "query":     "query",
            "summarize": "summarize",
            "artifact":  "artifact",
            "refuse":    "refuse",
        },
    )

    # ── After search-based agents: check if retrieval found anything ─────────
    # If nothing found → retrieval_refuse, else → response_builder
    graph.add_conditional_edges(
        "query",
        check_retrieval_result,
        {
            "retrieval_refuse": "retrieval_refuse",
            "response_builder": "response_builder",
        },
    )
    graph.add_conditional_edges(
        "summarize",
        check_retrieval_result,
        {
            "retrieval_refuse": "retrieval_refuse",
            "response_builder": "response_builder",
        },
    )
    graph.add_conditional_edges(
        "artifact",
        check_retrieval_result,
        {
            "retrieval_refuse": "retrieval_refuse",
            "response_builder": "response_builder",
        },
    )

    # ── Supervisor-level refuse goes directly to END via response_builder ───
    graph.add_edge("refuse", "response_builder")

    # ── Both refuse paths terminate at END ──────────────────────────────────
    graph.add_edge("retrieval_refuse", END)
    graph.add_edge("response_builder", END)

    # ── Compile with memory checkpointer ────────────────────────────────────
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    return compiled


# Module-level compiled graph (imported by API and UI)
graph = build_graph()