"""
Pydantic models and TypedDicts for the Census Chatbot system.
These define the contracts between every component.
"""

from __future__ import annotations
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A factual claim traced back to a source document."""
    source_file: str = Field(..., description="Filename of the source PDF/markdown")
    page_number: int = Field(..., description="Page number in the source document")
    snippet: str = Field(..., description="Verbatim short excerpt supporting the claim")
    chunk_id: str = Field(..., description="Internal ID of the retrieved chunk")


class Chunk(BaseModel):
    """A retrieved document chunk from the vector store."""
    chunk_id: str
    source_file: str
    page_number: int
    text: str
    score: float = Field(..., description="Similarity score from vector search")


class Message(BaseModel):
    """A single turn in the conversation."""
    role: Literal["user", "assistant", "tool"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    artifact_path: Optional[str] = None   # path to chart/table if produced


# ---------------------------------------------------------------------------
# Tool input / output schemas
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    query: str = Field(..., description="Natural language query to search census documents")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")


class SearchOutput(BaseModel):
    chunks: list[Chunk]
    query_used: str


class CodeExecutionInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout_seconds: int = Field(default=30)


class CodeExecutionOutput(BaseModel):
    stdout: str
    stderr: str
    success: bool
    artifact_paths: list[str] = Field(
        default_factory=list,
        description="Paths to any files written during execution"
    )


class WriteFileInput(BaseModel):
    filename: str
    content: str
    subdir: str = Field(default="artifacts")


class WriteFileOutput(BaseModel):
    path: str
    success: bool


# ---------------------------------------------------------------------------
# Agent response
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """Final structured response returned to the UI."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    artifact_path: Optional[str] = None
    intent: str = "query"
    refused: bool = False
    refusal_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# LangGraph shared state — the spine of the entire graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    user_query: str
    conversation_history: list[dict]          # list of Message dicts

    # ── Routing ────────────────────────────────────────────────────────────
    intent: str                               # "query" | "summarize" | "artifact" | "refuse"

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: list[dict]              # list of Chunk dicts
    citations: list[dict]                     # list of Citation dicts

    # ── Artifact (code execution) ───────────────────────────────────────────
    generated_code: str
    code_output: str
    code_error: str
    artifact_path: str
    retry_count: int

    # ── Output ─────────────────────────────────────────────────────────────
    final_response: str
    refused: bool
    refusal_reason: str
