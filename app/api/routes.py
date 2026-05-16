"""
FastAPI application.
Endpoints:
  POST /chat        — main chat endpoint
  GET  /health      — liveness check
"""

from __future__ import annotations
import uuid
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.graph.builder import graph

app = FastAPI(title="Census Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[dict]
    artifact_path: str
    intent: str
    refused: bool
    refusal_reason: str
    error: str = ""


def _initial_state(query: str) -> dict:
    return {
        "user_query": query,
        "conversation_history": [],
        "intent": "",
        "retrieved_chunks": [],
        "citations": [],
        "generated_code": "",
        "code_output": "",
        "code_error": "",
        "artifact_path": "",
        "retry_count": 0,
        "final_response": "",
        "refused": False,
        "refusal_reason": "",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = graph.invoke(_initial_state(req.query), config=config)

        return ChatResponse(
            session_id=session_id,
            answer=result.get("final_response", ""),
            citations=result.get("citations", []),
            artifact_path=result.get("artifact_path", ""),
            intent=result.get("intent", "query"),
            refused=result.get("refused", False),
            refusal_reason=result.get("refusal_reason", ""),
        )

    except Exception as e:
        # Log full traceback to terminal but return clean JSON to UI
        traceback.print_exc()
        return JSONResponse(
            status_code=200,   # return 200 so UI renders the error message
            content={
                "session_id": session_id,
                "answer": (
                    f"⚠️ The agent encountered an unexpected error:\n\n"
                    f"`{str(e)}`\n\n"
                    "Please try again or rephrase your question."
                ),
                "citations": [],
                "artifact_path": "",
                "intent": "query",
                "refused": False,
                "refusal_reason": "",
                "error": str(e),
            }
        )


@app.get("/health")
async def health():
    return {"status": "ok"}