"""
Graph Node: RAG Agent
Uses Groq native client for tool calling.
tool_choice="required" on search, "none" on answer to prevent XML format bug.
"""

from __future__ import annotations
import json
from app.graph.llm import get_groq_client, TOOL_MODEL
from app.models.schemas import AgentState, Citation
from app.tools.search import search_documents

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the Indian Census 2011 document corpus for relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a census data analyst for Indian Census 2011 ONLY.

Rules:
1. Always call search_documents first — never answer from memory.
2. Every fact must come from a retrieved chunk.
3. Cite sources as [filename, page N] after each claim.
4. Be precise with numbers.
5. CRITICAL: If the question is about any country or region outside India,
   or if the retrieved chunks do not contain relevant information, respond with:
   "This information is not available in the Indian Census 2011 documents."
   Never use census data to answer questions about other countries.
"""


def rag_agent_node(state: AgentState) -> AgentState:
    try:
        client = get_groq_client()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": state["user_query"]},
        ]

        citations: list[dict] = []
        retrieved_chunks: list[dict] = []

        # ── Step 1: Force tool call ───────────────────────────────────────────
        response = client.chat.completions.create(
            model=TOOL_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            temperature=0,
            max_tokens=512,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return {
                **state,
                "final_response": "I was unable to search the documents. Please try again.",
                "retrieved_chunks": [],
                "citations": [],
            }

        # ── Step 2: Execute tool calls ────────────────────────────────────────
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"query": state["user_query"], "top_k": 5}

            result = search_documents.invoke(args)
            result_str = json.dumps(result)

            chunks = result.get("chunks", [])
            retrieved_chunks.extend(chunks)
            for chunk in chunks[:3]:
                citations.append(
                    Citation(
                        source_file=chunk["source_file"],
                        page_number=chunk["page_number"],
                        snippet=chunk["text"][:200],
                        chunk_id=chunk["chunk_id"],
                    ).model_dump()
                )

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str,
            })

        # ── Step 3: Final answer with tool_choice="none" ──────────────────────
        answer_response = client.chat.completions.create(
            model=TOOL_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="none",
            temperature=0,
            max_tokens=2048,
        )

        final_answer = answer_response.choices[0].message.content or ""

        if not final_answer.strip():
            final_answer = "I found relevant documents but could not generate a response. Please try again."

        return {
            **state,
            "final_response":   final_answer,
            "retrieved_chunks": retrieved_chunks,
            "citations":        citations,
        }

    except Exception as e:
        return {
            **state,
            "final_response": (
                f"I encountered an error while searching the documents: {str(e)}\n\n"
                "Please try rephrasing your question."
            ),
            "retrieved_chunks": ["error"],
            "citations": [],
        }