"""
Graph Node: Summarizer Agent
Uses Groq native client for tool calling.
Uses top_k=8 for broader context coverage.
"""

from __future__ import annotations
import os
import json
from groq import Groq
from app.graph.llm import get_groq_client, TOOL_MODEL
from app.models.schemas import AgentState, Citation
from app.tools.search import search_documents

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the census document corpus for chunks relevant to the query. "
                "Use top_k=8 for summaries to get broad coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a census document summarizer.

Rules:
1. Call search_documents with top_k=8 for broad coverage.
2. Structure output under: ## Key Figures, ## Demographics, ## Trends, ## Gaps.
3. Cite every section as [filename, page N].
4. Stay factual — do not interpret beyond the data.
5. Flag missing or inconsistent data explicitly under Gaps.
"""


def summarizer_node(state: AgentState) -> AgentState:
    try:
        client = get_groq_client()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": state["user_query"]},
        ]

        citations: list[dict] = []
        retrieved_chunks: list[dict] = []
        max_iterations = 6
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            response = client.chat.completions.create(
                model=TOOL_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
                max_tokens=4096,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                final_answer = msg.content or ""
                break

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
                    args = {"query": state["user_query"], "top_k": 8}

                # Force top_k >= 8 for summarizer
                args["top_k"] = max(args.get("top_k", 8), 8)

                result = search_documents.invoke(args)
                result_str = json.dumps(result)

                chunks = result.get("chunks", [])
                retrieved_chunks.extend(chunks)
                seen = {c["chunk_id"] for c in citations}
                for chunk in chunks[:5]:
                    if chunk["chunk_id"] not in seen:
                        seen.add(chunk["chunk_id"])
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

        else:
            final_answer = "I was unable to complete the summary. Please try again."

        if not final_answer.strip():
            final_answer = "I found relevant documents but could not generate a summary. Please try again."

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
                f"I encountered an error while summarizing: {str(e)}\n\n"
                "Please try rephrasing your question."
            ),
            "retrieved_chunks": ["error"],
            "citations": [],
        }