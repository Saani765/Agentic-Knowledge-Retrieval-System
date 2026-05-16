"""
Graph Node: Artifact Agent
Uses Groq native client for tool calling.
Tools: search_documents, execute_python, write_file
Retry loop up to MAX_RETRIES on code execution errors.
"""

from __future__ import annotations
import os
import json
from groq import Groq
from app.graph.llm import get_groq_client, TOOL_MODEL
from app.models.schemas import AgentState, Citation
from app.tools.search import search_documents
from app.tools.code_executor import execute_python
from app.tools.file_tools import write_file

MAX_RETRIES = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search census documents for relevant data chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of chunks"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code to produce a chart or table. "
                "plt.show() auto-saves figures. Returns stdout, stderr, success, artifact_paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content (e.g. CSV) to workspace/artifacts/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content":  {"type": "string"},
                    "subdir":   {"type": "string", "description": "artifacts or notes"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a data visualization agent for census data.

Follow this exact sequence:
1. Call search_documents to retrieve the relevant census data.
2. Write Python code and call execute_python to produce the chart or table.
3. If execute_python returns success=false, fix the code and retry (max 3 times).
4. Report what was produced.

Python code rules:
- Use only: matplotlib, pandas, numpy.
- Call plt.show() at the end — it auto-saves figures to workspace/artifacts/.
- Always add title, axis labels, and:
  fig.text(0.99, 0.01, 'Source: Census of India 2011', ha='right', fontsize=8, color='gray')
- Use ONLY data from search results. Never invent numbers.
"""


def _run_tool(name: str, args: dict) -> str:
    if name == "search_documents":
        return json.dumps(search_documents.invoke(args))
    if name == "execute_python":
        return json.dumps(execute_python.invoke(args))
    if name == "write_file":
        return json.dumps(write_file.invoke(args))
    return json.dumps({"error": f"Unknown tool: {name}"})


def artifact_agent_node(state: AgentState) -> AgentState:
    try:
        client = get_groq_client()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": state["user_query"]},
        ]

        citations: list[dict] = []
        retrieved_chunks: list[dict] = []
        artifact_path = ""
        retry_count = 0
        max_iterations = 12
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
                    args = {}

                result_str = _run_tool(tc.function.name, args)
                result = json.loads(result_str)

                # Collect search results
                if tc.function.name == "search_documents":
                    chunks = result.get("chunks", [])
                    retrieved_chunks.extend(chunks)
                    seen = {c["chunk_id"] for c in citations}
                    for chunk in chunks[:3]:
                        if chunk["chunk_id"] not in seen:
                            citations.append(
                                Citation(
                                    source_file=chunk["source_file"],
                                    page_number=chunk["page_number"],
                                    snippet=chunk["text"][:200],
                                    chunk_id=chunk["chunk_id"],
                                ).model_dump()
                            )

                # Track artifacts and retries
                elif tc.function.name == "execute_python":
                    if result.get("success"):
                        paths = result.get("artifact_paths", [])
                        if paths:
                            artifact_path = paths[-1]
                    else:
                        retry_count += 1
                        if retry_count >= MAX_RETRIES:
                            messages.append({
                                "role":         "tool",
                                "tool_call_id": tc.id,
                                "content":      result_str,
                            })
                            messages.append({
                                "role":    "user",
                                "content": f"Max retries ({MAX_RETRIES}) reached. Describe what you tried and what went wrong.",
                            })
                            final_response = client.chat.completions.create(
                                model=TOOL_MODEL,
                                messages=messages,
                                temperature=0,
                                max_tokens=1024,
                            )
                            return {
                                **state,
                                "final_response":   final_response.choices[0].message.content,
                                "retrieved_chunks": retrieved_chunks,
                                "citations":        citations,
                                "retry_count":      retry_count,
                                "artifact_path":    "",
                            }

                elif tc.function.name == "write_file":
                    if result.get("success"):
                        artifact_path = result.get("path", artifact_path)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

        else:
            final_answer = "I was unable to complete the artifact generation. Please try again."

        if not final_answer.strip():
            final_answer = "Processing complete."

        return {
            **state,
            "final_response":   final_answer,
            "retrieved_chunks": retrieved_chunks,
            "citations":        citations,
            "artifact_path":    artifact_path,
            "retry_count":      retry_count,
        }

    except Exception as e:
        return {
            **state,
            "final_response": (
                f"I encountered an error generating the artifact: {str(e)}\n\n"
                "Please try rephrasing your request."
            ),
            "retrieved_chunks": ["error"],
            "citations":        [],
            "artifact_path":    "",
        }