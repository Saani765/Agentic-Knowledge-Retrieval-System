"""
Chainlit UI — Census Document Q&A Bot
Full observability: every node, tool call, chunk, and citation visible as steps.
Run: chainlit run ui/app.py --port 8001
"""

from __future__ import annotations
import uuid
import httpx
import chainlit as cl

API_URL = "http://localhost:8000"

INTENT_META = {
    "query":     ("🔍", "RAG Agent",      "Semantic search → cited answer"),
    "summarize": ("📝", "Summarizer",     "Broad retrieval → structured summary"),
    "artifact":  ("🎨", "Artifact Agent", "Retrieve data → write code → execute"),
    "refuse":    ("🚫", "Scope Guard",    "Question outside Indian Census 2011 scope"),
}

INTENT_COLOR = {
    "query":     "🟦",
    "summarize": "🟩",
    "artifact":  "🟨",
    "refuse":    "🟥",
}


@cl.on_chat_start
async def on_start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    await cl.Message(
        content=(
            "## 📊 Census Document Q&A — Indian Census 2011\n\n"
            "Every processing step is visible as it runs. "
            "Expand any step to see tools called, chunks retrieved, and citations used.\n\n"
            "| Query Type | Example |\n"
            "|---|---|\n"
            "| 🔍 **Factual** | *Literacy rate in Odisha 2011?* |\n"
            "| 📝 **Summary** | *Summarize key findings from Odisha data* |\n"
            "| 🎨 **Chart**   | *Bar chart of male vs female population in MP* |\n"
            "| 📋 **Table**   | *Build a table of literacy rates across states* |\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    data = {}

    # ════════════════════════════════════════════════════════
    # NODE 1 — SUPERVISOR
    # ════════════════════════════════════════════════════════
    async with cl.Step(name="🧭  Node 1 · Supervisor", type="tool") as sup_step:
        sup_step.input = (
            f"**Input query:** `{message.content}`\n\n"
            f"**Task:** Classify intent → route to correct agent\n"
            f"**Model:** `llama-3.1-8b-instant`"
        )

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    f"{API_URL}/chat",
                    json={"query": message.content, "session_id": session_id},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            sup_step.output = f"❌ Server error `{e.response.status_code}` — check backend terminal"
            await cl.Message(content=f"❌ Backend error `{e.response.status_code}`. Check terminal logs.").send()
            return
        except httpx.HTTPError as e:
            sup_step.output = "❌ Cannot reach backend"
            await cl.Message(content=f"❌ Cannot connect to `{API_URL}`. Is the backend running?").send()
            return

        intent        = data.get("intent", "query")
        refused       = data.get("refused", False)
        refusal_reason = data.get("refusal_reason", "")
        citations     = data.get("citations", [])
        artifact_path = data.get("artifact_path", "")
        answer        = data.get("answer", "").strip()

        emoji, label, _ = INTENT_META.get(intent, ("⚙️", "Agent", ""))
        color = INTENT_COLOR.get(intent, "⬜")

        sup_step.output = (
            f"{color} **Intent classified:** `{intent}`\n\n"
            f"**Routing to:** {emoji} {label}\n\n"
            f"**Model used:** `llama-3.1-8b-instant` (8B — fast routing)"
        )

    # ════════════════════════════════════════════════════════
    # NODE 2 — AGENT (RAG / Summarizer / Artifact / Refuse)
    # ════════════════════════════════════════════════════════
    emoji, label, agent_desc = INTENT_META.get(intent, ("⚙️", "Agent", "Processing..."))

    async with cl.Step(name=f"{emoji}  Node 2 · {label}", type="tool") as agent_step:
        agent_step.input = (
            f"**Task:** {agent_desc}\n\n"
            f"**Model:** `llama-3.3-70b-versatile` (70B)\n\n"
            f"**Tool called:** `search_documents`"
            + (f"\n\n**top_k:** 5 chunks" if intent == "query" else
               f"\n\n**top_k:** 8 chunks (broad)" if intent == "summarize" else
               f"\n\n**top_k:** 6 chunks + `execute_python`" if intent == "artifact" else "")
        )

        if refused:
            if refusal_reason == "retrieval_empty":
                agent_step.output = (
                    "📭 **Retrieval result:** 0 chunks matched\n\n"
                    "**Action:** Graceful refusal — data not in loaded documents"
                )
            else:
                agent_step.output = (
                    "🚫 **Scope check failed**\n\n"
                    "**Reason:** Question is outside Indian Census 2011 scope\n\n"
                    "**Action:** Refused before search — no documents queried"
                )

        elif intent == "artifact" and artifact_path:
            agent_step.output = (
                f"**🔍 Tool called:** `search_documents` → retrieved chunks\n\n"
                f"**⚙️ Tool called:** `execute_python` → code executed\n\n"
                f"**📁 Artifact produced:** `{artifact_path}`\n\n"
                f"**✅ Status:** Success"
                + (f"\n\n**🔁 Retries:** {data.get('retry_count', 0)}" if data.get("retry_count") else "")
            )

        elif intent == "artifact" and not artifact_path:
            agent_step.output = (
                f"**🔍 Tool called:** `search_documents` → retrieved chunks\n\n"
                f"**⚙️ Tool called:** `execute_python`\n\n"
                f"**⚠️ Status:** Code executed but no artifact file produced\n\n"
                f"**🔁 Retries:** {data.get('retry_count', 0)}"
            )

        else:
            chunk_count = len(citations)
            sources = list({f"`{c['source_file']}`" for c in citations})
            sources_str = ", ".join(sources) if sources else "none"

            agent_step.output = (
                f"**🔍 Tool called:** `search_documents`\n\n"
                f"**📦 Chunks retrieved:** {chunk_count}\n\n"
                f"**📄 Source files:** {sources_str}\n\n"
                f"**✅ Status:** Answer generated with {chunk_count} citation(s)"
            )

    # ════════════════════════════════════════════════════════
    # NODE 3 — CITATIONS DETAIL (expanded per source)
    # ════════════════════════════════════════════════════════
    if citations and not refused:
        async with cl.Step(name=f"📚  Node 2b · Citations ({len(citations)} chunks)", type="tool") as cite_step:
            cite_step.input = "Deduplicating and ranking retrieved chunks by relevance"

            lines = []
            seen = set()
            for i, c in enumerate(citations):
                key = f"{c['source_file']}:{c['page_number']}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"**Chunk {i+1}**\n"
                    f"- 📄 File: `{c['source_file']}`\n"
                    f"- 📃 Page: {c['page_number']}\n"
                    f"- 🔑 Chunk ID: `{c.get('chunk_id', 'N/A')}`\n"
                    f"- 📝 Snippet: *\"{c.get('snippet', '')[:120]}...\"*"
                )

            cite_step.output = "\n\n---\n\n".join(lines)

    # ════════════════════════════════════════════════════════
    # NODE 3 — RESPONSE BUILDER
    # ════════════════════════════════════════════════════════
    async with cl.Step(name="✅  Node 3 · Response Builder", type="tool") as rb_step:
        rb_step.input = (
            "Deduplicating citations by chunk_id\n\n"
            "Appending artifact path if produced\n\n"
            "Saving turn to conversation history"
        )

        unique_sources = len({c['source_file'] for c in citations})
        rb_step.output = (
            f"**Citations deduplicated:** {len(citations)} chunks → {unique_sources} unique source(s)\n\n"
            f"**Artifact:** {'`' + artifact_path + '`' if artifact_path else 'None'}\n\n"
            f"**Response assembled** ✓"
        )

    # ════════════════════════════════════════════════════════
    # FINAL MESSAGE
    # ════════════════════════════════════════════════════════
    if not answer:
        await cl.Message(content="⚠️ Agent returned an empty response. Check the backend terminal.").send()
        return

    elements = []

    # Citations inline block
    if citations and not refused:
        citation_lines = "\n\n---\n**📚 Sources**\n"
        seen = set()
        for c in citations:
            key = f"{c['source_file']}:{c['page_number']}"
            if key in seen:
                continue
            seen.add(key)
            citation_lines += f"- `{c['source_file']}` — page {c['page_number']}\n"
            if c.get("snippet"):
                elements.append(
                    cl.Text(
                        name=f"📄 {c['source_file']}  p.{c['page_number']}",
                        content=c["snippet"],
                        display="side",
                    )
                )
        answer += citation_lines

    # Inline chart
    if artifact_path and artifact_path.endswith((".png", ".jpg", ".jpeg")):
        try:
            elements.append(cl.Image(name="📊 Generated Chart", path=artifact_path, display="inline"))
        except Exception:
            answer += f"\n\n📊 Chart saved at: `{artifact_path}`"

    # CSV download
    if artifact_path and artifact_path.endswith(".csv"):
        try:
            elements.append(cl.File(name="📋 Data Table (.csv)", path=artifact_path, display="inline"))
        except Exception:
            answer += f"\n\n📋 Table saved at: `{artifact_path}`"

    await cl.Message(content=answer, elements=elements).send()