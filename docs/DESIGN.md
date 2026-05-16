# DESIGN.md — Census Document Q&A Chatbot

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Decisions](#2-architecture-decisions)
3. [LangGraph Agent Graph](#3-langgraph-agent-graph)
4. [Tool Design](#4-tool-design)
5. [Retrieval & Citations](#5-retrieval--citations)
6. [Working Memory & Filesystem](#6-working-memory--filesystem)
7. [Artifact Patterns](#7-artifact-patterns)
8. [Conversation Memory](#8-conversation-memory)
9. [Component Contracts (Pydantic)](#9-component-contracts-pydantic)
10. [Observability — UI Step Tracing](#10-observability--ui-step-tracing)
11. [Deployment Note](#11-deployment-note)
12. [Failure Analysis](#12-failure-analysis)
13. [Corners Cut & What I'd Do With More Time](#13-corners-cut--what-id-do-with-more-time)

---

## 1. System Overview

The system is a multi-agent pipeline built on LangGraph that takes a user's natural-language question about Indian Census 2011 documents and routes it through the appropriate handling path — factual lookup, structured summary, or code-executed artifact (chart/table) — before returning a cited, grounded answer.

```
User Query
    │
    ▼
FastAPI /chat
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Graph                    │
│                                                     │
│              ┌─────────────┐                        │
│              │  supervisor │  (llama-3.1-8b-instant)│
│              └──────┬──────┘                        │
│     ┌────────┬──────┴──────┬──────────┐             │
│     ▼        ▼             ▼          ▼             │
│  [query] [summarize]  [artifact]  [refuse]          │
│     │        │             │          │             │
│     └────────┴──────┬──────┘          │             │
│              ▼      │                 │             │
│   check_retrieval_result()            │             │
│        /          \                   │             │
│[retrieval_refuse]  │                  │             │
│        │    [response_builder] ◄──────┘             │
│        │           │                                │
│        └─────┬─────┘                                │
│              ▼                                      │
│             END                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
Chainlit UI
(per-node Step observability + inline artifact rendering)
```

Every factual claim is grounded in retrieved chunks. No node asserts a census figure it did not retrieve from ChromaDB.

---

## 2. Architecture Decisions

### Why LangGraph over a single-agent ReAct loop?

**Tool sprawl.** A single agent with all tools must reason about which to use every step. Separate agents with scoped tool sets remove this — the Summarizer literally cannot call execute_python.

**Retry loops for code execution.** The Artifact Agent cycles: write code → execute → check error → rewrite → execute again. LangGraph supports cycles natively with explicit retry count guards. A plain ReAct loop has no built-in loop control.

### Why two Groq models?

After testing all available Groq models via test_models.py:

| Model | Used by | Reason |
|---|---|---|
| llama-3.3-70b-versatile | RAG, Summarizer, Artifact agents | Best quality, confirmed tool calling works |
| llama-3.1-8b-instant | Supervisor only | Fast routing, no tools needed |

**Critical implementation note:** LangChain's bind_tools() caused Groq to generate tool calls in an invalid XML format instead of proper JSON, resulting in tool_use_failed 400 errors. All agents now use the Groq native client (from groq import Groq) with manually defined tool schemas. LangChain is still used for ChromaDB, embeddings, and document processing — just not for LLM tool binding.

**Two-phase tool calling in RAG Agent:** tool_choice="required" on the first call forces the search, tool_choice="none" on the answer call prevents the XML format bug that triggers when the model attempts a second tool call after receiving results.

### Why ChromaDB?

Runs in-process, no external service, persists to a local directory, zero configuration. Swap get_vectorstore() in retrieval/vectorstore.py for Pinecone, Weaviate, or pgvector at production scale — no other changes needed.

### Why Chainlit over Streamlit?

Native cl.Step components show per-node agent reasoning inline. Native cl.Image and cl.File elements render artifacts directly in chat. Handles async natively without st.session_state management.

### Supervisor keyword override

Beyond LLM classification, the Supervisor applies a hard-coded Python keyword check. Any query containing table, chart, graph, plot, bar, pie, csv, or visualization is forcibly routed to artifact regardless of LLM output. This prevents "build a table summarizing population" from being misclassified as summarize.

---

## 3. LangGraph Agent Graph

### Nodes

| Node | Responsibility | Model | Tools |
|---|---|---|---|
| supervisor | Classifies intent, routes | llama-3.1-8b-instant | None |
| query (RAG Agent) | Cited factual answer | llama-3.3-70b-versatile | search_documents |
| summarize (Summarizer) | Structured summary (top_k=8) | llama-3.3-70b-versatile | search_documents |
| artifact (Artifact Agent) | Code execution, chart/table | llama-3.3-70b-versatile | search_documents, execute_python, write_file |
| refuse | Out-of-scope graceful decline | — | None |
| retrieval_refuse | Searched but found zero chunks | — | None |
| response_builder | Dedupes citations, assembles output | — | None |

### Edges

```
supervisor ──[conditional: route_by_intent()]──► query / summarize / artifact / refuse

query     ──[conditional: check_retrieval_result()]──► retrieval_refuse / response_builder
summarize ──[conditional: check_retrieval_result()]──► retrieval_refuse / response_builder
artifact  ──[conditional: check_retrieval_result()]──► retrieval_refuse / response_builder

refuse           ──► response_builder
retrieval_refuse ──► END
response_builder ──► END
```

### Two distinct refusal types

| Type | Node | When | refusal_reason |
|---|---|---|---|
| Scope refusal | refuse_node | Supervisor classifies out-of-scope, no search done | out_of_scope |
| Retrieval refusal | retrieval_refuse_node | Search ran but returned zero chunks | retrieval_empty |

### Artifact filename uniqueness

Every chart gets a unique timestamped filename — nothing is ever overwritten:

```python
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"workspace/artifacts/figure_{ts}_{counter}.png"
```

---

## 4. Tool Design

### search_documents

Wraps ChromaDB similarity_search_with_score. Lazy-loaded — vector store opens only on first call. top_k varies by agent: 5 (RAG), 8 (Summarizer), 6 (Artifact).

### execute_python

- Subprocess isolation — runs in subprocess.run, not exec()
- Hard 30s timeout
- Matplotlib patch — plt.show() replaced with function that saves timestamped PNG
- Artifact detection — diffs workspace/artifacts/ before and after execution
- Retry loop — up to 3 rewrites on error, LLM sees stderr and corrects code

### write_file / read_file

Scoped to workspace/ only. write_file accepts subdir (artifacts or notes).

---

## 5. Retrieval & Citations

### Chunking strategy

RecursiveCharacterTextSplitter at 800 characters, 150-character overlap. Census documents have dense tables and short paragraphs — larger chunks mix topics, reducing retrieval precision.

### Citation contract

```
source_file   — filename
page_number   — parsed from ## Page N header or chunk index fallback
snippet       — first 200 chars, shown in Chainlit sidebar
chunk_id      — MD5(source_file::page::index), used for deduplication
```

response_builder deduplicates by chunk_id — the same chunk retrieved twice never produces duplicate citations.

---

## 6. Working Memory & Filesystem

```
workspace/
├── artifacts/    ← timestamped charts (.png), tables (.csv)
├── history/      ← per-session conversation JSON snapshots
├── notes/        ← intermediate agent reasoning notes
└── chroma/       ← ChromaDB persistent vector store
```

---

## 7. Artifact Patterns

patterns/chart.md and patterns/table.md describe how to produce consistent artifacts (required elements, conventions). Injected into Artifact Agent system prompt at startup. New types added by dropping a .md file in patterns/ — no Python changes needed.

---

## 8. Conversation Memory

**Within session** — MemorySaver checkpointer persists AgentState keyed by thread_id (= Chainlit session ID). Every graph invocation on the same thread restores full prior state.

**Across sessions** — session ID regenerates on browser refresh. Prior conversation not restored (known limitation — see Section 13).

**Supervisor context** — last 3 turns passed alongside new query for correct follow-up classification.

---

## 9. Component Contracts (Pydantic)

| Model | Where Used |
|---|---|
| Citation | All agent nodes → deduplicated in response_builder |
| Chunk | Returned by search_documents |
| Message | Conversation history entries |
| SearchInput / SearchOutput | Tool I/O |
| CodeExecutionInput / Output | Tool I/O |
| WriteFileInput / Output | Tool I/O |
| AgentResponse | FastAPI response |
| AgentState | LangGraph shared state (TypedDict) |

---

## 10. Observability — UI Step Tracing

Every LangGraph node renders as a collapsible cl.Step in the Chainlit UI showing inputs, model, tools called, chunk count, source files, and outputs.

```
🧭 Node 1 · Supervisor
  Input:  query text, model: llama-3.1-8b-instant
  Output: 🟦 Intent: query → Routing to 🔍 RAG Agent

🔍 Node 2 · RAG Agent
  Input:  model: llama-3.3-70b-versatile, tool: search_documents, top_k=5
  Output: Chunks: 3, Source: PC11_Odisha.md, Status: ✅

📚 Node 2b · Citations (3 chunks)
  Chunk 1: file, page, chunk_id, snippet preview...

✅ Node 3 · Response Builder
  Citations: 3 → 1 unique source, Artifact: None ✓
```

For artifact queries, the generated chart renders inline in the final message via cl.Image(display="inline"). CSV tables appear as cl.File download buttons.

---

## 11. Deployment Note

**Docker deployment was not tested due to hardware constraints on the development machine.**

A fully configured docker-compose.yml and Dockerfile are included in the repository. The API and UI services are defined with health checks, volume mounts for workspace/, and environment variable injection for GROQ_API_KEY.

The system was fully developed and validated by running backend and frontend locally in two separate terminals:

```bash
# Terminal 1 — FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload-dir app

# Terminal 2 — Chainlit UI
chainlit run ui/app.py --port 8001
```

All features — RAG, summarization, artifact generation with inline rendering, per-node observability, citation grounding, retrieval-level refusal, and scope-level refusal — were verified locally against real Indian Census 2011 markdown documents ingested into ChromaDB.

To run via Docker when hardware permits:
```bash
cp .env.example .env        # add GROQ_API_KEY
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8001
```

---

## 12. Failure Analysis

### Case 1: Table-heavy data lookup returns wrong numbers

**Input:** "What was the total population of Lucknow district in 2011?"

**What happens:** Chunker splits a census table at the 800-char boundary. The Lucknow row is in chunk B but column headers are only in chunk A. Chunk B is retrieved but the LLM cannot parse numbers without headers.

**Root cause:** Chunk boundary falls inside a markdown table.

**Proposed fix:** Table-aware chunking — pre-processor treats each markdown table as an atomic unit, prepends header row to every continuation chunk exceeding 800 chars.

---

### Case 2: Artifact agent produces chart with hallucinated values

**Input:** "Show me a bar chart of literacy rates across all districts."

**What happens:** top_k=6 returns ~6-8 districts. User expects all districts. Agent fills missing values from training data — hallucinated numbers presented as census data.

**Root cause:** top_k too small for "all districts" queries.

**Proposed fix:** Supervisor detects "all" / "across" keywords, sets retrieval_mode: "exhaustive" on state, raises top_k to 50. Agent prompt guard: "Only plot districts you retrieved. Note partial results in chart title."

---

### Case 3: Out-of-scope question returns census data

**Input:** "What is the population of France?"

**What happens:** Without explicit scope guards, Supervisor may route to query and RAG agent returns vaguely related chunks.

**Root cause:** LLM-only routing insufficient for geography boundary cases.

**Implemented fix (two layers):**
1. Supervisor prompt explicitly lists non-Indian geography as refuse with examples
2. RAG Agent system prompt: "If the question is about any country outside India, respond: 'This information is not available in the Indian Census 2011 documents.'"

---

## 13. Corners Cut & What I'd Do With More Time

**Cross-session memory.** Session ID regenerates on refresh. Fix: GET /sessions endpoint + session picker loading workspace/history/<id>.json.

**PDF table extraction.** Ingestion uses markdown. Fix: pymupdf + pdfplumber pipeline extracting tables as CSV, stored in a separate ChromaDB collection.

**Streaming responses.** /chat returns full response in one shot. Fix: graph.astream_events() + FastAPI StreamingResponse.

**Authentication.** API has no auth. Fix: bearer token on all endpoints.

**Evaluation harness.** No automated end-to-end tests. Fix: golden dataset of 20 Q&A pairs with known citations as regression suite.

**Pinned dependencies.** requirements.txt uses >= bounds. Fix: pip freeze → exact == pins.

**Rate limit handling.** Groq free tier returns 429s under load. Fix: exponential backoff with jitter via tenacity in app/graph/llm.py.

**Docker validation.** docker-compose.yml written but untested due to hardware constraints. Fix: validate on machine with sufficient RAM for embedding model and ChromaDB to load simultaneously in containers.
