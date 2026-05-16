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
10. [Observability](#10-observability)
11. [Failure Analysis](#11-failure-analysis)
12. [Corners Cut & What I'd Do With More Time](#12-corners-cut--what-id-do-with-more-time)

---

## 1. System Overview

The system is a multi-agent pipeline built on LangGraph that takes a user's natural-language question about census documents and routes it through the appropriate handling path — factual lookup, structured summary, or code-executed artifact (chart/table) — before returning a cited, grounded answer.

```
User
 │
 ▼
FastAPI /chat  ──────────────────────────────────────────────
                                                              │
                         LangGraph Graph                      │
                 ┌────────────────────────────┐               │
                 │                            │               │
                 │   [supervisor]             │               │
                 │        │                  │               │
                 │  ┌─────┼──────────┐       │               │
                 │  ▼     ▼          ▼       │               │
                 │ [rag] [summarizer] [artifact] [refuse]    │
                 │  │       │          │       │              │
                 │  └───────┴──────────┘───────┘             │
                 │              │                             │
                 │    [response_builder]                      │
                 │              │                             │
                 └──────────────┼─────────────────────────── ┘
                                │
                         Final Response
                         + Citations
                         + Artifact Path
                                │
                          Chainlit UI
```

Every factual claim is grounded in retrieved chunks. No node is allowed to assert a census figure it did not retrieve.

---

## 2. Architecture Decisions

### Why LangGraph over a single-agent ReAct loop?

A ReAct loop (reason → act → observe → repeat) with all tools available to one agent technically works, but creates two problems for this use case:

**Tool sprawl.** When a single agent has access to `search_documents`, `execute_python`, `write_file`, and `read_file` simultaneously, the LLM must reason about which tool to use and in what order on every step. For a summary question, it may "accidentally" try to execute code. For a code question, it may loop through search far more than needed. Separate agents with scoped tool sets remove this ambiguity — the Summarizer literally cannot call `execute_python`.

**Retry loops for code execution.** The Artifact Agent needs to cycle — write code, execute it, check for errors, fix the code, execute again. This is a cycle in the graph, which LangGraph supports natively. A plain ReAct loop can simulate this but with no explicit control over retry count, timeout policy, or bailout logic.

The Supervisor + specialized agent design gives us clean separation of concerns, explicit routing, and loop control — all things that matter when the system runs live in an interview.

### Why Groq (llama-3.3-70b-versatile)?

The assignment requires a provider with free credits and tool use support. Groq's llama-3.3-70b-versatile:
- Supports parallel tool calling natively
- Has a 128k context window (important for summarization with many chunks)
- Responds fast enough for interactive demo use (typically <2s for tool-use calls)
- Is free tier on Groq's API

The LLM is isolated behind `app/graph/llm.py`. Swapping to OpenAI, Anthropic, or any LangChain-compatible provider is a one-line change.

### Why ChromaDB over a managed vector DB?

The requirement is "quick to set up and run on our machine." ChromaDB runs in-process with no external service, persists to a local directory, and requires zero configuration beyond a path. For a demo system with O(hundreds of thousands) of chunks, this is exactly the right trade-off. At production scale, the retrieval layer would swap to Pinecone, Weaviate, or pgvector with zero changes to the rest of the system — the `get_vectorstore()` function in `retrieval/vectorstore.py` is the only boundary.

### Why Chainlit over Streamlit?

Both are acceptable per the requirements. Chainlit is chosen because:
- It has native `Step` components for showing agent reasoning inline (which satisfies "we should be able to see what happened")
- It has native `Element` types for inline image and file rendering — useful for chart artifacts
- It handles async streaming naturally — no `st.session_state` management

The trade-off: Chainlit is less widely known than Streamlit and has a smaller ecosystem. If the interviewer is more comfortable with Streamlit, swapping the UI is fully isolated to `ui/app.py` and does not touch any agent logic.

---

## 3. LangGraph Agent Graph

### Nodes

| Node | Responsibility | Tools Bound |
|---|---|---|
| `supervisor` | Classifies intent (`query / summarize / artifact / refuse`) | None |
| `query` (RAG Agent) | Retrieves relevant chunks and produces a cited factual answer | `search_documents` |
| `summarize` (Summarizer) | Broader retrieval (top_k=8), structures output under fixed headings | `search_documents` |
| `artifact` (Artifact Agent) | Writes Python, executes it, retries on error (max 3), saves output | `search_documents`, `execute_python`, `write_file` |
| `refuse` | Declines gracefully without hallucinating | None |
| `response_builder` | Deduplicates citations, appends artifact path, saves turn to history | `read_file` |

### Edges

```
supervisor ──[conditional on intent]──► query / summarize / artifact / refuse
query          ──► response_builder
summarize      ──► response_builder
artifact       ──► response_builder
refuse         ──► response_builder
response_builder ──► END
```

The conditional edge at `supervisor` is implemented via `route_by_intent()` which maps `state["intent"]` → node name. This is a pure function — easy to test independently of any LLM call.

### State

All nodes communicate through a single `AgentState` TypedDict. No node accepts arguments or returns values outside of state — this is LangGraph's contract. The state carries:

```
user_query            – the raw user input
conversation_history  – full prior turns (used for context in supervisor + agents)
intent                – set by supervisor, read by router
retrieved_chunks      – accumulated across tool calls
citations             – accumulated, deduplicated in response_builder
generated_code        – last code written by artifact agent
code_output / error   – stdout/stderr from execution
artifact_path         – path to produced file
retry_count           – guards the artifact retry loop
final_response        – assembled by response_builder
refused / refusal_reason
```

### Memory / Checkpointing

LangGraph's `MemorySaver` checkpointer persists the full graph state keyed by `thread_id` (= session ID). This means conversation history is automatically maintained across turns without any manual session management in the API layer. The `session_id` is created once per Chainlit session and passed as `thread_id` in every graph invocation.

---

## 4. Tool Design

Every tool is a LangChain `@tool` with a Pydantic `args_schema`. This gives us:
- Automatic JSON schema generation for LLM tool binding
- Input validation before the tool function body runs
- Clear, inspectable contracts for testing

### `search_documents`

Wraps ChromaDB's `similarity_search_with_score`. Returns a `SearchOutput` (list of `Chunk` objects with `source_file`, `page_number`, `text`, `score`). The tool is lazy-loaded — the vector store is not opened until the first search call, keeping startup fast.

The RAG and Summarizer agents differ only in `top_k`: RAG uses 5 (precise), Summarizer forces at least 8 (broader coverage for summaries).

### `execute_python`

This is the most complex tool. Key design decisions:

**Subprocess isolation.** Code runs in a subprocess (`subprocess.run`), not `exec()`. This means agent-written code cannot modify the Python runtime, cannot access the agent's memory, and is killed cleanly on timeout.

**Timeout.** Hard 30-second default, configurable. On `TimeoutExpired`, stderr contains a readable message and `success=False`.

**Matplotlib patch.** A preamble is prepended to every script that replaces `plt.show()` with a function that saves the figure to `workspace/artifacts/` and prints the path with a `[artifact_saved]` prefix. The agent does not need to know the save path — it just calls `plt.show()` as normal.

**Artifact detection.** The tool snapshots `workspace/artifacts/` before and after execution. The diff is the set of newly produced files, returned as `artifact_paths`. This catches any file produced by any means — `plt.savefig`, `df.to_csv`, `open(...)` — not just the matplotlib patch.

**Error feedback loop.** The Artifact Agent reads `success` and `stderr` from the output and injects them back into the LLM conversation. The LLM sees its own error and can correct the code. This loop runs up to `MAX_RETRIES` (3) times before the agent reports failure gracefully.

### `write_file` / `read_file`

Simple filesystem tools scoped to `workspace/`. They do not allow paths outside the workspace directory. `write_file` accepts a `subdir` parameter (`artifacts` or `notes`) to keep the working directory organized.

---

## 5. Retrieval & Citations

### Chunking strategy

Documents are split with `RecursiveCharacterTextSplitter` at 800 characters with 150-character overlap. This was chosen because:

- Census documents contain dense tables and short paragraphs. Large chunks (1500+) often mix multiple topics, reducing retrieval precision.
- 800 characters fits comfortably in a single LLM context slot while still containing enough surrounding text for the LLM to understand the claim.
- 150-character overlap ensures that a sentence spanning a chunk boundary is still retrievable from either chunk.

### Citation contract

Every `Citation` object carries:
- `source_file` — the filename, so the user can locate the document
- `page_number` — either parsed from a `## Page N` header in the markdown or derived from chunk index as a fallback
- `snippet` — first 200 characters of the chunk, shown in the Chainlit citation sidebar
- `chunk_id` — MD5 of `source_file::page::index`, used for deduplication in `response_builder`

The `response_builder` deduplicates by `chunk_id` before returning — multiple agents retrieving the same chunk do not produce duplicate citations.

### Why not PDF direct parsing?

The assignment provides markdown representations of the PDFs alongside the PDFs themselves. The markdown representations are used for ingestion because:
- Markdown is clean text — no PDF layout artifacts, no OCR errors
- Table structures in markdown are preserved as text, which the chunker handles well
- PDF parsing with `pymupdf` or `pdfplumber` often loses table cell boundaries

If only PDFs were available, `pymupdf` would be used with a table extraction pass before chunking.

---

## 6. Working Memory & Filesystem

```
workspace/
├── artifacts/    ← charts (.png), tables (.csv), any file produced by execute_python
├── history/      ← conversation turn snapshots (JSON, one file per session)
├── notes/        ← intermediate reasoning notes the agent can write and re-read
└── chroma/       ← ChromaDB persistent store
```

The `workspace/` directory is the agent's "scratch pad." It is mounted as a Docker volume so artifacts survive container restarts during a demo. It is gitignored.

The agent can write notes between turns using `write_file` with `subdir="notes"`. This is useful for multi-step artifact generation — the agent can write intermediate results to notes, then read them back on the next turn rather than re-retrieving.

Conversation history is maintained in two places:
1. `AgentState["conversation_history"]` — in-memory, managed by LangGraph's checkpointer, available within a session
2. `workspace/history/<session_id>.json` — on-disk snapshot written by `response_builder` after each turn, useful for debugging and the walkthrough video

---

## 7. Artifact Patterns

The `patterns/` directory contains markdown files that describe how to produce clean, consistent artifacts:

- `patterns/chart.md` — matplotlib code template with required elements (title, axis labels, source footer, colorblind-friendly palette)
- `patterns/table.md` — pandas DataFrame template with naming conventions, sort order, CSV export

These are loaded at Artifact Agent startup and injected into the system prompt. The agent reads the pattern before writing any code. This is the "reusable declarative patterns the agent reads at runtime" approach described in the assignment.

The advantage over hard-coded instructions in the system prompt is that patterns can be edited without changing any Python code, and new artifact types (e.g. `heatmap.md`, `choropleth.md`) can be added by dropping a file in `patterns/` without touching the agent.

---

## 8. Conversation Memory

Memory works at two levels:

**Within a session (LangGraph checkpointer).** The `MemorySaver` checkpointer stores the full `AgentState` after every graph run, keyed by `thread_id`. On the next turn, the graph is invoked with the same `thread_id` — LangGraph restores the last checkpoint, so `conversation_history` already contains all prior turns. The agent doesn't need to re-fetch or re-compute anything.

**Across sessions (filesystem).** If a user closes the tab and returns, the `thread_id` (session ID) is regenerated. The prior conversation is not automatically restored. To restore it, the API would need to load `workspace/history/<old_session_id>.json` and replay it into the new state. This is noted as a known limitation (see Section 12).

**Supervisor uses history.** The Supervisor receives the last 3 conversation turns alongside the new question. This helps it correctly classify follow-up questions like "now make a chart of that" — without prior context, this would be classified as `artifact` with no data reference.

---

## 9. Component Contracts (Pydantic)

All boundaries between components are typed. Key models in `app/models/schemas.py`:

| Model | Where Used |
|---|---|
| `Citation` | Produced by all agent nodes, deduplicated in response_builder |
| `Chunk` | Returned by `search_documents`, consumed by agent nodes |
| `Message` | Conversation history entries |
| `SearchInput / SearchOutput` | Tool input/output contract |
| `CodeExecutionInput / Output` | Tool input/output contract |
| `WriteFileInput / Output` | Tool input/output contract |
| `AgentResponse` | API response contract |
| `AgentState` | LangGraph shared state (TypedDict) |

Pydantic validation runs at the tool boundary — if the LLM generates a malformed tool call (e.g. passes a string where `top_k` expects an int), Pydantic raises a `ValidationError` before the tool function runs, which LangChain surfaces as a tool error message back to the LLM for self-correction.

---

## 10. Observability

**What fires and what it returns** is visible in three places:

1. **Chainlit `Step` component.** The UI wraps each API call in a `cl.Step("🔍 Thinking...")` that shows the detected intent. This is the user-facing trace.

2. **LangGraph's built-in tracing.** The compiled graph emits events for every node entry, node exit, and tool call. These can be piped to LangSmith with one environment variable (`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`). For the demo, they stream to stdout via `graph.stream()`.

3. **Subprocess stdout.** The `execute_python` tool prints `[artifact_saved] <path>` for every file produced. This appears in the API logs and is also captured in `code_output` on the state, so the agent can reference it.

To see the full trace for a question during the demo:
```bash
docker compose logs api -f
```

---

## 11. Failure Analysis

### Case 1: Table-heavy data lookup returns wrong numbers

**Input:** "What was the total population of Lucknow district in 2011?"

**What happens:** The chunker splits a census table across two chunks at the 800-character boundary. The row for Lucknow district is in chunk B, but the column headers ("District", "Total Population", "Urban", "Rural") are only in chunk A. Chunk B is retrieved (high similarity to "Lucknow population") but the LLM cannot parse the numbers without the headers, so it either returns the wrong column or hallucinates.

**Root cause:** Chunk boundary falls inside a table. Header context is lost.

**Proposed fix:** Table-aware chunking. Before the `RecursiveCharacterTextSplitter` pass, run a pre-processor that detects markdown tables and treats each table as an atomic unit (never splits inside one). Tables that exceed 800 characters are duplicated with their header row prepended to every continuation chunk.

---

### Case 2: Artifact agent produces a chart with hallucinated values

**Input:** "Show me a bar chart of literacy rates across all districts."

**What happens:** The census documents cover many districts across many files. The `search_documents` tool returns 5 chunks (default `top_k`), which cover perhaps 6-8 districts. The Artifact Agent uses only these retrieved values. The user sees a chart with 6-8 bars but expects all 75 districts. Worse, if the agent's prompt doesn't explicitly guard against it, it may fill in the "missing" districts from its training data (hallucinated values presented as census data).

**Root cause:** `top_k=5` is too small for an "all districts" query. The agent is not told how many districts exist, so it does not know the results are partial.

**Proposed fix:** Two changes. First, increase `top_k` for artifact queries that mention "all" or "across" — the Supervisor can detect this and set a `retrieval_mode: "exhaustive"` flag on state, which tells the Artifact Agent to use `top_k=50`. Second, inject a guard in the Artifact Agent system prompt: "Only plot districts for which you have retrieved data. Never infer or fill in values. Print a note in the chart title if results are partial: 'Showing N of M districts'."

---

### Case 3: Code execution hangs on an import of a missing library

**Input:** "Show me a choropleth map of population density." (user expects a geographic map)

**What happens:** The Artifact Agent writes code that imports `geopandas` or `folium`, neither of which is in `requirements.txt`. The `subprocess.run` call hits the 30-second timeout waiting for the import error (actually, the import fails fast, but the LLM may retry with `plotly`, then `cartopy`, each failing). After 3 retries the agent reports failure, but the error message to the user is cryptic (`ModuleNotFoundError: No module named 'geopandas'`).

**Root cause:** The agent has no inventory of what libraries are available in the runtime environment. It writes code using libraries it knows from training, not libraries that are installed.

**Proposed fix:** Inject the available library list into the Artifact Agent system prompt at startup: `"Available libraries: matplotlib, pandas, numpy. Do not import any other library."` This is generated dynamically from `pip list` at startup and cached. Additionally, the error handler in `artifact_agent.py` should detect `ModuleNotFoundError` specifically and respond with "This chart type requires a library not available in this environment. I can produce a matplotlib-based alternative instead." — skipping the remaining retry budget for this class of error.

---

## 12. Corners Cut & What I'd Do With More Time

**Cross-session memory.** Currently, closing the browser ends the session. The conversation history is written to disk, but the new session doesn't automatically load it. With more time: add a `GET /sessions` endpoint and a session-picker in the UI so the user can resume a prior conversation.

**PDF table extraction.** The ingestion pipeline uses markdown representations. If the markdown is absent or poorly formatted, table data is lost. With more time: add a `pymupdf` + `pdfplumber` pipeline that extracts tables as structured CSV before chunking, with a separate "table chunks" collection in ChromaDB queried alongside text chunks.

**Streaming responses.** The FastAPI endpoint returns the full response in one shot. Chainlit supports streaming, and LangGraph supports `graph.astream_events()`. With more time: stream the response token-by-token so the UI feels interactive even on slow LLM calls.

**Authentication.** The API has no authentication. In production, every endpoint would require a bearer token.

**Evaluation harness.** There are no automated end-to-end tests with real census data. With more time: build a small golden dataset (20 question-answer pairs with known correct citations) and run the full graph against it as a regression suite.

**Pinned dependencies.** `requirements.txt` uses `>=` version bounds. For a production ship, every dependency would be pinned to an exact version (`==`) and generated via `pip freeze` to guarantee reproducibility.

**Rate limit handling.** The Groq free tier has rate limits. Under load, the LLM calls will return 429 errors. With more time: add exponential backoff with jitter in `app/graph/llm.py` using `tenacity`.
