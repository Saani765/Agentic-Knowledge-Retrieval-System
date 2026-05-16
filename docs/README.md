# Knowledge Retrieval  Q&A System

A multi-agent RAG system for querying  documents with citations, summaries, and auto-generated charts/tables.

Built with **LangGraph**, **Groq LLaMA**, **ChromaDB**, **FastAPI**, and **Chainlit**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  Supervisor │  llama-3.1-8b-instant — classifies intent, routes
└──────┬──────┘
       │
  ┌────┴─────────────────────┐
  ▼        ▼                 ▼          ▼
[query] [summarize]      [artifact]  [refuse]
  │        │                 │          │
  └────────┴────┬────────────┘          │
                ▼                       │
   check_retrieval_result()             │
      /             \                   │
[retrieval_refuse]  [response_builder] ◄┘
                          │
                         END
```

All agents use `llama-3.3-70b-versatile` via **Groq native client** (not LangChain bind_tools — avoids tool_use_failed errors). Every node is visible as a collapsible step in the Chainlit UI.

---

## Query Types

| Type | Example | Agent |
|---|---|---|
| 🔍 Factual | *Literacy rate in Uttar Pradesh 2011?* | RAG Agent |
| 📝 Summary | *Summarize key findings from Odisha* | Summarizer |
| 🎨 Chart | *Bar chart of male vs female population* | Artifact Agent |
| 📋 Table | *Build a table of literacy rates* | Artifact Agent |
| 🚫 Out of scope | *Population of France?* | Refuse |

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo>
cd census-chatbot
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Add census documents

Place Indian Census 2011 markdown files in `data/documents/`

### 4. Ingest documents into ChromaDB

```bash
python -m app.retrieval.vectorstore
```

You should see:
```
Ingesting: PC11_PCA_Data_Highlights_Odisha.md
  → 342 chunks added
Ingestion complete. Total chunks: XXXX
```

### 5. Run backend (Terminal 1)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload-dir app
```

Test it's alive:
```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

### 6. Run frontend (Terminal 2)

```bash
chainlit run ui/app.py --port 8001
```

Open `http://localhost:8001` in your browser.

---

## Docker (not tested locally due to hardware constraints)

A `docker-compose.yml` and `Dockerfile` are included. To run when hardware permits:

```bash
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8001
```

---

## Project Structure

```
census-chatbot/
├── main.py                    ← FastAPI entrypoint
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── test_models.py             ← run to find working Groq models
│
├── app/
│   ├── graph/
│   │   ├── builder.py         ← compiles LangGraph with all nodes + edges
│   │   ├── supervisor.py      ← intent classification + keyword override
│   │   ├── rag_agent.py       ← factual Q&A, tool_choice=required/none pattern
│   │   ├── summarizer.py      ← structured summary, top_k=8
│   │   ├── artifact_agent.py  ← code write + execute + retry loop
│   │   ├── response_builder.py← deduplication + refusal nodes
│   │   └── llm.py             ← model constants + Groq client factory
│   │
│   ├── tools/
│   │   ├── search.py          ← search_documents (ChromaDB)
│   │   ├── code_executor.py   ← execute_python (subprocess + timestamp filenames)
│   │   └── file_tools.py      ← write_file / read_file
│   │
│   ├── retrieval/
│   │   └── vectorstore.py     ← ingest + ChromaDB setup
│   │
│   ├── models/
│   │   └── schemas.py         ← AgentState + all Pydantic models
│   │
│   └── api/
│       └── routes.py          ← /chat + /health endpoints
│
├── ui/
│   └── app.py                 ← Chainlit UI with per-node Steps
│
├── patterns/
│   ├── chart.md               ← matplotlib recipe for Artifact Agent
│   └── table.md               ← pandas CSV recipe for Artifact Agent
│
├── tests/
│   └── test_components.py     ← 15 unit tests
│
├── docs/
│   ├── README.md
│   └── DESIGN.md
│
└── workspace/                 ← runtime only, gitignored
    ├── artifacts/             ← timestamped charts + tables
    ├── history/
    ├── notes/
    └── chroma/
```

---

## Tool → Agent Binding

| Tool | RAG Agent | Summarizer | Artifact Agent |
|---|---|---|---|
| search_documents | ✅ top_k=5 | ✅ top_k=8 | ✅ top_k=6 |
| execute_python | ❌ | ❌ | ✅ |
| write_file | ❌ | ❌ | ✅ |

---

## Key Implementation Notes

**Groq native client** — LangChain's bind_tools() caused tool_use_failed errors with llama-3.3-70b-versatile (XML format instead of JSON). All agents use the Groq Python SDK directly with manually defined tool schemas.

**Two-phase tool calling (RAG)** — tool_choice="required" forces the search call, tool_choice="none" on the answer call prevents the XML bug on the second invocation.

**Unique artifact filenames** — charts are saved as figure_YYYYMMDD_HHMMSS_N.png. Nothing is ever overwritten across multiple queries.

**Dual refusal system** — scope refusal (Supervisor, before search) and retrieval refusal (after search returns zero chunks) are separate nodes with distinct refusal_reason values.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description |
|---|---|
| GROQ_API_KEY | Your Groq API key (required) |
| DOCS_DIR | Path to census markdown files (default: data/documents) |
| CHROMA_DIR | ChromaDB persistence path (default: workspace/chroma) |