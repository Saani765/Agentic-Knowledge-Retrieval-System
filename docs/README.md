# Census Document Q&A Chatbot

A multi-agent LangGraph system for querying government census documents with citations, summaries, and auto-generated charts/tables.

## Architecture

```
User Query
    │
    ▼
Supervisor  ──routes──►  RAG Agent        (factual Q&A + citations)
                     ──►  Summarizer Agent  (structured summaries)
                     ──►  Artifact Agent   (charts + tables via code execution)
                     ──►  Refuse Node      (graceful out-of-scope decline)
                              │
                              ▼
                     Response Builder  ──► Final Answer
```

## Quick Start

### 1. Clone and configure
```bash
git clone <repo>
cd census-chatbot
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 2. Add your documents
Place census PDF/markdown files in `data/documents/`

### 3. Ingest documents
```bash
pip install -r requirements.txt
python -m app.retrieval.vectorstore
```

### 4. Run with Docker
```bash
docker compose up --build
```

- API: http://localhost:8000
- Chat UI: http://localhost:8001

### 5. Run tests
```bash
pytest tests/ -v
```

## Project Structure

```
census-chatbot/
├── app/
│   ├── graph/          # LangGraph nodes + builder
│   ├── tools/          # search, execute_python, file I/O
│   ├── retrieval/      # ChromaDB ingestion + search
│   ├── models/         # Pydantic schemas + AgentState
│   └── api/            # FastAPI routes
├── ui/                 # Chainlit frontend
├── patterns/           # Artifact recipe templates
├── workspace/          # Runtime artifacts (gitignored)
├── tests/
├── docker-compose.yml
└── requirements.txt
```

## Tools per Agent

| Agent | Tools |
|---|---|
| Supervisor | None (pure routing) |
| RAG Agent | `search_documents` |
| Summarizer | `search_documents` (top_k=8) |
| Artifact Agent | `search_documents`, `execute_python`, `write_file` |
| Response Builder | `read_file` |

Local Runs : Backend -> uvicorn main:app --host 0.0.0.0 --port 8000 
                        uvicorn main:app --reload --host 0.0.0.0 --port 8000

                UI   -> chainlit run ui/app.py --host 0.0.0.0 --port 8001

                Ingestion -> python -m app.retrieval.vectorstore