# newsroomAgent

Work-in-progress newsroom research assistant. LangGraph pipeline, RAG over a news corpus, custom MCP server.

## What it does

Ingests a folder of news articles into a local vector store, then answers questions about them with citations back to the original files. Exposes the same retrieval through a FastAPI endpoint and an MCP server, so it can be driven from a browser, curl, or an MCP-aware agent like Claude.

## Stack

- Python 3.12, uv
- LangChain + LangGraph for the pipeline
- Chroma for the vector store, Ollama (`nomic-embed-text`) for embeddings
- Anthropic Claude (`claude-sonnet-4-6`) for generation
- FastAPI for the HTTP layer, FastMCP for the MCP server
- Tavily for live web search

## Architecture

- `newsroomagent/ingest.py` loads `data/raw/*.txt`, chunks with a recursive splitter, embeds, and persists to `data/chroma/`.
- `newsroomagent/retrieval.py` opens the persisted store and runs top-k similarity search, then prompts Claude with a citation-aware template.
- `newsroomagent/mcp_server.py` exposes `archive_search`, `web_search`, and `get_current_time` over stdio so an MCP client can use the archive as a tool.
- `newsroomagent/graph.py` defines the LangGraph worker nodes (researcher, fact-checker) that consume the MCP tools. Researcher gathers notes with citations while fact-checker verifies claims against the archive and emits structured verdicts.
- `main.py` serves a FastAPI app with `/health` and `POST /research`.

## Quickstart

Install dependencies:

```bash
uv sync
```

Build the vector store from `data/raw/*.txt`. Rerun whenever source articles or chunking config changes:

```bash
uv run python -c "from newsroomagent.ingest import ingest; ingest()"
```

### Option 1: HTTP API

```bash
uv run uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for Swagger, or:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic":"What recent elections happened?","k":3}'
```

### Option 2: MCP server

Runs over stdio for use with an MCP client:

```bash
uv run python -m newsroomagent.mcp_server
```

Requires `TAVILY_API_KEY` in the environment for `web_search`.

### Option 3: LangGraph pipeline

Smoke test for the researcher and fact checker nodes. Spawns the MCP server, runs the researcher node on a sample topic, and prints its notes. Feeds the notes into the fact-checker, and prints verdicts with evidence.

```bash
uv run python -m newsroomagent.graph
```

### Option 4: Retrieval via REPL

```bash
uv run python
```

```python
from newsroomagent.retrieval import retrieve
chunks = retrieve("What election occurred recently?", k=3)
for c in chunks:
    print(c.metadata["source"])
```

## Status

Active prototype. Working: ingest, retrieval, citation-aware answers, FastAPI endpoint, MCP server with archive + web search. Next: LangGraph multi-step research loop, evaluation harness for retrieval quality.
