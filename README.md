# Stock Research Agent

An AI agent with a Gradio web UI that orchestrates:
- **Ollama** (local LLM) — for natural language market analysis
- **SerpAPI Google Finance Markets** — for live market data

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| [uv](https://docs.astral.sh/uv/) | latest |
| [Ollama](https://ollama.com/) | running locally on port 11434 |
| SerpAPI account | free or paid key |

## LangGraph agent graph

```
validate ──(ok)──► fetch ──(ok)──► format ──► analyze ──► END
    │                 │
  (error)           (error)
    └──────────────────┴──► error ──► END
```

Each node is a pure function that receives and returns `AgentState`:

| Node | What it does |
|---|---|
| `validate` | Checks the market category is a known trend |
| `fetch` | Calls SerpAPI Google Finance Markets endpoint |
| `format` | Converts raw JSON to a human-readable text table |
| `analyze` | Calls Ollama via `langchain-ollama` `ChatOllama` |
| `error` | Pass-through — error message is in state |

### Install a model in Ollama
```bash
ollama pull llama3.2
```

---

## Setup

```bash
# 1. Clone / enter the project
cd StockResearchAgent

# 2. Copy the env template and fill in your key
cp .env.example .env
# Edit .env — set SERPAPI_API_KEY

# 3. Install dependencies with uv (creates a .venv automatically)
uv sync

# 4. Launch the UI
uv run python -m stock_agent
```

Open **http://localhost:7860** in your browser.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `SERPAPI_API_KEY` | *(required)* | Your SerpAPI key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Default Ollama model |

---

## Market categories

| Value | Description |
|---|---|
| `indexes` | Major market indexes (default) |
| `most-active` | Most actively traded stocks |
| `gainers` | Top gaining stocks |
| `losers` | Top losing stocks |
| `climate-leaders` | ESG / climate-leader stocks |
| `cryptocurrencies` | Crypto market overview |
| `currencies` | Currency pairs |

---

## Project structure

```
StockResearchAgent/
├── src/
│   └── stock_agent/
│       ├── __init__.py
│       ├── __main__.py      # uv run entry-point
│       ├── config.py        # env-based configuration
│       ├── market_data.py   # SerpAPI client
│       ├── llm.py           # Ollama streaming client
│       ├── agent.py         # Orchestrator
│       └── ui.py            # Gradio interface
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```
