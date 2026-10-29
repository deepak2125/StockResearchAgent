"""
LLM helpers — wrappers around langchain-ollama.

get_llm()             → ChatOllama instance (used by agent.py)
get_available_models()→ list of locally pulled Ollama models (used by ui.py)

stream_llm()          → kept for backward-compat; not used by the tool-calling agent
"""
from collections.abc import Iterator

import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .config import OLLAMA_MODEL, OLLAMA_BASE_URL


def get_llm(model: str | None = None) -> ChatOllama:
    """Return a ChatOllama instance for the given (or default) model."""
    return ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        streaming=True,
    )


def get_available_models() -> list[str]:
    """Return locally available Ollama model names."""
    try:
        result = ollama.list()
        models = (
            result.get("models", [])
            if isinstance(result, dict)
            else getattr(result, "models", [])
        )
        return [(m["name"] if isinstance(m, dict) else m.model) for m in models]
    except Exception:
        return [OLLAMA_MODEL]


# ── Kept for backward-compat (used by run_tool_agent fallback path) ───────────

_SYSTEM_PROMPT = """You are a concise, professional financial analyst assistant.
Summarise the provided market data clearly, highlighting key movers and trends.
Use plain English. Keep responses under 400 words unless asked for more detail."""


def stream_llm(
    user_question: str,
    market_context: str,
    model: str | None = None,
) -> Iterator[str]:
    """Stream LLM tokens given pre-fetched market context text."""
    llm = get_llm(model)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Here is the latest market data:\n\n{market_context}\n\n"
                f"User question: {user_question}"
            )
        ),
    ]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
