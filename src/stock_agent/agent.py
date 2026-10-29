"""
LangGraph tool-calling agent.

Architecture
------------

  User question
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │              LangGraph ReAct Agent                  │
  │                                                     │
  │  ┌──────────┐     tool_call?     ┌───────────────┐  │
  │  │  agent   │ ─────────────────► │ tool_executor │  │
  │  │ (Ollama) │ ◄───────────────── │               │  │
  │  └──────────┘    tool_result     └───────────────┘  │
  │       │                                             │
  │  no more tools                                      │
  │       │                                             │
  │       ▼                                             │
  │   final answer                                      │
  └─────────────────────────────────────────────────────┘

Tools available to the LLM
---------------------------
  get_market_data(country_or_region)  →  market snapshot for a region
  get_stock_price(ticker)             →  quote + stats + financials + news

The LLM (ChatOllama) reads both tool schemas and decides which to call
based purely on the user's question.  If the user asks "How is India doing
and what is Reliance's stock price?" the LLM will call BOTH tools.

Flow per turn
-------------
  1. LLM receives [SystemMessage, HumanMessage(question)]
  2. If LLM issues a tool_call → tool_executor runs the tool → result appended
  3. LLM sees the tool result and either calls another tool or writes a final answer
  4. Tokens streamed to Gradio as they arrive
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from .llm import get_llm
from .tools import ALL_TOOLS, get_market_data, get_stock_price

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Finley, a senior personal financial advisor with 20 years of experience across equity markets, fixed income, and portfolio management.

Your personality:
  • Warm, trustworthy, and direct — you speak like a knowledgeable friend, not a textbook.
  • You explain complex ideas in plain English without dumbing them down.
  • You are candid about risk: you never sugarcoat downside scenarios.
  • You tailor your advice to the user's apparent goal (growth, income, preservation, curiosity).

You have access to live market data tools:
  • get_market_data(country_or_region) — use when the user asks about a country, region, or asset class
  • get_stock_price(ticker)            — use when the user asks about a specific company or stock

How you work:
  1. ALWAYS call the appropriate tool(s) first. Never invent or guess prices or data.
  2. After receiving live data, give your honest professional opinion.
  3. Structure your response as a brief advisor would in a client meeting:
       – What the data shows (facts)
       – What it means for the user (insight)
       – What to watch or consider next (actionable guidance)
  4. Flag significant risks clearly — use "⚠️ Risk:" when warranted.
  5. Keep responses focused and under 400 words unless the user asks for deeper analysis.
  6. End with a short follow-up question to keep the conversation going when appropriate.

Disclaimer: remind the user once per conversation (not every message) that your analysis
is for informational purposes only and does not constitute regulated financial advice."""


# ── Tool executor ─────────────────────────────────────────────────────────────

_TOOL_MAP: dict[str, Any] = {
    "get_market_data": get_market_data,
    "get_stock_price": get_stock_price,
}


def _execute_tool_call(tool_call: dict) -> ToolMessage:
    """Execute a single tool call dict and return a ToolMessage."""
    name = tool_call["name"]
    args = tool_call.get("args", {})
    tool_fn = _TOOL_MAP.get(name)

    if tool_fn is None:
        result = f"Unknown tool: {name}"
    else:
        try:
            result = tool_fn.invoke(args)
        except Exception as exc:
            result = f"Tool error: {exc}"

    return ToolMessage(content=str(result), tool_call_id=tool_call["id"])


# ── Core agent loop ───────────────────────────────────────────────────────────

def run_tool_agent(
    user_question: str,
    model: str | None = None,
) -> Iterator[str]:
    """
    Run the tool-calling ReAct loop and stream the final answer.

    Flow
    ----
    1. Build messages: [SystemMessage, HumanMessage]
    2. Call LLM with tools bound
    3. If response has tool_calls → execute each → append ToolMessages → go to 2
    4. If no tool_calls → stream the final text answer token by token

    Yields
    ------
    str
        Progress indicators and final LLM tokens for Gradio.
    """
    llm = get_llm(model)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    # ── ReAct loop (max 5 iterations to prevent runaway) ─────────────────────
    max_iterations = 5
    for iteration in range(max_iterations):

        # Non-streaming call to check for tool use first
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            # No tools called — this is the final answer, stream it
            break

        # ── Execute tool calls ────────────────────────────────────────────
        for tc in tool_calls:
            name  = tc.get("name", "?")
            args  = tc.get("args", {})
            label = list(args.values())[0] if args else ""
            yield f"🔧 Calling **{name}**({label})…\n\n"

            tool_msg = _execute_tool_call(tc)
            messages.append(tool_msg)

            # Show a brief preview of the raw data
            preview = tool_msg.content[:600]
            ellipsis = "…" if len(tool_msg.content) > 600 else ""
            yield f"📊 **Tool result preview:**\n```\n{preview}{ellipsis}\n```\n\n"

    else:
        # Hit max iterations without a final answer
        yield "⚠️ Agent reached max iterations. Partial results above may still be useful."
        return

    # ── Stream the final answer ───────────────────────────────────────────────
    yield "---\n\n🤖 **AI Analysis:**\n\n"

    # The last message is the AIMessage with the final text — stream it
    final_text = response.content if isinstance(response.content, str) else ""
    if final_text:
        # Simulate streaming by yielding the whole text
        # (invoke() already completed, we just output it)
        yield final_text
    else:
        # Fallback: stream a fresh response from the accumulated messages
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content


# ── Convenience wrappers kept for backward compat with ui.py ─────────────────

def run_agent(
    user_question: str,
    trend: str = "indexes",   # kept for signature compat, ignored
    model: str | None = None,
) -> Iterator[str]:
    """
    Backward-compatible wrapper used by the Market Overview tab.
    Passes the question directly to the tool-calling agent.
    """
    yield from run_tool_agent(user_question=user_question, model=model)


def run_stock_agent(
    ticker: str,
    user_question: str,
    model: str | None = None,
) -> Iterator[str]:
    """
    Backward-compatible wrapper used by the Stock Lookup tab.
    Enriches the question with the ticker and passes to the tool-calling agent.
    """
    ticker = ticker.strip().upper().replace(" ", ":")
    if not ticker:
        yield "❌ Please enter a stock ticker, e.g. **AAPL:NASDAQ** or **RELIANCE:NSE**."
        return

    enriched = (
        f"{user_question} (Stock: {ticker})"
        if user_question.strip()
        else f"Give me a comprehensive analysis of {ticker}: price, key stats, financials, and recent news."
    )
    yield from run_tool_agent(user_question=enriched, model=model)
