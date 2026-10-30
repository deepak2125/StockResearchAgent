"""
LangGraph tool-calling agent with automatic fallback.

Primary path  — native tool-calling (models that support it: llama3.1+, mistral-nemo, etc.)
Fallback path — prompt-based tool selection (works with any model)

The fallback asks the LLM to output a JSON decision describing which tool
to call, executes the tool, then asks the LLM again with the real data.

Tools
-----
  get_market_data(country_or_region)  →  live market snapshot for a region/country
  get_stock_price(ticker)             →  quote + stats + financials + news
"""
from __future__ import annotations

import json
import re
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


# Prompt used by the fallback path to get a structured tool-selection decision
_TOOL_SELECTION_PROMPT = """You are a tool-routing assistant. Based on the user question below, decide which tool(s) to call.

Available tools:
  1. get_market_data(country_or_region: str)
     Use for: questions about a country's market, a region (US, Europe, Asia), or asset class (crypto, currencies, futures).
     Example args: "US", "India", "Europe", "crypto", "currencies"

  2. get_stock_price(ticker: str)
     Use for: questions about a specific company or stock ticker.
     Ticker format: SYMBOL:EXCHANGE  e.g. AAPL:NASDAQ, RELIANCE:NSE, TSLA:NASDAQ, TCS:NSE, BTC-USD
     If the user gives a company name without a ticker, infer the most likely ticker.

Respond ONLY with a JSON array of tool calls. No other text. Example:
[
  {{"tool": "get_stock_price", "args": {{"ticker": "AAPL:NASDAQ"}}}},
  {{"tool": "get_market_data", "args": {{"country_or_region": "US"}}}}
]

If no tool is needed (e.g. a general greeting), return an empty array: []

User question: {question}"""


# ── Tool executor ─────────────────────────────────────────────────────────────

_TOOL_MAP: dict[str, Any] = {
    "get_market_data": get_market_data,
    "get_stock_price": get_stock_price,
}


def _execute_tool_call(name: str, args: dict, call_id: str = "0") -> ToolMessage:
    """Execute a tool by name and return a ToolMessage."""
    tool_fn = _TOOL_MAP.get(name)
    if tool_fn is None:
        result = f"Unknown tool: {name}"
    else:
        try:
            result = tool_fn.invoke(args)
        except Exception as exc:
            result = f"Tool error: {exc}"
    return ToolMessage(content=str(result), tool_call_id=call_id)


# ── Native tool-calling path ──────────────────────────────────────────────────

def _run_native_tool_agent(
    user_question: str,
    llm: Any,
) -> Iterator[str]:
    """ReAct loop using llm.bind_tools() — works only on tool-capable models."""
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    response: AIMessage | None = None
    for _ in range(5):  # max iterations
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for tc in tool_calls:
            name  = tc.get("name", "?")
            args  = tc.get("args", {})
            label = list(args.values())[0] if args else ""
            yield f"🔧 Calling **{name}**({label})…\n\n"

            tool_msg = _execute_tool_call(name, args, tc.get("id", "0"))
            messages.append(tool_msg)

            preview = tool_msg.content[:600]
            suffix  = "…" if len(tool_msg.content) > 600 else ""
            yield f"📊 **Tool result preview:**\n```\n{preview}{suffix}\n```\n\n"
    else:
        yield "⚠️ Agent reached max iterations.\n\n"

    yield "---\n\n🤖 **AI Analysis:**\n\n"

    final_text = (response.content if response and isinstance(response.content, str) else "")
    if final_text:
        yield final_text
    else:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content


# ── Prompt-based fallback path ────────────────────────────────────────────────

def _parse_tool_calls(raw: str) -> list[dict]:
    """Extract the JSON array from the LLM's tool-selection response."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find the first [...] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def _run_fallback_agent(
    user_question: str,
    llm: Any,
) -> Iterator[str]:
    """
    Prompt-based tool selection for models that don't support bind_tools().

    Step 1: Ask the LLM which tool(s) to call → get JSON
    Step 2: Execute the tools
    Step 3: Ask the LLM to analyse the results
    """
    # Step 1 — tool selection
    selection_prompt = _TOOL_SELECTION_PROMPT.format(question=user_question)
    selection_response = llm.invoke([HumanMessage(content=selection_prompt)])
    raw_selection = selection_response.content if hasattr(selection_response, "content") else str(selection_response)

    tool_calls = _parse_tool_calls(raw_selection)

    # Step 2 — execute tools
    tool_results: list[str] = []
    if tool_calls:
        for tc in tool_calls:
            name = tc.get("tool", "")
            args = tc.get("args", {})
            if not name or name not in _TOOL_MAP:
                continue
            label = list(args.values())[0] if args else ""
            yield f"🔧 Calling **{name}**({label})…\n\n"

            tool_msg = _execute_tool_call(name, args)
            preview = tool_msg.content[:600]
            suffix  = "…" if len(tool_msg.content) > 600 else ""
            yield f"📊 **Tool result preview:**\n```\n{preview}{suffix}\n```\n\n"
            tool_results.append(f"[{name}({label})]\n{tool_msg.content}")
    else:
        # No tools needed — answer directly
        yield "---\n\n🤖 **AI Analysis:**\n\n"
        for chunk in llm.stream([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_question),
        ]):
            if chunk.content:
                yield chunk.content
        return

    # Step 3 — final analysis with tool data
    yield "---\n\n🤖 **AI Analysis:**\n\n"

    context = "\n\n".join(tool_results)
    final_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Here is the live market data you retrieved:\n\n{context}\n\n"
            f"Now answer the user's question: {user_question}"
        )),
    ]
    for chunk in llm.stream(final_messages):
        if chunk.content:
            yield chunk.content


# ── Public entry-point ────────────────────────────────────────────────────────

def run_tool_agent(
    user_question: str,
    model: str | None = None,
) -> Iterator[str]:
    """
    Try native tool-calling first; fall back to prompt-based selection
    automatically if the model reports it doesn't support tools.
    """
    llm = get_llm(model)

    try:
        yield from _run_native_tool_agent(user_question, llm)
    except Exception as exc:
        err = str(exc).lower()
        # Models that don't support tool-calling return a 400 with "does not support tools"
        if "does not support tools" in err or "tool" in err and "400" in err:
            yield f"ℹ️ Model doesn't support native tool-calling — using prompt-based selection.\n\n"
            yield from _run_fallback_agent(user_question, llm)
        else:
            yield f"❌ Error: {exc}"


# ── Convenience wrappers (called by ui.py) ────────────────────────────────────

def run_agent(
    user_question: str,
    trend: str = "indexes",
    model: str | None = None,
) -> Iterator[str]:
    yield from run_tool_agent(user_question=user_question, model=model)


def run_stock_agent(
    ticker: str,
    user_question: str,
    model: str | None = None,
) -> Iterator[str]:
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
