"""
LangChain tool definitions.

Two tools are exposed to the LLM:
  1. get_market_data  — returns market data for a given country/region
  2. get_stock_price  — returns a quote + fundamentals for a given company ticker

The LLM (ChatOllama) sees these tool schemas and decides which to call
based on the user's question.  LangGraph's prebuilt ReAct agent handles
the tool-call → execute → respond loop automatically.
"""
from langchain_core.tools import tool

from .market_data import (
    fetch_market_data,
    format_market_data_as_text,
    fetch_stock_data,
    format_stock_data_as_text,
    MarketDataError,
)

# Country/region → section key that appears in the SerpAPI indexes response
# Used so the LLM can say "India" and we know which part of the response to highlight
COUNTRY_SECTION_MAP: dict[str, str] = {
    # Americas
    "us": "us", "usa": "us", "united states": "us", "america": "us",
    # Europe
    "europe": "europe", "european": "europe", "eu": "europe",
    "uk": "europe", "germany": "europe", "france": "europe",
    # Asia
    "asia": "asia", "india": "asia", "japan": "asia",
    "china": "asia", "hong kong": "asia",
    # Other sections always in response
    "crypto": "crypto", "cryptocurrency": "crypto", "bitcoin": "crypto",
    "currencies": "currencies", "forex": "currencies", "fx": "currencies",
    "futures": "futures", "commodities": "futures",
}


@tool
def get_market_data(country_or_region: str) -> str:
    """
    Fetch live stock market data for a given country or region.

    Use this tool when the user asks about market performance for a specific
    country, region, or asset class such as:
    - A country: "India", "US", "Japan", "Germany"
    - A region:  "Europe", "Asia"
    - An asset class: "crypto", "currencies", "futures"

    Parameters
    ----------
    country_or_region : str
        The country, region, or asset class name, e.g. "India", "US", "crypto".

    Returns
    -------
    str
        Formatted market snapshot text with prices and movements.
    """
    try:
        raw = fetch_market_data(trend="indexes")
    except MarketDataError as exc:
        return f"Error fetching market data: {exc}"

    full_text = format_market_data_as_text(raw)

    # Try to filter to the relevant section if a specific country was asked for
    key = COUNTRY_SECTION_MAP.get(country_or_region.lower().strip())
    if key:
        markets = raw.get("markets", {})
        section_data = markets.get(key)
        if section_data and isinstance(section_data, list):
            from .config import MARKET_SECTIONS
            label = MARKET_SECTIONS.get(key, key.upper())
            lines = [f"=== {label} ==="]
            for item in section_data:
                name = item.get("name", "N/A")
                price = item.get("price", 0) or 0
                mv = item.get("price_movement", {}) or {}
                direction = mv.get("movement", "")
                pct = mv.get("percentage", 0.0) or 0.0
                val = mv.get("value", 0.0) or 0.0
                arrow = "▲" if direction == "Up" else "▼" if direction == "Down" else "─"
                sign = "+" if direction == "Up" else "-"
                try:
                    lines.append(
                        f"  {arrow} {name:<22} {float(price):>12,.2f}   "
                        f"({sign}{abs(pct):.2f}%  {sign}{abs(val):.2f})"
                    )
                except (TypeError, ValueError):
                    lines.append(f"  {arrow} {name}")
            return "\n".join(lines)

    # No specific region matched — return everything
    return full_text


@tool
def get_stock_price(ticker: str) -> str:
    """
    Fetch the current stock price, key statistics, financials, and recent news
    for a specific company.

    Use this tool when the user asks about:
    - A specific stock or company, e.g. "Apple", "Reliance", "Tesla", "TCS"
    - A ticker symbol, e.g. "AAPL", "RELIANCE:NSE", "TSLA:NASDAQ"
    - Price, valuation, fundamentals, or news for a single company

    The ticker should be in SYMBOL:EXCHANGE format, e.g.:
      AAPL:NASDAQ, GOOGL:NASDAQ, TSLA:NASDAQ,
      RELIANCE:NSE, TCS:NSE, INFY:NSE,
      BTC-USD, ETH-USD

    Parameters
    ----------
    ticker : str
        The stock ticker in SYMBOL:EXCHANGE format.

    Returns
    -------
    str
        Formatted snapshot with price, movement, key stats, financials, and news.
    """
    ticker = ticker.strip()
    if not ticker:
        return "Error: no ticker provided."

    try:
        raw = fetch_stock_data(ticker)
    except MarketDataError as exc:
        return f"Error fetching stock data for '{ticker}': {exc}"

    return format_stock_data_as_text(raw)


# The list passed to llm.bind_tools() and the ReAct agent
ALL_TOOLS = [get_market_data, get_stock_price]
