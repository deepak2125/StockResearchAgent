"""
SerpAPI Google Finance clients.

Two endpoints:
  - google_finance_markets  →  global market overview  (fetch_market_data)
  - google_finance          →  individual stock quote   (fetch_stock_data)

Docs:
  https://serpapi.com/google-finance-markets
  https://serpapi.com/google-finance-api
"""
import httpx
from typing import Any

from .config import SERPAPI_API_KEY, SERPAPI_BASE_URL, MARKET_SECTIONS


class MarketDataError(Exception):
    """Raised when a SerpAPI request fails."""


# ── Global market overview ────────────────────────────────────────────────────

def fetch_market_data(trend: str = "indexes", language: str = "en") -> dict[str, Any]:
    """
    Fetch global market data via the google_finance_markets engine.

    Parameters
    ----------
    trend : str
        Currently only "indexes" is supported by SerpAPI.
    language : str
        BCP-47 language code (default: "en").
    """
    if not SERPAPI_API_KEY:
        raise MarketDataError("SERPAPI_API_KEY is not set. Add it to your .env file.")

    params = {
        "engine": "google_finance_markets",
        "trend": trend,
        "hl": language,
        "api_key": SERPAPI_API_KEY,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(SERPAPI_BASE_URL, params=params)

    if response.status_code != 200:
        raise MarketDataError(
            f"SerpAPI returned HTTP {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    if "error" in data:
        raise MarketDataError(f"SerpAPI error: {data['error']}")

    return data


def format_market_data_as_text(data: dict[str, Any]) -> str:
    """Convert raw google_finance_markets JSON to a plain-text table for the LLM."""
    markets: dict[str, list] = data.get("markets", {})
    if not markets:
        return "No market data returned."

    lines: list[str] = []
    for region, stocks in markets.items():
        if not isinstance(stocks, list):
            continue
        label = MARKET_SECTIONS.get(region, region.upper())
        lines.append(f"\n=== {label} ===")
        for item in stocks:
            name = item.get("name", "N/A")
            price = item.get("price", 0) or 0
            movement = item.get("price_movement", {}) or {}
            direction = movement.get("movement", "")
            pct = movement.get("percentage", 0.0) or 0.0
            value = movement.get("value", 0.0) or 0.0

            arrow = "▲" if direction == "Up" else "▼" if direction == "Down" else "─"
            sign = "+" if direction == "Up" else "-"
            try:
                lines.append(
                    f"  {arrow} {name:<22} {float(price):>12,.2f}   "
                    f"({sign}{abs(pct):.2f}%  {sign}{abs(value):.2f})"
                )
            except (TypeError, ValueError):
                lines.append(f"  {arrow} {name:<22} {price}")

    return "\n".join(lines)


# ── Individual stock quote ────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, language: str = "en") -> dict[str, Any]:
    """
    Fetch a single stock quote via the google_finance engine.

    Parameters
    ----------
    ticker : str
        Google Finance ticker, e.g. "RELIANCE:NSE", "GOOGL:NASDAQ", "AAPL:NASDAQ".
        The user can type just "RELIANCE NSE" or "AAPL NASDAQ" — we normalise it.
    language : str
        BCP-47 language code (default: "en").

    Returns
    -------
    dict
        Parsed JSON response from SerpAPI containing summary, knowledge_graph,
        financials, news_results, graph (price history).
    """
    if not SERPAPI_API_KEY:
        raise MarketDataError("SERPAPI_API_KEY is not set. Add it to your .env file.")

    # Normalise ticker: "reliance nse" → "RELIANCE:NSE", "AAPL NASDAQ" → "AAPL:NASDAQ"
    # Also accept already-formatted "RELIANCE:NSE"
    q = ticker.strip().upper().replace(" ", ":")
    # Collapse multiple colons
    while "::" in q:
        q = q.replace("::", ":")

    params = {
        "engine": "google_finance",
        "q": q,
        "hl": language,
        "api_key": SERPAPI_API_KEY,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(SERPAPI_BASE_URL, params=params)

    if response.status_code != 200:
        raise MarketDataError(
            f"SerpAPI returned HTTP {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    if "error" in data:
        raise MarketDataError(f"SerpAPI error: {data['error']}")

    return data


def format_stock_data_as_text(data: dict[str, Any]) -> str:
    """
    Convert a google_finance stock response to a rich plain-text summary
    suitable for feeding to the LLM.

    Includes: price/movement, key stats, about, latest financials, recent news.
    """
    lines: list[str] = []

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = data.get("summary", {})
    if summary:
        title    = summary.get("title", "N/A")
        stock    = summary.get("stock", "")
        exchange = summary.get("exchange", "")
        price    = summary.get("price", "N/A")
        currency = summary.get("currency", "")
        mv       = summary.get("price_movement", {}) or {}
        direction = mv.get("movement", "")
        pct      = mv.get("percentage", 0.0) or 0.0
        val      = mv.get("value", 0.0) or 0.0
        arrow    = "▲" if direction == "Up" else "▼" if direction == "Down" else "─"
        sign     = "+" if direction == "Up" else "-"
        date_str = summary.get("date", "")

        lines.append(f"=== {title} ({stock}:{exchange}) ===")
        lines.append(f"  Price : {price} {currency}  {arrow} {sign}{abs(pct):.2f}%  ({sign}{abs(val):.2f})")
        if date_str:
            lines.append(f"  As of : {date_str}")

        # After-hours / pre-market
        market = summary.get("market", {})
        if market:
            m_label = market.get("trading", "Extended Hours")
            m_price = market.get("price", "")
            m_mv    = market.get("price_movement", {}) or {}
            m_dir   = m_mv.get("movement", "")
            m_pct   = m_mv.get("percentage", 0.0) or 0.0
            m_arrow = "▲" if m_dir == "Up" else "▼" if m_dir == "Down" else "─"
            m_sign  = "+" if m_dir == "Up" else "-"
            lines.append(f"  {m_label}: {m_price}  {m_arrow} {m_sign}{abs(m_pct):.2f}%")

    # ── Key stats ─────────────────────────────────────────────────────────────
    kg = data.get("knowledge_graph", {})
    key_stats = kg.get("key_stats", {})
    stats = key_stats.get("stats", [])
    if stats:
        lines.append("\n--- Key Stats ---")
        for s in stats:
            lines.append(f"  {s.get('label',''):<28} {s.get('value','')}")

    # ── About ─────────────────────────────────────────────────────────────────
    about_list = kg.get("about", [])
    for about_block in about_list:
        desc = about_block.get("description", {})
        snippet = desc.get("snippet", "") if isinstance(desc, dict) else str(desc)
        if snippet:
            # Trim to first 400 chars to avoid overloading the LLM context
            lines.append(f"\n--- About ---\n  {snippet[:400]}{'…' if len(snippet) > 400 else ''}")
        info_items = about_block.get("info", [])
        if info_items:
            for item in info_items[:6]:  # top 6 facts
                lines.append(f"  {item.get('label',''):<20} {item.get('value','')}")
        break  # only first about block

    # ── Latest financials (most recent period only) ───────────────────────────
    financials = data.get("financials", [])
    if financials:
        lines.append("\n--- Financials (latest period) ---")
        for statement in financials:
            title = statement.get("title", "")
            results = statement.get("results", [])
            if results:
                latest = results[0]  # most recent quarter
                period = f"{latest.get('date','')} ({latest.get('period_type','')})"
                lines.append(f"\n  {title} — {period}")
                for row in latest.get("table", [])[:5]:  # top 5 line items
                    lbl = row.get("title", "")
                    raw_val = row.get("value", "")
                    change  = row.get("change", "")
                    # Format large numbers
                    try:
                        num = int(raw_val)
                        if abs(num) >= 1_000_000_000:
                            fmt = f"${num/1_000_000_000:.2f}B"
                        elif abs(num) >= 1_000_000:
                            fmt = f"${num/1_000_000:.2f}M"
                        else:
                            fmt = f"${num:,}"
                    except (ValueError, TypeError):
                        fmt = raw_val
                    lines.append(f"    {lbl:<35} {fmt:<14} {change}")

    # ── Recent news ───────────────────────────────────────────────────────────
    news_results = data.get("news_results", [])
    news_items: list[dict] = []
    for block in news_results:
        if isinstance(block, dict):
            if "items" in block:
                news_items.extend(block["items"])
            elif "snippet" in block:
                news_items.append(block)
    if news_items:
        lines.append("\n--- Recent News ---")
        for item in news_items[:5]:
            snippet = item.get("snippet", item.get("title", ""))
            source  = item.get("source", "")
            date    = item.get("date", "")
            lines.append(f"  • [{source} | {date}] {snippet}")

    return "\n".join(lines) if lines else "No stock data found."
