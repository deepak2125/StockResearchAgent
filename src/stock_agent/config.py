"""
Central configuration loaded from environment variables / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# SerpAPI
SERPAPI_API_KEY: str = os.environ.get("SERPAPI_API_KEY", "")
SERPAPI_BASE_URL: str = os.environ.get("SERPAPI_BASE_URL", "https://serpapi.com/search")

# Ollama
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Valid trend categories supported by the Google Finance Markets endpoint.
# Note: only "indexes" is currently supported by SerpAPI's google_finance_markets
# engine. The indexes response already includes US, Europe, Asia, currencies,
# crypto, and futures in one call.
MARKET_TRENDS = {
    "indexes": "Global Market Overview",
}

# Sub-categories present inside the "indexes" response — used for display only
MARKET_SECTIONS = {
    "us":         "🇺🇸 US Markets",
    "europe":     "🇪🇺 European Markets",
    "asia":       "🌏 Asian Markets",
    "currencies": "💱 Currencies",
    "crypto":     "₿ Crypto",
    "futures":    "📦 Futures",
}
