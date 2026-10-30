"""
Gradio 6 UI for the Stock Research Agent.
Run with: uv run python -m stock_agent

Tabs:
  1. Market Overview  — global indexes, crypto, currencies, futures
  2. Stock Lookup     — individual company / ticker analysis
"""
import gradio as gr

from .agent import run_agent, run_stock_agent
from .config import MARKET_SECTIONS, OLLAMA_MODEL
from .llm import get_available_models


# ── helpers ───────────────────────────────────────────────────────────────────

FOCUS_CHOICES = ["All regions"] + list(MARKET_SECTIONS.values())


def _model_choices() -> list[str]:
    models = get_available_models()
    return models if models else [OLLAMA_MODEL]


# ── Tab 1 callback — market overview ─────────────────────────────────────────

def chat_stream(
    user_message: str,
    history: list[dict],
    focus: str,
    model: str,
):
    """Stream global market overview chat."""
    if not user_message.strip():
        yield history, ""
        return

    focused_question = (
        user_message
        if focus == "All regions"
        else f"{user_message} (Focus on: {focus})"
    )

    history = [*history, {"role": "user", "content": user_message}]
    history = [*history, {"role": "assistant", "content": ""}]

    accumulated = ""
    for chunk in run_agent(user_question=focused_question, trend="indexes", model=model):
        accumulated += chunk
        history = [*history[:-1], {"role": "assistant", "content": accumulated}]
        yield history, ""

    yield history, ""


# ── Tab 2 callback — individual stock lookup ──────────────────────────────────

def stock_chat_stream(
    user_message: str,
    ticker: str,
    history: list[dict],
    model: str,
):
    """Stream individual stock analysis chat."""
    if not user_message.strip() and not ticker.strip():
        yield history, ""
        return

    # If the user didn't type a question, default to a general analysis request
    question = user_message.strip() or f"Give me a full analysis of {ticker.strip()}."

    display = f"[{ticker.upper()}] {user_message}" if user_message.strip() else f"Analyse {ticker.upper()}"
    history = [*history, {"role": "user", "content": display}]
    history = [*history, {"role": "assistant", "content": ""}]

    accumulated = ""
    for chunk in run_stock_agent(ticker=ticker, user_question=question, model=model):
        accumulated += chunk
        history = [*history[:-1], {"role": "assistant", "content": accumulated}]
        yield history, ""

    yield history, ""


# ── layout ────────────────────────────────────────────────────────────────────

CSS = """
#header { text-align: center; padding: 10px 0 4px; }
#header h1 { font-size: 2rem; }
#header p  { color: #6b7280; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Finley — Financial Advisor") as demo:

        # ── header ────────────────────────────────────────────────────────
        with gr.Column(elem_id="header"):
            gr.HTML(
                "<h1>📈 Finley — Your Financial Advisor</h1>"
                "<p>Powered by Ollama (local LLM) &nbsp;·&nbsp; Live data via Google Finance / SerpAPI</p>"
            )

        # ── shared model selector (sits above tabs) ───────────────────────
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=_model_choices(),
                value=_model_choices()[0],
                label="Ollama Model",
                info="Local model to use for all analysis",
                scale=4,
                allow_custom_value=True,
            )
            refresh_models_btn = gr.Button("↻ Refresh Models", scale=1, size="sm")

        # ══ Tabs ══════════════════════════════════════════════════════════
        with gr.Tabs():

            # ── Tab 1: Market Overview ─────────────────────────────────
            with gr.TabItem("🌍 Market Overview"):
                focus_dropdown = gr.Dropdown(
                    choices=FOCUS_CHOICES,
                    value="All regions",
                    label="Focus Area",
                    info="Steer the LLM towards a specific region",
                )

                market_chatbot = gr.Chatbot(
                    label="Market Research Chat",
                    height=480,
                    render_markdown=True,
                    sanitize_html=False,
                )

                with gr.Row():
                    market_msg = gr.Textbox(
                        placeholder="e.g. 'How are global markets doing today?'",
                        label="Your question",
                        lines=2,
                        scale=5,
                    )
                    market_send = gr.Button("Send ➤", variant="primary", scale=1)

                market_clear = gr.Button("🗑  Clear", size="sm")

                gr.Examples(
                    examples=[
                        ["Give me a full global market overview.",      "All regions"],
                        ["How are US markets performing today?",         "🇺🇸 US Markets"],
                        ["What's happening in European markets?",        "🇪🇺 European Markets"],
                        ["Give me a quick crypto market overview.",      "₿ Crypto"],
                        ["How are currency pairs moving today?",         "💱 Currencies"],
                        ["What are futures telling us about tomorrow?",  "📦 Futures"],
                    ],
                    inputs=[market_msg, focus_dropdown],
                    label="Quick-start examples",
                )

            # ── Tab 2: Stock Lookup ────────────────────────────────────
            with gr.TabItem("🔎 Stock Lookup"):
                gr.Markdown(
                    "Enter a ticker in **SYMBOL:EXCHANGE** format. "
                    "Examples: `AAPL:NASDAQ` · `RELIANCE:NSE` · `TSLA:NASDAQ` · `TCS:NSE` · `GOOGL:NASDAQ`"
                )

                ticker_box = gr.Textbox(
                    placeholder="e.g. AAPL:NASDAQ  or  RELIANCE:NSE",
                    label="Stock Ticker",
                    lines=1,
                    scale=1,
                )

                stock_chatbot = gr.Chatbot(
                    label="Stock Analysis Chat",
                    height=440,
                    render_markdown=True,
                    sanitize_html=False,
                )

                with gr.Row():
                    stock_msg = gr.Textbox(
                        placeholder="e.g. 'What are the key risks?' or leave blank for a full analysis",
                        label="Your question (optional)",
                        lines=2,
                        scale=5,
                    )
                    stock_send = gr.Button("Analyse ➤", variant="primary", scale=1)

                stock_clear = gr.Button("🗑  Clear", size="sm")

                gr.Examples(
                    examples=[
                        ["Is Apple a good buy right now?",                  "AAPL:NASDAQ"],
                        ["Give me a full fundamental analysis.",             "RELIANCE:NSE"],
                        ["What are the key risks for Tesla?",               "TSLA:NASDAQ"],
                        ["How has TCS been performing financially?",        "TCS:NSE"],
                        ["Summarise the latest news and financials.",       "GOOGL:NASDAQ"],
                        ["What's the current Bitcoin sentiment?",           "BTC-USD"],
                    ],
                    inputs=[stock_msg, ticker_box],
                    label="Quick-start examples",
                )

        # ── wiring ────────────────────────────────────────────────────────

        # Tab 1
        market_submit = dict(
            fn=chat_stream,
            inputs=[market_msg, market_chatbot, focus_dropdown, model_dropdown],
            outputs=[market_chatbot, market_msg],
        )
        market_send.click(**market_submit)
        market_msg.submit(**market_submit)
        market_clear.click(fn=lambda: ([], ""), outputs=[market_chatbot, market_msg])

        # Tab 2
        stock_submit = dict(
            fn=stock_chat_stream,
            inputs=[stock_msg, ticker_box, stock_chatbot, model_dropdown],
            outputs=[stock_chatbot, stock_msg],
        )
        stock_send.click(**stock_submit)
        stock_msg.submit(**stock_submit)
        ticker_box.submit(**stock_submit)
        stock_clear.click(fn=lambda: ([], "", ""), outputs=[stock_chatbot, stock_msg, ticker_box])

        # Shared
        refresh_models_btn.click(
            fn=lambda: gr.update(choices=_model_choices()),
            outputs=model_dropdown,
        )

    return demo


# ── entry-point ───────────────────────────────────────────────────────────────

def main() -> None:
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
