from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import math


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState):
    """
    Analyzes stocks using George Soros's theory of reflexivity:
    1. Focus on the feedback loop between market perceptions and reality.
    2. Identify potential market imbalances and bubbles.
    3. Evaluate sentiment and momentum in addition to fundamentals.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "total_assets",
                "total_liabilities",
                "current_assets",
                "current_liabilities",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=10,
        )

        progress.update_status("ben_graham_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        # Perform sub-analyses based on Soros's reflexivity theory
        progress.update_status("ben_graham_agent", ticker, "Analyzing market sentiment")
        sentiment_analysis = analyze_market_sentiment(metrics, financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing reflexivity feedback loops")
        reflexivity_analysis = analyze_reflexivity(metrics, financial_line_items, market_cap)

        progress.update_status("ben_graham_agent", ticker, "Evaluating potential market imbalances")
        imbalance_analysis = evaluate_market_imbalances(metrics, financial_line_items, market_cap)

        # Aggregate scoring
        total_score = sentiment_analysis["score"] + reflexivity_analysis["score"] + imbalance_analysis["score"]
        max_possible_score = 15  # Adjust as needed

        # Map total_score to signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "sentiment_analysis": sentiment_analysis,
            "reflexivity_analysis": reflexivity_analysis,
            "imbalance_analysis": imbalance_analysis,
        }

        progress.update_status("ben_graham_agent", ticker, "Generating Soros-style analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {
            "signal": graham_output.signal,
            "confidence": graham_output.confidence,
            "reasoning": graham_output.reasoning,
        }

        progress.update_status("ben_graham_agent", ticker, "Done")

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_market_sentiment(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze market sentiment and momentum, which are key to Soros's reflexivity theory.
    """
    score = 0
    details = []

    # Example: Use revenue growth as a proxy for sentiment
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        growth_rate = (revenues[-1] - revenues[0]) / abs(revenues[0])
        if growth_rate > 0.2:
            score += 3
            details.append(f"Strong revenue growth: {(growth_rate * 100):.1f}%")
        elif growth_rate > 0.1:
            score += 2
            details.append(f"Moderate revenue growth: {(growth_rate * 100):.1f}%")
        else:
            details.append(f"Low revenue growth: {(growth_rate * 100):.1f}%")
    else:
        details.append("Insufficient revenue data for sentiment analysis")

    return {"score": score, "details": "; ".join(details)}


def analyze_reflexivity(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """
    Analyze the feedback loop between market perceptions and reality.
    """
    score = 0
    details = []

    # Example: Compare market cap to book value
    latest = financial_line_items[-1]
    book_value = latest.total_assets - latest.total_liabilities
    if book_value > 0 and market_cap > 0:
        price_to_book = market_cap / book_value
        if price_to_book < 1.0:
            score += 3
            details.append(f"Undervalued: Price-to-Book = {price_to_book:.2f}")
        elif price_to_book < 2.0:
            score += 2
            details.append(f"Fairly valued: Price-to-Book = {price_to_book:.2f}")
        else:
            details.append(f"Overvalued: Price-to-Book = {price_to_book:.2f}")
    else:
        details.append("Insufficient data for reflexivity analysis")

    return {"score": score, "details": "; ".join(details)}


def evaluate_market_imbalances(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """
    Evaluate potential market imbalances or bubbles.
    """
    score = 0
    details = []

    # Example: Check for excessive leverage
    latest = financial_line_items[-1]
    debt_to_equity = latest.total_liabilities / (latest.total_assets - latest.total_liabilities)
    if debt_to_equity > 1.0:
        score += 3
        details.append(f"High leverage: Debt-to-Equity = {debt_to_equity:.2f}")
    elif debt_to_equity > 0.5:
        score += 2
        details.append(f"Moderate leverage: Debt-to-Equity = {debt_to_equity:.2f}")
    else:
        details.append(f"Low leverage: Debt-to-Equity = {debt_to_equity:.2f}")

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of George Soros.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a George Soros AI agent, making investment decisions using his principles:
            1. Focus on the feedback loop between market perceptions and reality.
            2. Identify potential market imbalances and bubbles.
            3. Evaluate sentiment and momentum in addition to fundamentals.
            4. Be prepared to act quickly when market conditions change.
            5. Avoid over-reliance on traditional valuation metrics.

            Rules:
            - Look for signs of reflexivity in market behavior.
            - Identify overvalued or undervalued assets based on market sentiment.
            - Consider macroeconomic factors and market psychology.
            - Provide a rational, data-driven recommendation (bullish, bearish, or neutral)."""
        ),
        (
            "human",
            """Based on the following analysis, create a Soros-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}"""
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )
