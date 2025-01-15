from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
import json
import pandas as pd
from tools.api import get_prices, get_financial_metrics, prices_to_df

##### Reflexivity Agent #####
def reflexivity_agent(state: AgentState):
    """Analyzes the interaction between market prices, fundamentals, and sentiment to detect reflexive cycles."""
    data = state["data"]
    ticker = data["ticker"]
    start_date = data["start_date"]
    end_date = data["end_date"]

    # Fetch price data
    prices = get_prices(ticker, start_date, end_date)
    prices_df = prices_to_df(prices)

    # Fetch fundamental metrics
    financial_metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)
    metrics = financial_metrics[0]

    # Fetch sentiment signals from other agents
    sentiment_signal = state["data"]["analyst_signals"].get("sentiment_agent", {}).get("signal", "neutral")
    news_sentiment_signal = state["data"]["analyst_signals"].get("news_sentiment_agent", {}).get("signal", "neutral")

    # Combine sentiment signals
    sentiment_score = 0
    if sentiment_signal == "bullish":
        sentiment_score += 1
    elif sentiment_signal == "bearish":
        sentiment_score -= 1
    if news_sentiment_signal == "bullish":
        sentiment_score += 1
    elif news_sentiment_signal == "bearish":
        sentiment_score -= 1

    # Analyze price trends
    price_change = (prices_df["close"].iloc[-1] - prices_df["close"].iloc[0]) / prices_df["close"].iloc[0]
    volatility = prices_df["close"].pct_change().std()

    # Analyze fundamentals
    earnings_growth = metrics.earnings_growth or 0
    revenue_growth = metrics.revenue_growth or 0
    pe_ratio = metrics.price_to_earnings_ratio or 0

    # Reflexivity analysis
    if price_change > 0.2 and earnings_growth < price_change and sentiment_score > 0:
        signal = "bearish"  # Possible overvaluation due to reflexive cycle
        reasoning = "Price growth exceeds earnings growth, and sentiment is overly optimistic. Risk of a reflexive bubble."
    elif price_change < -0.2 and earnings_growth > abs(price_change) and sentiment_score < 0:
        signal = "bullish"  # Possible undervaluation due to reflexive cycle
        reasoning = "Price decline exceeds earnings decline, and sentiment is overly pessimistic. Potential buying opportunity."
    else:
        signal = "neutral"
        reasoning = "No significant reflexive cycle detected."

    # Confidence calculation
    confidence = min(abs(price_change) * 100, 100)
    confidence = round(confidence, 2)  # Округляем до 2 знаков после запятой

    # Create the reflexivity message
    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="reflexivity_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message_content, "Reflexivity Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["reflexivity_agent"] = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    return {
        "messages": [message],
        "data": data,
    }
