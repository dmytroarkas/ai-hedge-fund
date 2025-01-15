import json
from pydantic import BaseModel, Field
from typing_extensions import Literal
from graph.state import AgentState, show_agent_reasoning

class PortfolioManagerOutput(BaseModel):
    action: Literal["buy", "sell", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")

def make_trading_decision(technical_signal, fundamentals_signal, sentiment_signal, valuation_signal, news_sentiment_signal, max_position_size, portfolio_cash, portfolio_stock):
    # Simple decision-making logic based on majority signals
    signals = {
        "buy": 0,
        "sell": 0,
        "hold": 0
    }

    # Count the signals
    if technical_signal.lower() in ["buy", "sell", "hold"]:
        signals[technical_signal.lower()] += 1
    if fundamentals_signal.lower() in ["buy", "sell", "hold"]:
        signals[fundamentals_signal.lower()] += 1
    if sentiment_signal.lower() in ["buy", "sell", "hold"]:
        signals[sentiment_signal.lower()] += 1
    if valuation_signal.lower() in ["buy", "sell", "hold"]:
        signals[valuation_signal.lower()] += 1
    if news_sentiment_signal.lower() in ["buy", "sell", "hold"]:
        signals[news_sentiment_signal.lower()] += 1

    # Determine the action based on the majority of signals
    action = max(signals, key=signals.get)
    
    # Determine quantity based on portfolio constraints
    if action == "buy":
        quantity = min(max_position_size, int(portfolio_cash))
    elif action == "sell":
        quantity = min(max_position_size, portfolio_stock)
    else:
        quantity = 0

    # Confidence is arbitrary in this simple model
    confidence = 80.0  # Example confidence value

    reasoning = f"Decision based on majority signals: {signals}"

    return PortfolioManagerOutput(
        action=action,
        quantity=quantity,
        confidence=confidence,
        reasoning=reasoning
    )

def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]

    # Make the trading decision
    result = make_trading_decision(
        technical_signal=analyst_signals.get("technical_analyst_agent", {}).get("signal", ""),
        fundamentals_signal=analyst_signals.get("fundamentals_agent", {}).get("signal", ""),
        sentiment_signal=analyst_signals.get("sentiment_agent", {}).get("signal", ""),
        valuation_signal=analyst_signals.get("valuation_agent", {}).get("signal", ""),
        news_sentiment_signal=analyst_signals.get("news_sentiment_agent", {}).get("signal", ""),
        max_position_size=analyst_signals.get("risk_management_agent", {}).get("max_position_size", 0),
        portfolio_cash=portfolio["cash"],
        portfolio_stock=portfolio["stock"]
    )

    message_content = {
        "action": result.action.lower(),
        "quantity": int(result.quantity),
        "confidence": float(result.confidence),
        "reasoning": result.reasoning,
    }

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message_content, "Portfolio Management Agent")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
