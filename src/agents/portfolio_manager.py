import json
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning

class PortfolioManagerOutput(BaseModel):
    action: Literal["buy", "sell", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")

def make_trading_decision(
    technical_signal, technical_confidence,
    fundamentals_signal, fundamentals_confidence,
    sentiment_signal, sentiment_confidence,
    valuation_signal, valuation_confidence,
    news_sentiment_signal, news_sentiment_confidence,
    reflexivity_signal, reflexivity_confidence,  # Новый параметр
    max_position_size, portfolio_cash, portfolio_stock
):
    # Преобразование сигналов в "buy", "sell" или "hold"
    def convert_signal(signal):
        signal = signal.lower()
        if signal == "bullish":
            return "buy"
        elif signal == "bearish":
            return "sell"
        elif signal == "neutral":
            return "hold"
        return "hold"  # По умолчанию, если сигнал не распознан

    # Взвешенный подсчет сигналов и сбор Confidence для каждого решения
    weighted_signals = {
        "buy": {"total_weight": 0.0, "confidences": []},
        "sell": {"total_weight": 0.0, "confidences": []},
        "hold": {"total_weight": 0.0, "confidences": []},
    }

    # Преобразование и взвешенный подсчет каждого сигнала
    def process_signal(signal, confidence):
        action = convert_signal(signal)
        weighted_signals[action]["total_weight"] += confidence
        weighted_signals[action]["confidences"].append(confidence)

    process_signal(technical_signal, technical_confidence)
    process_signal(fundamentals_signal, fundamentals_confidence)
    process_signal(sentiment_signal, sentiment_confidence)
    process_signal(valuation_signal, valuation_confidence)
    process_signal(news_sentiment_signal, news_sentiment_confidence)
    process_signal(reflexivity_signal, reflexivity_confidence)  # Новый сигнал

    # Определение действия на основе наибольшего веса
    action = max(weighted_signals, key=lambda k: weighted_signals[k]["total_weight"])
    
    # Определение количества на основе ограничений портфеля
    if action == "buy":
        quantity = min(max_position_size, int(portfolio_cash))
    elif action == "sell":
        quantity = min(max_position_size, portfolio_stock)
    else:
        quantity = 0

    # Расчет усредненной уверенности для каждого решения
    avg_confidence = {}
    for decision, data in weighted_signals.items():
        if data["confidences"]:
            avg_confidence[decision] = sum(data["confidences"]) / len(data["confidences"])
        else:
            avg_confidence[decision] = 0.0

    # Итоговая уверенность для выбранного действия
    confidence = avg_confidence[action]

    # Формирование Reasoning с усредненной уверенностью для каждого решения
    reasoning = f"Decision based on weighted signals with average confidence: {avg_confidence}"

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
        technical_confidence=analyst_signals.get("technical_analyst_agent", {}).get("confidence", 0.0),
        fundamentals_signal=analyst_signals.get("fundamentals_agent", {}).get("signal", ""),
        fundamentals_confidence=analyst_signals.get("fundamentals_agent", {}).get("confidence", 0.0),
        sentiment_signal=analyst_signals.get("sentiment_agent", {}).get("signal", ""),
        sentiment_confidence=analyst_signals.get("sentiment_agent", {}).get("confidence", 0.0),
        valuation_signal=analyst_signals.get("valuation_agent", {}).get("signal", ""),
        valuation_confidence=analyst_signals.get("valuation_agent", {}).get("confidence", 0.0),
        news_sentiment_signal=analyst_signals.get("news_sentiment_agent", {}).get("signal", ""),
        news_sentiment_confidence=analyst_signals.get("news_sentiment_agent", {}).get("confidence", 0.0),
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
