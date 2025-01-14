import requests
from textblob import TextBlob
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
import json

# Константы для API GNews
GNEWS_API_KEY = '6e9b533fb8e5c435201c7eda22d809ee'
GNEWS_API_URL = 'https://gnews.io/api/v4/search'

##### News Sentiment Agent #####
def news_sentiment_agent(state: AgentState):
    """Analyzes news sentiment and generates trading signals."""
    data = state["data"]
    ticker = data["ticker"]
    end_date = data["end_date"]

    # Получаем новости через API GNews
    news_articles = fetch_news(ticker, end_date)

    # Анализируем тональность новостей
    sentiment_scores = analyze_sentiment(news_articles)

    # Генерируем сигнал на основе тональности
    overall_sentiment = determine_overall_sentiment(sentiment_scores)
    signal, confidence = generate_signal(overall_sentiment)

    # Формируем сообщение с результатами
    reasoning = f"Overall sentiment: {overall_sentiment:.2f} (Bullish if > 0, Bearish if < 0, Neutral if = 0)"
    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    # Создаем сообщение для добавления в состояние
    message = HumanMessage(
        content=json.dumps(message_content),
        name="news_sentiment_agent",
    )

    # Отображаем рассуждения, если флаг установлен
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message_content, "News Sentiment Analysis Agent")

    # Добавляем сигнал в список analyst_signals
    state["data"]["analyst_signals"]["news_sentiment_agent"] = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    return {
        "messages": [message],
        "data": data,
    }


def fetch_news(ticker: str, end_date: str, limit: int = 10):
    """Fetch news articles related to the ticker using GNews API."""
    params = {
        "q": ticker,  # Поиск по тикеру
        "lang": "en",  # Язык новостей (английский)
        "max": limit,  # Максимальное количество новостей
        "sortby": "publishedAt",  # Сортировка по дате публикации
        "apikey": GNEWS_API_KEY,  # API ключ
    }
    response = requests.get(GNEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        print(f"Error fetching news: {response.status_code}")
        return []


def analyze_sentiment(news_articles: list):
    """Analyze sentiment of news articles using TextBlob."""
    sentiment_scores = []
    for article in news_articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = f"{title}. {description}"
        blob = TextBlob(content)
        sentiment_scores.append(blob.sentiment.polarity)  # Полярность тональности (-1 до 1)
    return sentiment_scores


def determine_overall_sentiment(sentiment_scores: list):
    """Determine the overall sentiment based on the average of sentiment scores."""
    if not sentiment_scores:
        return 0  # Нейтральный сигнал, если новостей нет
    return sum(sentiment_scores) / len(sentiment_scores)


def generate_signal(overall_sentiment: float):
    """Generate a trading signal based on the overall sentiment."""
    if overall_sentiment > 0.1:  # Бычий сигнал
        return "bullish", round(overall_sentiment * 100, 2)
    elif overall_sentiment < -0.1:  # Медвежий сигнал
        return "bearish", round(abs(overall_sentiment) * 100, 2)
    else:  # Нейтральный сигнал
        return "neutral", 50.0