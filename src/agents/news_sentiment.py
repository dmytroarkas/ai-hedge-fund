import requests
from textblob import TextBlob
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants for GNews API
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")  # Get API key from .env
GNEWS_API_URL = 'https://gnews.io/api/v4/search'

##### News Sentiment Agent #####
def news_sentiment_agent(state: AgentState):
    """Analyzes news sentiment and generates trading signals."""
    data = state["data"]
    ticker = data["ticker"]
    start_date = data.get("start_date")  # Get start_date from state
    end_date = data["end_date"]

    # Fetch news articles using GNews API
    news_articles = fetch_news(ticker, start_date, end_date)

    # Log the list of news articles
    print("\n=== Fetched News ===")
    for article in news_articles:
        title = article.get("title", "No title")
        published_at = article.get("publishedAt", "No date")
        print(f"Date: {published_at}, Title: {title}")

    # Analyze sentiment of news articles
    sentiment_scores = analyze_sentiment(news_articles)

    # Generate signal based on sentiment
    overall_sentiment = determine_overall_sentiment(sentiment_scores)
    signal, confidence = generate_signal(overall_sentiment)

    # Prepare the message with results
    reasoning = f"Overall sentiment: {overall_sentiment:.2f} (Bullish if > 0, Bearish if < 0, Neutral if = 0)"
    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    # Create the message to add to the state
    message = HumanMessage(
        content=json.dumps(message_content),
        name="news_sentiment_agent",
    )

    # Display reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message_content, "News Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["news_sentiment_agent"] = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    return {
        "messages": [message],
        "data": data,
    }


def fetch_news(ticker: str, start_date: str, end_date: str, limit: int = 100):
    """Fetch news articles related to the ticker using GNews API.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for news articles in 'YYYY-MM-DD' format.
        end_date (str): End date for news articles in 'YYYY-MM-DD' format.
        limit (int): Maximum number of news articles to fetch.

    Returns:
        list: List of news articles.

    Raises:
        ValueError: If the API key is missing or invalid.
        Exception: If there is an error fetching news from the API.
    """
    # Check if the API key is set
    if not GNEWS_API_KEY:
        raise ValueError(
            "GNEWS_API_KEY is missing. Please ensure it is set in the .env file. "
            "You can obtain a key from https://gnews.io/."
        )

    # Convert end_date to datetime object
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    # If start_date is not provided, default to 3 months before end_date
    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_date_obj = end_date_obj - timedelta(days=90)  # Default to 90 days (~3 months)
    
    # Format dates for API request
    start_date_formatted = start_date_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date_formatted = end_date_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Prepare API request parameters
    params = {
        "q": ticker,  # Search by ticker
        "lang": "en",  # Language of news (English)
        "max": limit,  # Maximum number of news articles
        "sortby": "publishedAt",  # Sort by publication date
        "from": start_date_formatted,  # Start date
        "to": end_date_formatted,  # End date
        "apikey": GNEWS_API_KEY,  # API key
    }
    
    try:
        # Send request to GNews API
        response = requests.get(GNEWS_API_URL, params=params)
        
        # Check for API errors
        if response.status_code == 400:
            raise ValueError(
                "Invalid API request. This could be due to an invalid API key, "
                "incorrect date range, or other parameters. Please check your .env file "
                "and ensure the GNEWS_API_KEY is correct and up-to-date."
            )
        elif response.status_code == 401:
            raise ValueError(
                "Unauthorized access. The GNEWS_API_KEY is either missing or invalid. "
                "Please verify your API key at https://gnews.io/ and update the .env file."
            )
        elif response.status_code == 403:
            raise ValueError(
                "Access forbidden. Your API key may have reached its usage limit or "
                "is restricted. Check your GNews account for details."
            )
        elif response.status_code != 200:
            raise Exception(
                f"Error fetching news: {response.status_code}. "
                f"Response: {response.text}"
            )
        
        # Return the list of articles
        return response.json().get("articles", [])
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to GNews API: {e}")


def analyze_sentiment(news_articles: list):
    """Analyze sentiment of news articles using TextBlob."""
    sentiment_scores = []
    for article in news_articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = f"{title}. {description}"
        blob = TextBlob(content)
        sentiment_scores.append(blob.sentiment.polarity)  # Sentiment polarity (-1 to 1)
    return sentiment_scores


def determine_overall_sentiment(sentiment_scores: list):
    """Determine the overall sentiment based on the average of sentiment scores."""
    if not sentiment_scores:
        return 0  # Neutral signal if no news
    return sum(sentiment_scores) / len(sentiment_scores)


def generate_signal(overall_sentiment: float):
    """Generate a trading signal based on the overall sentiment."""
    if overall_sentiment > 0.1:  # Bullish signal
        return "bullish", round(overall_sentiment * 100, 2)
    elif overall_sentiment < -0.1:  # Bearish signal
        return "bearish", round(abs(overall_sentiment) * 100, 2)
    else:  # Neutral signal
        return "neutral", 50.0
