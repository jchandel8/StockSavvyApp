import streamlit as st
import requests
import os
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)
def get_news(ticker: str, days: int = 7) -> list:
    """Fetch news articles for a given stock ticker."""
    # Using Alpha Vantage News API (free tier)
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
    base_url = "https://www.alphavantage.co/query"
    
    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key
        }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "feed" not in data:
            return []
        
        articles = []
        for article in data["feed"][:10]:  # Limit to 10 articles
            # Extract image URL from the article
            image_url = article.get('banner_image', '') or article.get('image', '')
            
            articles.append({
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'time_published': article.get('time_published', ''),
                'sentiment': article.get('overall_sentiment_score', 0),
                'image_url': image_url
            })
        
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def format_news_sentiment(sentiment: float) -> tuple:
    """Format news sentiment score with color coding."""
    if sentiment >= 0.3:
        return "Positive", "green"
    elif sentiment <= -0.3:
        return "Negative", "red"
    else:
        return "Neutral", "gray"
