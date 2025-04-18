import streamlit as st
import requests
import os
from datetime import datetime, timedelta
from utils.stock_data import is_crypto

@st.cache_data(ttl=3600)
def get_news(ticker: str, days: int = 7) -> list:
    """Fetch news articles for stocks and cryptocurrencies."""
    try:
        # For cryptocurrencies, use CoinMarketCap news API
        if is_crypto(ticker):
            headers = {
                'X-CMC_PRO_API_KEY': st.secrets["COINMARKETCAP_API_KEY"],
            }
            
            # Get crypto ID first
            symbol = ticker.split('-')[0]  # Remove -USD suffix
            response = requests.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                headers=headers,
                params={'symbol': symbol}
            )
            data = response.json()
            
            if 'data' in data and symbol in data['data']:
                crypto_id = data['data'][symbol]['id']
                
                # Fetch news using crypto ID
                news_response = requests.get(
                    f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/news",
                    headers=headers,
                    params={
                        'id': crypto_id,
                        'limit': 10
                    }
                )
                
                news_data = news_response.json()
                articles = []
                
                for article in news_data.get('data', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', ''),
                        'time_published': article.get('published_at', ''),
                        'sentiment': 0.5,  # Neutral default
                        'image_url': article.get('cover', '')
                    })
                return articles
        
        # For stocks, use Alpha Vantage News API
        api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        base_url = "https://www.alphavantage.co/query"
        
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
        for article in data["feed"][:10]:
            image_url = article.get('banner_image') or article.get('image', '')
            if image_url and (image_url.startswith('http://') or image_url.startswith('https://')):
                try:
                    response = requests.head(image_url, timeout=2)
                    if response.status_code != 200:
                        image_url = ''
                except:
                    image_url = ''
            else:
                image_url = ''
            
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
    # Handle None or invalid values
    if sentiment is None:
        return "Neutral", "gray"
    
    try:
        # Ensure sentiment is a float
        sentiment_val = float(sentiment)
        
        if sentiment_val >= 0.3:
            return "Positive", "green"
        elif sentiment_val <= -0.3:
            return "Negative", "red"
        else:
            return "Neutral", "gray"
    except (ValueError, TypeError):
        # Return neutral for any conversion errors
        return "Neutral", "gray"
