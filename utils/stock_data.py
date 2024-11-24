import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance with caching."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_stock_info(ticker: str) -> dict:
    """Get company information and fundamentals."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('forwardPE', 0),
            'eps': info.get('trailingEps', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'price_to_book': info.get('priceToBook', 0)
        }
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def search_stocks(query: str) -> list:
    """Search for stocks matching the query."""
    try:
        if not query or len(query) < 2:
            return []
            
        # Common stock exchanges to search
        exchanges = ['NASDAQ', 'NYSE']
        results = []
        
        for exchange in exchanges:
            try:
                # Search for stocks in each exchange
                stock = yf.Ticker(f"{query}.{exchange}")
                if 'longName' in stock.info:
                    results.append({
                        'symbol': f"{query}",
                        'name': stock.info['longName'],
                        'exchange': exchange
                    })
            except:
                continue
                
        # Also try searching without exchange
        try:
            stock = yf.Ticker(query)
            if 'longName' in stock.info:
                results.append({
                    'symbol': query,
                    'name': stock.info['longName'],
                    'exchange': stock.info.get('exchange', 'Unknown')
                })
        except:
            pass
            
        return results[:10]  # Limit to 10 suggestions
    except Exception as e:
        st.error(f"Error searching stocks: {str(e)}")
        return []
