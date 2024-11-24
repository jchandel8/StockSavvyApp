import yfinance as yf
import requests
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
            
        results = []
        
        # Search using yfinance
        tickers = yf.Tickers(query)
        for ticker in tickers.tickers:
            try:
                info = ticker.info
                if 'longName' in info:
                    results.append({
                        'symbol': info.get('symbol', ''),
                        'name': info.get('longName', ''),
                        'exchange': info.get('exchange', 'Unknown')
                    })
            except:
                continue
                
        # Also search Yahoo Finance API for company names
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {'q': query, 'quotesCount': 10, 'newsCount': 0}
            r = requests.get(url, params=params)
            data = r.json()
            
            for quote in data.get('quotes', []):
                results.append({
                    'symbol': quote.get('symbol', ''),
                    'name': quote.get('longname', quote.get('shortname', '')),
                    'exchange': quote.get('exchange', 'Unknown')
                })
        except:
            pass
            
        # Remove duplicates and sort by relevance
        unique_results = []
        seen = set()
        for item in results:
            if item['symbol'] not in seen:
                seen.add(item['symbol'])
                unique_results.append(item)
                
        return unique_results[:10]
    except Exception as e:
        st.error(f"Error searching stocks: {str(e)}")
        return []
