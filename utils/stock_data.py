import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

def is_crypto(symbol: str) -> bool:
    """Check if the symbol is a cryptocurrency."""
    return symbol.endswith('-USD') or symbol.endswith('USDT') or symbol.endswith('BTC')

@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock/crypto data from Yahoo Finance with caching."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_stock_info(ticker: str) -> dict:
    """Get stock/crypto information and fundamentals."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if is_crypto(ticker):
            return {
                'name': info.get('longName', ''),
                'type': 'crypto',
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
                'total_supply': info.get('totalSupply', 0),
                'max_supply': info.get('maxSupply', 0),
                'trading_pairs': len(info.get('tradingPairs', [])),
                'price_change_24h': info.get('priceChangePercent24h', 0)
            }
        else:
            return {
                'name': info.get('longName', ''),
                'type': 'stock',
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
    """Search for stocks and cryptocurrencies matching the query."""
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
                    symbol = info.get('symbol', '')
                    is_crypto_symbol = is_crypto(symbol)
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', ''),
                        'exchange': 'Crypto' if is_crypto_symbol else info.get('exchange', 'Unknown'),
                        'type': 'crypto' if is_crypto_symbol else 'stock',
                        'market_cap': info.get('marketCap', 0),
                        'volume_24h': info.get('volume24Hr', 0) if is_crypto_symbol else None
                    })
            except:
                continue
                
        # Also search Yahoo Finance API
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {'q': query, 'quotesCount': 20, 'newsCount': 0}
            r = requests.get(url, params=params)
            data = r.json()
            
            for quote in data.get('quotes', []):
                symbol = quote.get('symbol', '')
                is_crypto_symbol = is_crypto(symbol)
                if is_crypto_symbol or not any(r['symbol'] == symbol for r in results):
                    results.append({
                        'symbol': symbol,
                        'name': quote.get('longname', quote.get('shortname', '')),
                        'exchange': 'Crypto' if is_crypto_symbol else quote.get('exchange', 'Unknown'),
                        'type': 'crypto' if is_crypto_symbol else 'stock',
                        'market_cap': quote.get('marketCap', 0),
                        'volume_24h': quote.get('volume24Hr', 0) if is_crypto_symbol else None
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
        
        # Sort results to show cryptocurrencies first when the query looks like a crypto symbol
        if query.upper().endswith(('BTC', 'ETH', 'USD', 'USDT')):
            unique_results.sort(key=lambda x: 0 if x['type'] == 'crypto' else 1)
                
        return unique_results[:10]
    except Exception as e:
        st.error(f"Error searching stocks/crypto: {str(e)}")
        return []
