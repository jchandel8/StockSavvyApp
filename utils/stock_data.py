import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

def is_crypto(symbol: str) -> bool:
    """Check if the symbol is a cryptocurrency."""
    crypto_suffixes = ['-USD', 'USD', 'USDT', 'BTC']
    crypto_prefixes = ['BTC', 'ETH', 'XRP', 'DOGE', 'ADA']
    
    # Clean the symbol
    symbol = symbol.upper().strip()
    
    # Check if it's a known crypto by prefix
    if any(symbol.startswith(prefix) for prefix in crypto_prefixes):
        return True
    
    # Check for crypto suffixes
    if any(symbol.endswith(suffix) for suffix in crypto_suffixes):
        return True
        
    return False

def format_crypto_symbol(symbol: str) -> str:
    """Format cryptocurrency symbol to ensure proper data fetching."""
    # Common mappings for crypto symbols
    crypto_mappings = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'BITCOIN': 'BTC-USD',
        'ETHEREUM': 'ETH-USD',
    }
    
    # Check if we have a direct mapping
    symbol = symbol.upper().strip()
    if symbol in crypto_mappings:
        return crypto_mappings[symbol]
    
    # Remove any existing suffixes
    base_symbol = symbol
    for suffix in ['-USD', 'USD', 'USDT', 'BTC']:
        if base_symbol.endswith(suffix):
            base_symbol = base_symbol[:-len(suffix)]
            break
    
    # Add -USD suffix if not present
    if not base_symbol.endswith('-USD'):
        return f"{base_symbol}-USD"
    return base_symbol

@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock/crypto data from Yahoo Finance with caching."""
    try:
        stock = yf.Ticker(ticker)
        if is_crypto(ticker):
            df = stock.history(period='1mo')
        else:
            df = stock.history(period=period)
            
        if not df.empty:
            from utils.technical_analysis import calculate_technical_indicators
            df = calculate_technical_indicators(df)
            
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_coinmarketcap_data(symbol: str) -> dict:
    """Fetch cryptocurrency data from CoinMarketCap API."""
    headers = {
        'X-CMC_PRO_API_KEY': st.secrets["COINMARKETCAP_API_KEY"],
    }
    base_url = "https://pro-api.coinmarketcap.com/v1"
    
    try:
        # Get cryptocurrency data
        response = requests.get(
            f"{base_url}/cryptocurrency/quotes/latest",
            headers=headers,
            params={'symbol': symbol.split('-')[0]}  # Remove -USD suffix
        )
        data = response.json()
        if 'data' in data and symbol.split('-')[0] in data['data']:
            crypto_data = data['data'][symbol.split('-')[0]]
            market_data = crypto_data['quote']['USD']
            return {
                'name': crypto_data['name'],
                'type': 'crypto',
                'market_cap': market_data['market_cap'],
                'volume_24h': market_data['volume_24h'],
                'circulating_supply': crypto_data['circulating_supply'],
                'total_supply': crypto_data['total_supply'],
                'max_supply': crypto_data['max_supply'],
                'price_change_24h': market_data['percent_change_24h'],
                'market_dominance': market_data.get('market_cap_dominance', 0),
                'trading_pairs': [],  # Will be populated from exchange data
                'exchanges': []  # Will be populated from exchange data
            }
        return {
            'name': symbol.split('-')[0],
            'type': 'crypto',
            'market_cap': 0,
            'volume_24h': 0,
            'circulating_supply': 0,
            'total_supply': 0,
            'max_supply': 0,
            'price_change_24h': 0,
            'market_dominance': 0,
            'trading_pairs': [],
            'exchanges': []
        }
    except Exception as e:
        st.error(f"Error fetching crypto data: {str(e)}")
        return {
            'name': symbol.split('-')[0],
            'type': 'crypto',
            'market_cap': 0,
            'volume_24h': 0,
            'circulating_supply': 0,
            'total_supply': 0,
            'max_supply': 0,
            'price_change_24h': 0,
            'market_dominance': 0,
            'trading_pairs': [],
            'exchanges': []
        }

def get_stock_info(ticker: str) -> dict:
    """Get stock/crypto information and fundamentals."""
    try:
        if is_crypto(ticker):
            crypto_data = get_coinmarketcap_data(ticker)
            if crypto_data:
                return crypto_data
            
            # Fallback to Yahoo Finance if CoinMarketCap fails
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker.upper()),  # Fallback to ticker if name not found
                'type': 'crypto',
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
                'total_supply': info.get('totalSupply', 0),
                'max_supply': info.get('maxSupply', 0),
                'price_change_24h': info.get('priceChangePercent24h', 0)
            }
        else:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker.upper()),  # Fallback to ticker if name not found
                'type': 'stock',
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'eps': info.get('trailingEps', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'price_to_book': info.get('priceToBook', 0)
            }
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return {
            'name': ticker.upper(),  # Use ticker as fallback
            'type': 'crypto' if is_crypto(ticker) else 'stock',
            'market_cap': 0
        }

def fuzzy_match(query: str, target: str, threshold: float = 0.6) -> bool:
    """Simple fuzzy matching function."""
    query = query.lower()
    target = target.lower()
    
    # Exact match or substring match
    if query in target or target in query:
        return True
    
    # Calculate simple similarity score
    shorter = query if len(query) < len(target) else target
    longer = target if len(query) < len(target) else query
    
    matches = sum(1 for char in shorter if char in longer)
    return matches / len(longer) >= threshold

@st.cache_data(ttl=3600)
def search_stocks(query: str) -> list:
    """Search for stocks and cryptocurrencies matching the query."""
    try:
        if not query or len(query) < 2:
            return []
            
        st.session_state['last_error'] = None  # Clear any previous errors
        results = []
        
        # First try Yahoo Finance API
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {'q': query, 'quotesCount': 20, 'newsCount': 0}
            r = requests.get(url, params=params)
            data = r.json()
            
            for quote in data.get('quotes', []):
                symbol = quote.get('symbol', '')
                if symbol:  # Only add if symbol exists
                    is_crypto_symbol = is_crypto(symbol)
                    if is_crypto_symbol:
                        symbol = format_crypto_symbol(symbol)
                    results.append({
                        'symbol': symbol,
                        'name': quote.get('longname', quote.get('shortname', symbol)),
                        'exchange': quote.get('exchange', 'Unknown'),
                        'type': 'crypto' if is_crypto_symbol else 'stock'
                    })
        except Exception as e:
            st.session_state['last_error'] = f"Yahoo Finance search error: {str(e)}"
            
        # If no results, try direct symbol match
        if not results:
            try:
                stock = yf.Ticker(query.upper())
                info = stock.info
                if 'symbol' in info:
                    is_crypto_symbol = is_crypto(info['symbol'])
                    symbol = format_crypto_symbol(info['symbol']) if is_crypto_symbol else info['symbol']
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'exchange': info.get('exchange', 'Unknown'),
                        'type': 'crypto' if is_crypto_symbol else 'stock'
                    })
            except Exception as e:
                if not st.session_state.get('last_error'):
                    st.session_state['last_error'] = f"Symbol lookup error: {str(e)}"
        
        # Try cryptocurrency search if still no results or query looks like crypto
        if not results or any(q in query.upper() for q in ['BTC', 'ETH', 'USDT', 'USD']):
            try:
                response = requests.get(
                    "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                    headers={'X-CMC_PRO_API_KEY': st.secrets["COINMARKETCAP_API_KEY"]},
                    params={'start': 1, 'limit': 100, 'convert': 'USD'}
                )
                crypto_data = response.json()
                
                for crypto in crypto_data.get('data', []):
                    if fuzzy_match(query.upper(), crypto['symbol']) or fuzzy_match(query.lower(), crypto['name'].lower()):
                        symbol = f"{crypto['symbol']}-USD"
                        results.append({
                            'symbol': symbol,
                            'name': crypto['name'],
                            'exchange': 'Crypto',
                            'type': 'crypto'
                        })
            except Exception as e:
                if not st.session_state.get('last_error'):
                    st.session_state['last_error'] = f"Cryptocurrency search error: {str(e)}"
        
        return results[:10]  # Limit to top 10 results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []
            
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
