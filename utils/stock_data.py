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
    if not query or len(query) < 2:
        return []
        
    st.session_state['last_error'] = None  # Clear any previous errors
    results = []
    
    # Define common tickers for direct matching
    common_tickers = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com, Inc.",
        "GOOGL": "Alphabet Inc.",
        "GOOG": "Alphabet Inc.",
        "META": "Meta Platforms, Inc.",
        "TSLA": "Tesla, Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "BRK.A": "Berkshire Hathaway Inc.",
        "BRK.B": "Berkshire Hathaway Inc.",
        "V": "Visa Inc.",
        "JNJ": "Johnson & Johnson",
        "WMT": "Walmart Inc.",
        "MA": "Mastercard Incorporated",
        "PG": "The Procter & Gamble Company",
        "UNH": "UnitedHealth Group Incorporated",
        "HD": "The Home Depot, Inc.",
        "BAC": "Bank of America Corporation",
        "DIS": "The Walt Disney Company",
        "NFLX": "Netflix, Inc.",
        "PYPL": "PayPal Holdings, Inc.",
        "INTC": "Intel Corporation",
        "VZ": "Verizon Communications Inc.",
        "ADBE": "Adobe Inc.",
        "CMCSA": "Comcast Corporation",
        "PFE": "Pfizer Inc.",
        "KO": "The Coca-Cola Company",
        "T": "AT&T Inc.",
        "CSCO": "Cisco Systems, Inc.",
        "BTC-USD": "Bitcoin USD",
        "ETH-USD": "Ethereum USD",
        "USDT-USD": "Tether USD",
        "XRP-USD": "XRP USD",
        "BNB-USD": "Binance Coin USD",
        "ADA-USD": "Cardano USD",
        "SOL-USD": "Solana USD",
        "DOGE-USD": "Dogecoin USD",
        "DOT-USD": "Polkadot USD",
        "SHIB-USD": "Shiba Inu USD",
        "AVAX-USD": "Avalanche USD",
        "MATIC-USD": "Polygon USD",
        "LTC-USD": "Litecoin USD"
    }
    
    # Method 1: Direct symbol match from known tickers
    query_upper = query.upper()
    for symbol, name in common_tickers.items():
        if query_upper in symbol or query.lower() in name.lower():
            is_crypto_symbol = is_crypto(symbol)
            results.append({
                'symbol': symbol,
                'name': name,
                'exchange': 'Crypto' if is_crypto_symbol else 'Major Exchange',
                'type': 'crypto' if is_crypto_symbol else 'stock'
            })
    
    # Method 2: Try Yahoo Finance Search API (with more robust error handling)
    if not results:
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {'q': query, 'quotesCount': 20, 'newsCount': 0}
            r = requests.get(url, params=params, timeout=5)
            
            # Make sure we have valid JSON before parsing
            if r.status_code == 200 and r.text and r.text.strip():
                try:
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
                except ValueError as json_err:
                    st.session_state['last_error'] = f"Yahoo Finance API response not valid JSON: {str(json_err)}"
            else:
                st.session_state['last_error'] = f"Yahoo Finance API returned status code {r.status_code}"
        except Exception as e:
            st.session_state['last_error'] = f"Yahoo Finance search API error: {str(e)}"
    
    # Method 3: Try direct yfinance lookup - try it as a symbol
    if not results:
        try:
            # Direct ticker lookup
            symbol = query.upper()
            stock = yf.Ticker(symbol)
            
            # Use a lightweight request first to check if it's valid
            hist = stock.history(period="1d")
            
            if not hist.empty:
                info = stock.info
                name = info.get('longName', info.get('shortName', symbol))
                is_crypto_symbol = is_crypto(symbol)
                
                if is_crypto_symbol:
                    symbol = format_crypto_symbol(symbol)
                
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'exchange': info.get('exchange', 'Unknown'),
                    'type': 'crypto' if is_crypto_symbol else 'stock'
                })
        except Exception as e:
            if not st.session_state.get('last_error'):
                st.session_state['last_error'] = f"Direct symbol lookup failed: {str(e)}"
    
    # Method 4: Try cryptocurrency search with CoinMarketCap API if appropriate
    if (not results or any(q in query_upper for q in ['BTC', 'ETH', 'USDT', 'USD', 'COIN', 'CRYPTO'])):
        try:
            # Check if we have the API key before making the request
            if "COINMARKETCAP_API_KEY" in st.secrets:
                response = requests.get(
                    "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                    headers={'X-CMC_PRO_API_KEY': st.secrets["COINMARKETCAP_API_KEY"]},
                    params={'start': 1, 'limit': 100, 'convert': 'USD'},
                    timeout=5
                )
                
                # Check for valid response before parsing
                if response.status_code == 200:
                    crypto_data = response.json()
                    
                    for crypto in crypto_data.get('data', []):
                        if fuzzy_match(query_upper, crypto['symbol']) or fuzzy_match(query.lower(), crypto['name'].lower()):
                            symbol = f"{crypto['symbol']}-USD"
                            results.append({
                                'symbol': symbol,
                                'name': crypto['name'],
                                'exchange': 'Crypto',
                                'type': 'crypto'
                            })
            else:
                st.session_state['last_error'] = "CoinMarketCap API key not configured. Please add the COINMARKETCAP_API_KEY to secrets."
        except Exception as e:
            if not st.session_state.get('last_error'):
                st.session_state['last_error'] = f"Cryptocurrency search error: {str(e)}"
    
    # Remove duplicates and sort by relevance
    unique_results = []
    seen = set()
    for item in results:
        if item['symbol'] not in seen:
            seen.add(item['symbol'])
            unique_results.append(item)
    
    # Sort results to show cryptocurrencies first when the query looks like a crypto symbol
    if query_upper.endswith(('BTC', 'ETH', 'USD', 'USDT')):
        unique_results.sort(key=lambda x: 0 if x['type'] == 'crypto' else 1)
    
    # If we still have no results, provide feedback
    if not unique_results and not st.session_state.get('last_error'):
        st.session_state['last_error'] = f"No results found for '{query}'. Try a different symbol or company name."
            
    return unique_results[:10]  # Limit to top 10 results
