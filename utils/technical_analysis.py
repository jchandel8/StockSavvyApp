import pandas as pd
import numpy as np
import streamlit as st

def calculate_rsi(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

@st.cache_data
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given stock data."""
    if df.empty:
        return df
    
    try:
        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        df['RSI'] = calculate_rsi(df)
        
        # MACD
        macd, signal, hist = calculate_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(df)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def generate_signals(df: pd.DataFrame) -> dict:
    """Generate trading signals based on technical indicators."""
    signals = {
        'buy_signals': [],
        'sell_signals': [],
        'reasoning': []
    }
    
    if df.empty:
        return signals
    
    # RSI Signals
    if df['RSI'].iloc[-1] < 30:
        signals['buy_signals'].append('RSI oversold')
    elif df['RSI'].iloc[-1] > 70:
        signals['sell_signals'].append('RSI overbought')
    
    # MACD Signals
    if df['MACD_Hist'].iloc[-1] > 0 and df['MACD_Hist'].iloc[-2] <= 0:
        signals['buy_signals'].append('MACD bullish crossover')
    elif df['MACD_Hist'].iloc[-1] < 0 and df['MACD_Hist'].iloc[-2] >= 0:
        signals['sell_signals'].append('MACD bearish crossover')
    
    # Moving Average Signals
    if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1] and \
       df['Close'].iloc[-2] <= df['SMA_50'].iloc[-2]:
        signals['buy_signals'].append('Price crossed above 50-day MA')
    elif df['Close'].iloc[-1] < df['SMA_50'].iloc[-1] and \
         df['Close'].iloc[-2] >= df['SMA_50'].iloc[-2]:
        signals['sell_signals'].append('Price crossed below 50-day MA')
    
    return signals
