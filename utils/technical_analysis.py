import pandas as pd
import numpy as np
import talib
import streamlit as st

@st.cache_data
def calculate_indicators(df: pd.DataFrame) -> dict:
    """Calculate technical indicators for the given stock data."""
    if df.empty:
        return {}
    
    try:
        # Calculate Moving Averages
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'])
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
