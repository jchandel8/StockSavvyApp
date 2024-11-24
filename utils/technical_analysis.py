import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st

@st.cache_data
def calculate_indicators(df: pd.DataFrame) -> dict:
    """Calculate technical indicators for the given stock data."""
    if df.empty:
        return {}
    
    try:
        # Calculate Moving Averages
        df['SMA_20'] = df.ta.sma(length=20)
        df['SMA_50'] = df.ta.sma(length=50)
        df['EMA_20'] = df.ta.ema(length=20)
        
        # RSI
        df['RSI'] = df.ta.rsi(length=14)
        
        # MACD
        macd = df.ta.macd()
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bbands = df.ta.bbands()
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0']
        
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
