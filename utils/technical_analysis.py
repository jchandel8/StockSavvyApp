import pandas as pd
import numpy as np
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rsi(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    try:
        if data is None or data.empty or 'Close' not in data.columns:
            logger.error("Invalid data for RSI calculation")
            return pd.Series(index=data.index if data is not None else None)
            
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = pd.Series(100 - (100 / (1 + rs)), index=data.index)
        logger.info("RSI calculation successful")
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=data.index if data is not None else None)

def calculate_macd(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    try:
        if data is None or data.empty or 'Close' not in data.columns:
            logger.error("Invalid data for MACD calculation")
            empty_series = pd.Series(index=data.index if data is not None else None)
            return empty_series, empty_series, empty_series
            
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        logger.info("MACD calculation successful")
        return macd, signal, hist
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        empty_series = pd.Series(index=data.index if data is not None else None)
        return empty_series, empty_series, empty_series

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    try:
        if data is None or data.empty or 'Close' not in data.columns:
            logger.error("Invalid data for Bollinger Bands calculation")
            empty_series = pd.Series(index=data.index if data is not None else None)
            return empty_series, empty_series, empty_series
            
        middle = pd.Series(data['Close'].rolling(window=window).mean(), index=data.index)
        std = data['Close'].rolling(window=window).std()
        upper = pd.Series(middle + (std * num_std), index=data.index)
        lower = pd.Series(middle - (std * num_std), index=data.index)
        logger.info("Bollinger Bands calculation successful")
        return upper, middle, lower
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        empty_series = pd.Series(index=data.index if data is not None else None)
        return empty_series, empty_series, empty_series

@st.cache_data
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given stock data."""
    if df is None:
        logger.error("DataFrame is None")
        return pd.DataFrame()
        
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
        
    required_columns = ['Close', 'High', 'Low', 'Open']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns. Required: {required_columns}")
        return df
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Calculate Moving Averages
        logger.info("Calculating Moving Averages")
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        logger.info("Calculating RSI")
        df['RSI'] = calculate_rsi(df)
        
        # MACD
        logger.info("Calculating MACD")
        macd, signal, hist = calculate_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # Bollinger Bands
        logger.info("Calculating Bollinger Bands")
        upper, middle, lower = calculate_bollinger_bands(df)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # Verify calculations
        required_indicators = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
        if missing_indicators:
            logger.warning(f"Missing indicators after calculation: {missing_indicators}")
        else:
            logger.info("All indicators calculated successfully")
            
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def generate_signals(df: pd.DataFrame) -> dict:
    """Generate trading signals based on technical indicators."""
    signals = {
        'buy_signals': [],
        'sell_signals': [],
        'reasoning': [],
        'status': 'loading'  # New status field
    }
    
    if df is None or df.empty:
        logger.warning("No data available for signal generation")
        signals['status'] = 'no_data'
        return signals
    
    required_columns = ['Close', 'RSI', 'MACD_Hist', 'SMA_50']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns for signal generation: {[col for col in required_columns if col not in df.columns]}")
        signals['status'] = 'missing_data'
        return signals
    
    try:
        logger.info("Starting signal generation process")
        
        # Get latest values and check for NaN
        latest_values = {
            'rsi': df['RSI'].iloc[-1],
            'macd_hist_current': df['MACD_Hist'].iloc[-1],
            'macd_hist_prev': df['MACD_Hist'].iloc[-2],
            'close_current': df['Close'].iloc[-1],
            'close_prev': df['Close'].iloc[-2],
            'sma50_current': df['SMA_50'].iloc[-1],
            'sma50_prev': df['SMA_50'].iloc[-2]
        }
        
        # Check for NaN values
        nan_values = {k: v for k, v in latest_values.items() if pd.isna(v)}
        if nan_values:
            logger.warning(f"NaN values found in: {list(nan_values.keys())}")
            signals['status'] = 'invalid_data'
            return signals
        
        # RSI Signals
        logger.info("Checking RSI signals")
        if latest_values['rsi'] < 30:
            signals['buy_signals'].append('RSI oversold')
            logger.info("RSI oversold signal generated")
        elif latest_values['rsi'] > 70:
            signals['sell_signals'].append('RSI overbought')
            logger.info("RSI overbought signal generated")
        
        # MACD Signals
        logger.info("Checking MACD signals")
        if latest_values['macd_hist_current'] > 0 and latest_values['macd_hist_prev'] <= 0:
            signals['buy_signals'].append('MACD bullish crossover')
            logger.info("MACD bullish crossover signal generated")
        elif latest_values['macd_hist_current'] < 0 and latest_values['macd_hist_prev'] >= 0:
            signals['sell_signals'].append('MACD bearish crossover')
            logger.info("MACD bearish crossover signal generated")
        
        # Moving Average Signals
        logger.info("Checking Moving Average signals")
        if (latest_values['close_current'] > latest_values['sma50_current'] and 
            latest_values['close_prev'] <= latest_values['sma50_prev']):
            signals['buy_signals'].append('Price crossed above 50-day MA')
            logger.info("MA bullish crossover signal generated")
        elif (latest_values['close_current'] < latest_values['sma50_current'] and 
              latest_values['close_prev'] >= latest_values['sma50_prev']):
            signals['sell_signals'].append('Price crossed below 50-day MA')
            logger.info("MA bearish crossover signal generated")
        
        signals['status'] = 'complete'
        logger.info("Signal generation completed successfully")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        signals['status'] = 'error'
        return signals
