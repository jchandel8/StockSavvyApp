import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given dataframe."""
    try:
        # Calculate SMA
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['BB_middle'] - (df['Close'].rolling(window=20).std() * 2)
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return df

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
def calculate_indicators(df: pd.DataFrame, is_crypto: bool = False) -> pd.DataFrame:
    """Calculate technical indicators for the given stock/crypto data."""
    # Adjust time windows for crypto (24/7 trading)
    rsi_period = 14 if not is_crypto else 24
    macd_fast = 12 if not is_crypto else 20
    macd_slow = 26 if not is_crypto else 44
    macd_signal = 9 if not is_crypto else 15
    bb_period = 20 if not is_crypto else 34
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

def calculate_mvrv_ratio(df: pd.DataFrame) -> pd.Series:
    """Calculate Market Value to Realized Value (MVRV) ratio for cryptocurrencies."""
    try:
        if df is None or df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
            return pd.Series(index=df.index if df is not None else None)
        
        # Calculate realized value (average cost basis)
        volume_price = df['Close'] * df['Volume']
        realized_value = volume_price.rolling(window=30).sum() / df['Volume'].rolling(window=30).sum()
        
        # Calculate MVRV ratio
        market_value = df['Close']
        mvrv_ratio = market_value / realized_value
        
        return mvrv_ratio
    except Exception as e:
        logger.error(f"Error calculating MVRV ratio: {str(e)}")
        return pd.Series(index=df.index if df is not None else None)

def calculate_gap_and_go_signals(df: pd.DataFrame) -> pd.Series:
    """Calculate Gap and Go signals."""
    try:
        if df is None or df.empty or not all(col in df.columns for col in ['Close', 'Open']):
            return pd.Series(index=df.index if df is not None else None)
            
        # Calculate price gaps
        gaps = df['Open'] - df['Close'].shift(1)
        gap_threshold = df['Close'].rolling(window=20).std() * 1.5
        
        # Identify significant gaps
        gap_signals = (abs(gaps) > gap_threshold) & (df['Close'] > df['Open'])
        
        return gap_signals
    except Exception as e:
        logger.error(f"Error calculating Gap and Go signals: {str(e)}")
        return pd.Series(index=df.index if df is not None else None)

def calculate_trend_continuation(df: pd.DataFrame) -> pd.Series:
    """Calculate trend continuation signals."""
    try:
        if df is None or df.empty:
            return pd.Series(index=df.index if df is not None else None)
            
        # Calculate trending conditions
        sma20 = df['Close'].rolling(window=20).mean()
        sma50 = df['Close'].rolling(window=50).mean()
        
        # Price above both MAs and 20 MA above 50 MA indicates uptrend
        uptrend = (df['Close'] > sma20) & (sma20 > sma50)
        
        # Volume confirmation
        volume_increase = df['Volume'] > df['Volume'].rolling(window=20).mean()
        
        # Combine signals
        trend_continuation = uptrend & volume_increase
        
        return trend_continuation
        
    except Exception as e:
        logger.error(f"Error calculating trend continuation: {str(e)}")
        return pd.Series(index=df.index if df is not None else None)

def calculate_fibonacci_signals(df: pd.DataFrame) -> pd.Series:
    """Calculate Fibonacci retracement signals."""
    try:
        if df is None or df.empty:
            return pd.Series(index=df.index if df is not None else None)
            
        signals = pd.Series(False, index=df.index)
        
        # Calculate Fibonacci levels
        window = 20
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            high = window_data['High'].max()
            low = window_data['Low'].min()
            diff = high - low
            
            # Fibonacci levels
            fib_382 = high - diff * 0.382
            fib_618 = high - diff * 0.618
            
            # Generate signal when price bounces from Fibonacci levels
            current_price = df['Close'].iloc[i]
            if (fib_382 * 0.99 <= current_price <= fib_382 * 1.01) or \
               (fib_618 * 0.99 <= current_price <= fib_618 * 1.01):
                signals.iloc[i] = True
                
        return signals
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci signals: {str(e)}")
        return pd.Series(index=df.index if df is not None else None)

def calculate_weekly_trendline_break(df: pd.DataFrame) -> pd.Series:
    """Calculate weekly trendline breakout signals."""
    try:
        if df is None or df.empty:
            return pd.Series(index=df.index if df is not None else None)
            
        signals = pd.Series(False, index=df.index)
        
        # Calculate weekly high and low points
        df['Week'] = df.index.isocalendar().week
        weekly_highs = df.groupby('Week')['High'].max()
        weekly_lows = df.groupby('Week')['Low'].min()
        
        # Look for breakouts from previous week's range
        for i in range(1, len(df)):
            current_week = df.index[i].isocalendar().week
            prev_week = df.index[i-1].isocalendar().week
            
            if current_week != prev_week:
                prev_high = weekly_highs[prev_week]
                prev_low = weekly_lows[prev_week]
                
                # Breakout signal
                if df['Close'].iloc[i] > prev_high * 1.02:  # 2% breakout threshold
                    signals.iloc[i] = True
                    
        return signals
        
    except Exception as e:
        logger.error(f"Error calculating weekly trendline breaks: {str(e)}")
        return pd.Series(index=df.index if df is not None else None)

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
    
    required_columns = ['Close', 'Open', 'High', 'Low', 'RSI', 'MACD_Hist', 'SMA_50']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns for signal generation: {missing_columns}")
        signals['status'] = 'missing_data'
        return signals
    
    try:
        logger.info("Starting signal generation process")
        
        # Calculate new signals with proper error handling
        try:
            gap_and_go = calculate_gap_and_go_signals(df)
        except Exception as e:
            logger.warning(f"Error calculating gap_and_go signals: {str(e)}")
            gap_and_go = pd.Series(False, index=df.index)
            
        try:
            trend_continuation = calculate_trend_continuation(df)
        except Exception as e:
            logger.warning(f"Error calculating trend_continuation signals: {str(e)}")
            trend_continuation = pd.Series(False, index=df.index)
            
        try:
            fibonacci = calculate_fibonacci_signals(df)
        except Exception as e:
            logger.warning(f"Error calculating fibonacci signals: {str(e)}")
            fibonacci = pd.Series(False, index=df.index)
            
        try:
            weekly_trendline = calculate_weekly_trendline_break(df)
        except Exception as e:
            logger.warning(f"Error calculating weekly_trendline signals: {str(e)}")
            weekly_trendline = pd.Series(False, index=df.index)
        
        # Get latest values with safe defaults for NaN values
        # Helper function to safely get values with defaults
        def safe_get(series, idx=-1, default=0.0):
            try:
                val = series.iloc[idx]
                return default if pd.isna(val) else val
            except (IndexError, AttributeError, KeyError):
                return default
        
        # Safely populate all values with defaults
        latest_values = {
            'rsi': safe_get(df['RSI'], -1, 50.0),  # Neutral RSI
            'macd_hist_current': safe_get(df['MACD_Hist'], -1, 0.0),
            'macd_hist_prev': safe_get(df['MACD_Hist'], -2, 0.0),
            'close_current': safe_get(df['Close'], -1, 100.0),
            'close_prev': safe_get(df['Close'], -2, 100.0),
            'sma50_current': safe_get(df['SMA_50'], -1, 100.0),
            'sma50_prev': safe_get(df['SMA_50'], -2, 100.0),
            'gap_and_go': safe_get(gap_and_go, -1, False),
            'trend_continuation': safe_get(trend_continuation, -1, False),
            'fibonacci': safe_get(fibonacci, -1, False),
            'weekly_trendline': safe_get(weekly_trendline, -1, False)
        }
        
        # Check for NaN values in critical fields only
        critical_fields = ['rsi', 'close_current', 'macd_hist_current']
        nan_values = {k: v for k, v in latest_values.items() if k in critical_fields and pd.isna(v)}
        if nan_values:
            logger.warning(f"Critical NaN values found in: {list(nan_values.keys())}")
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
        
        # Gap and Go Signals
        logger.info("Checking Gap and Go signals")
        if latest_values['gap_and_go']:
            signals['buy_signals'].append('Gap and Go pattern detected')
            logger.info("Gap and Go signal generated")
        
        # Trend Continuation Signals
        logger.info("Checking Trend Continuation signals")
        if latest_values['trend_continuation']:
            signals['buy_signals'].append('Trend continuation confirmed')
            logger.info("Trend continuation signal generated")
        
        # Fibonacci Signals
        logger.info("Checking Fibonacci signals")
        if latest_values['fibonacci']:
            signals['buy_signals'].append('Fibonacci retracement level reached')
            logger.info("Fibonacci signal generated")
        
        # Weekly Trendline Break Signals
        logger.info("Checking Weekly Trendline Break signals")
        if latest_values['weekly_trendline']:
            signals['buy_signals'].append('Weekly trendline breakout')
            logger.info("Weekly trendline break signal generated")
        
        signals['status'] = 'complete'
        logger.info("Signal generation completed successfully")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        signals['status'] = 'error'
        return signals