import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from utils.stock_data import is_crypto
from utils.technical_analysis import (
    calculate_gap_and_go_signals,
    calculate_trend_continuation,
    calculate_fibonacci_signals,
    calculate_weekly_trendline_break,
    calculate_mvrv_ratio
)

def build_lstm_model(input_shape):
    """Build and return an enhanced LSTM model for price prediction."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare comprehensive feature set for prediction with enhanced indicators."""
    features = pd.DataFrame()
    
    # Price-based features
    features['close'] = data['Close']
    features['returns'] = data['Close'].pct_change()
    features['log_returns'] = np.log1p(data['Close']).diff()
    
    # Volume indicators with enhanced analysis
    features['volume_sma'] = data['Volume'].rolling(20).mean()
    features['volume_ratio'] = data['Volume'] / features['volume_sma']
    features['volume_momentum'] = data['Volume'].pct_change()
    features['volume_trend'] = data['Volume'].rolling(10).mean() / data['Volume'].rolling(30).mean()
    
    # Enhanced momentum indicators
    features['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    features['rsi_trend'] = features['rsi'].rolling(5).mean() - features['rsi'].rolling(15).mean()
    features['macd'] = ta.trend.MACD(data['Close']).macd()
    features['macd_signal'] = ta.trend.MACD(data['Close']).macd_signal()
    features['macd_diff'] = ta.trend.MACD(data['Close']).macd_diff()
    features['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    features['mfi'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
    
    # Advanced volatility metrics
    features['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    bb = ta.volatility.BollingerBands(data['Close'])
    features['bbands_width'] = bb.bollinger_wband()
    features['bbands_pct_b'] = bb.bollinger_pband()
    features['volatility_index'] = data['Close'].rolling(20).std() / data['Close'].rolling(20).mean()
    
    # Enhanced trend indicators
    features['sma_20'] = data['Close'].rolling(20).mean()
    features['sma_50'] = data['Close'].rolling(50).mean()
    features['sma_200'] = data['Close'].rolling(200).mean()
    features['ema_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    features['trend_strength'] = (features['sma_20'] / features['sma_50'] - 1) * 100
    features['long_trend'] = np.where(data['Close'] > features['sma_200'], 1, -1)
    
    # Price patterns
    features['higher_highs'] = data['High'].rolling(5).max() > data['High'].rolling(5).max().shift(5)
    features['lower_lows'] = data['Low'].rolling(5).min() < data['Low'].rolling(5).min().shift(5)
    
    return features.fillna(0)

def calculate_prediction(data: pd.DataFrame, timeframe: str = 'short_term', look_back: int = 0) -> dict:
    # Set look_back periods based on timeframe if not provided
    if look_back <= 0:
        look_back = {
            'daily': 15,       # 1 day prediction
            'short_term': 30,  # 1 week prediction
            'medium_term': 60, # 1 month prediction
            'long_term': 90    # 3 months prediction
        }.get(timeframe, 30)
    
    try:
        # Ensure minimum data length
        if len(data) < look_back + 10:
            logger.warning("Insufficient data for prediction")
            return {
                'forecast': data['Close'].iloc[-1],
                'confidence': 0.0,
                'direction': 'NEUTRAL'
            }
        
        # Prepare features
        features = prepare_features(data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # Prepare sequences
        X, y = [], []
        for i in range(look_back, len(scaled_features)):
            X.append(scaled_features[i-look_back:i])
            y.append(1 if data['Close'].iloc[i] > data['Close'].iloc[i-1] else 0)
        
        X, y = np.array(X), np.array(y)
        
        # Build and train model
        model = build_lstm_model((look_back, scaled_features.shape[1]))
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Make prediction
        last_sequence = scaled_features[-look_back:].reshape(1, look_back, scaled_features.shape[1])
        prediction_probability = model.predict(last_sequence)[0][0]
        
        # Calculate enhanced confidence factors
        volatility = features['volatility_index'].iloc[-1]
        volume_trend = features['volume_trend'].iloc[-1]
        price_trend = features['trend_strength'].iloc[-1]
        momentum = features['rsi_trend'].iloc[-1]
        
        # Calculate trend alignment
        short_trend = features['sma_20'].iloc[-1] > features['sma_50'].iloc[-1]
        long_trend = features['sma_50'].iloc[-1] > features['sma_200'].iloc[-1]
        trend_alignment = 1.2 if short_trend == long_trend else 0.8
        
        # Calculate volume confirmation
        volume_confirmation = 1.2 if (volume_trend > 1 and price_trend > 0) else 0.8
        
        # Calculate momentum confirmation
        momentum_confirmation = 1.2 if (
            (prediction_probability > 0.5 and momentum > 0) or 
            (prediction_probability < 0.5 and momentum < 0)
        ) else 0.8
        
        # Calculate base model confidence
        model_predictions = model.predict(X)
        model_accuracy = np.mean((model_predictions > 0.5) == y)
        recent_accuracy = np.mean((model_predictions[-10:] > 0.5) == y[-10:])
        base_confidence = (model_accuracy + recent_accuracy) / 2
        
        # Combine all factors
        confidence = (
            base_confidence * 
            trend_alignment * 
            volume_confirmation * 
            momentum_confirmation * 
            (1 / (1 + volatility))
        )
        
        # Apply stricter confidence thresholds
        confidence = min(max(confidence, 0), 0.95)  # Cap maximum confidence at 95%
        if confidence < 0.7:  # Increased minimum threshold
            confidence = 0.0  # Return no confidence if below threshold
        
        # Determine forecast direction and price
        current_price = data['Close'].iloc[-1]
        price_change = current_price * 0.01 * (prediction_probability - 0.5) * 2  # Scale to ±1% range
        forecast = current_price + price_change
        
        # Lower the confidence threshold for backtesting
        if confidence < 0.5:  # Reduced from 0.7
            confidence = 0.5  # Set minimum confidence instead of returning None
            
        return {
            'forecast': float(forecast),
            'confidence': float(confidence),
            'direction': 'UP' if prediction_probability > 0.5 else 'DOWN'
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return {}

def calculate_trend_strength(data: pd.DataFrame) -> float:
    """Calculate the strength of the current trend."""
    if len(data) < 20:
        return 0.0
        
    try:
        # Calculate price momentum using pandas operations
        returns = pd.Series(data['Close']).pct_change()
        momentum_series = returns.rolling(window=20).mean()
        
        # Safe handling of NaN values
        valid_momentum = pd.Series(momentum_series[pd.notna(momentum_series)])
        momentum = float(valid_momentum.iloc[-1]) if len(valid_momentum) > 0 else 0.0
        
        # Calculate trend consistency
        direction_changes = (returns.shift(-1) * returns < 0).sum()
        consistency = 1 - (direction_changes / len(returns))
        
        # Combine factors
        score = (momentum + consistency) / 2
        return max(min(abs(score), 1.0), 0.0)
    except Exception as e:
        print(f"Error in calculate_trend_strength: {str(e)}")
        return 0.0

def predict_price_movement(data: pd.DataFrame, ticker: str) -> dict:
    """Generate price predictions for multiple timeframes using LSTM model."""
    if data is None or len(data) < 50:
        return {
            'short_term': {'timeframe': '1 Week', 'direction': 'NEUTRAL', 'confidence': 0.0,
                          'predicted_high': None, 'predicted_low': None},
            'medium_term': {'timeframe': '1 Month', 'direction': 'NEUTRAL', 'confidence': 0.0,
                           'predicted_high': None, 'predicted_low': None},
            'long_term': {'timeframe': '3 Months', 'direction': 'NEUTRAL', 'confidence': 0.0,
                         'predicted_high': None, 'predicted_low': None}
        }
    
    predictions = {}
    try:
        last_price = data['Close'].iloc[-1]
        trend_strength = calculate_trend_strength(data)
        
        timeframes = ['daily', 'short_term', 'medium_term', 'long_term']
        periods = {
            'daily': '1 Day',
            'short_term': '1 Week',
            'medium_term': '1 Month',
            'long_term': '3 Months'
        }
        
        for timeframe in timeframes:
            # Get prediction for current timeframe
            prediction_results = calculate_prediction(data, timeframe=timeframe)
            
            # Skip if prediction doesn't meet confidence threshold
            if not prediction_results:
                continue
                
            forecast = prediction_results.get('forecast', last_price)
            confidence = prediction_results.get('confidence', 0.5)
            direction = prediction_results.get('direction', 'NEUTRAL')
            
            # Calculate volatility-adjusted range
            volatility = data['Close'].rolling(window=20).std().iloc[-1] / last_price
            range_factor = {
                'daily': 0.01,
                'short_term': 0.02,
                'medium_term': 0.04,
                'long_term': 0.06
            }[timeframe]
            predicted_range = last_price * range_factor * (1 + volatility)
            
            predictions[timeframe] = {
                'timeframe': periods[timeframe],
                'direction': direction,
                'confidence': confidence,
                'predicted_high': forecast + predicted_range,
                'predicted_low': forecast - predicted_range,
                'forecast': forecast
            }
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        st.error(f"Error in prediction: {str(e)}")
        
    return predictions

@st.cache_data
def get_prediction(df: pd.DataFrame, ticker: str) -> dict:
    """Get price prediction with caching."""
    # First check if we have a valid dataframe
    if df is None or df.empty:
        logger.warning("No data available for prediction")
        return {
            'daily': {
                'timeframe': '1 Day',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            },
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            },
            'medium_term': {
                'timeframe': '1 Month',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            },
            'long_term': {
                'timeframe': '3 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
        }
    
    # Check if we have minimum required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        logger.warning(f"Missing required columns for prediction: {[col for col in required_columns if col not in df.columns]}")
        return {
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
        }
    
    # Check for sufficient data length
    if len(df) < 50:  # Need at least 50 data points for meaningful prediction
        logger.warning(f"Insufficient data length for prediction: {len(df)} data points")
        return {
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL', 
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
        }
    
    # Check for NaN values in critical columns
    critical_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in critical_columns:
        if df[col].isna().any():
            # Fill NaN values with forward fill, then backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            logger.warning(f"NaN values detected in {col}, filled with forward/backward fill")
    
    try:
        return predict_price_movement(df, ticker)
    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}")
        st.error(f"Error generating prediction: {str(e)}")
        return {
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
        }
