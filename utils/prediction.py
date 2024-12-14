import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

def build_model():
    """Build and return a Random Forest model for price prediction."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare comprehensive feature set for prediction."""
    features = pd.DataFrame()
    
    # Price-based features
    features['close'] = data['Close']
    features['returns'] = data['Close'].pct_change()
    features['log_returns'] = np.log1p(data['Close']).diff()
    
    # Volume indicators
    features['volume_sma'] = data['Volume'].rolling(20).mean()
    features['volume_ratio'] = data['Volume'] / features['volume_sma']
    
    # Momentum indicators
    features['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    features['macd'] = ta.trend.MACD(data['Close']).macd()
    features['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    
    # Volatility
    features['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    features['bbands_width'] = ta.volatility.BollingerBands(data['Close']).bollinger_wband()
    
    # Trend indicators
    features['sma_20'] = data['Close'].rolling(20).mean()
    features['sma_50'] = data['Close'].rolling(50).mean()
    features['trend'] = np.where(features['sma_20'] > features['sma_50'], 1, -1)
    
    return features.fillna(0)

def calculate_prediction(data: pd.DataFrame, timeframe: str = 'short_term', look_back: int = None) -> dict:
    # Set look_back periods based on timeframe
    if look_back is None:
        look_back = {
            'daily': 15,       # 1 day prediction
            'short_term': 30,  # 1 week prediction
            'medium_term': 60, # 1 month prediction
            'long_term': 90    # 3 months prediction
        }.get(timeframe, 30)
    
    try:
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
        model = build_model()
        model.fit(X, y)
        
        # Make prediction
        last_sequence = scaled_features[-look_back:].reshape(1, -1)
        prediction_probability = model.predict(last_sequence)[0]
        
        # Calculate confidence factors
        volatility = features['atr'].iloc[-1] / data['Close'].iloc[-1]
        volume_trend = features['volume_ratio'].iloc[-1]
        trend_strength = 1 if features['trend'].iloc[-1] * prediction_probability > 0.5 else 0.5
        
        # Calculate prediction confidence
        model_accuracy = np.mean(model.predict(X) == y)
        base_confidence = model_accuracy
        volatility_factor = 1 / (1 + volatility)
        volume_factor = min(volume_trend, 2)
        
        confidence = base_confidence * volatility_factor * volume_factor * trend_strength
        confidence = min(max(confidence, 0), 0.95)  # Cap maximum confidence at 95%
        
        # Determine forecast direction and price
        current_price = data['Close'].iloc[-1]
        price_change = current_price * 0.01 * (prediction_probability - 0.5) * 2  # Scale to ±1% range
        forecast = current_price + price_change
        
        # Only return prediction if confidence meets minimum threshold
        if confidence < 0.7:
            return None
            
        return {
            'forecast': float(forecast),
            'confidence': confidence,
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
