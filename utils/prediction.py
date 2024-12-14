import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction model."""
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['close'] = df['Close']
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    
    # Volume features
    features['volume'] = df['Volume']
    features['volume_ma'] = df['Volume'].rolling(window=20).mean()
    features['volume_std'] = df['Volume'].rolling(window=20).std()
    
    # Technical indicators
    features['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    features['macd'] = ta.trend.MACD(df['Close']).macd()
    features['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
    features['macd_diff'] = ta.trend.MACD(df['Close']).macd_diff()
    
    # Moving averages
    features['sma_20'] = df['Close'].rolling(window=20).mean()
    features['sma_50'] = df['Close'].rolling(window=50).mean()
    features['ema_20'] = df['Close'].ewm(span=20).mean()
    
    # Volatility
    features['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    features['bbands_width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()
    
    # Fill NaN values with 0
    features = features.fillna(0)
    
    return features

def create_sequences(features: pd.DataFrame, look_back: int = 10) -> tuple:
    """Create sequences for training the model."""
    X, y = [], []
    
    for i in range(look_back, len(features)):
        # Get the sequence of features
        sequence = features.iloc[i-look_back:i].values
        X.append(sequence.flatten())  # Flatten the sequence for RandomForest
        
        # Target is 1 if price goes up, 0 if down
        current_price = features['close'].iloc[i-1]
        next_price = features['close'].iloc[i]
        y.append(1 if next_price > current_price else 0)
    
    return np.array(X), np.array(y)

def calculate_prediction(df: pd.DataFrame, timeframe: str = 'daily') -> dict:
    """Calculate predictions for the given timeframe."""
    try:
        # Set lookback period based on timeframe
        look_back = {
            'daily': 10,
            'short_term': 20,
            'medium_term': 40,
            'long_term': 60
        }.get(timeframe, 10)
        
        # Prepare features
        features_df = prepare_features(df)
        
        # Create sequences
        X, y = create_sequences(features_df, look_back)
        
        if len(X) == 0:
            logger.warning("Not enough data for prediction")
            return None
            
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Prepare last sequence for prediction
        last_sequence = features_df.iloc[-look_back:].values.flatten().reshape(1, -1)
        last_sequence_scaled = scaler.transform(last_sequence)
        
        # Make prediction
        prediction_prob = model.predict_proba(last_sequence_scaled)[0][1]
        
        # Calculate confidence based on model metrics
        confidence = min(max(prediction_prob, 0.5), 0.95)
        
        # Calculate predicted price change
        current_price = df['Close'].iloc[-1]
        avg_daily_change = df['Close'].pct_change().mean()
        predicted_change = avg_daily_change * (prediction_prob - 0.5) * 2
        
        forecast = current_price * (1 + predicted_change)
        
        return {
            'forecast': float(forecast),
            'confidence': float(confidence),
            'direction': 'UP' if prediction_prob > 0.5 else 'DOWN'
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return None

@st.cache_data
def get_prediction(df: pd.DataFrame, ticker: str) -> dict:
    """Get cached predictions for multiple timeframes."""
    try:
        predictions = {}
        timeframes = {
            'daily': '1 Day',
            'short_term': '1 Week',
            'medium_term': '1 Month',
            'long_term': '3 Months'
        }
        
        for timeframe, label in timeframes.items():
            pred = calculate_prediction(df, timeframe)
            if pred:
                predictions[timeframe] = {
                    'timeframe': label,
                    'direction': pred['direction'],
                    'confidence': pred['confidence'],
                    'predicted_high': pred['forecast'] * 1.02,  # 2% above forecast
                    'predicted_low': pred['forecast'] * 0.98,   # 2% below forecast
                    'forecast': pred['forecast']
                }
                
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}")
        return {
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
        }