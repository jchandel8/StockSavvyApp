import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def calculate_trend_strength(data: pd.DataFrame) -> float:
    """Calculate the strength of the current trend."""
    # Use recent price movement and volume
    recent_data = data.tail(20)
    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
    volume_change = recent_data['Volume'].mean() / data['Volume'].mean()
    
    return price_change * volume_change

def calculate_technical_score(data: pd.DataFrame) -> float:
    """Calculate a technical analysis score."""
    latest = data.iloc[-1]
    
    score = 0
    # RSI
    if latest['RSI'] < 30:
        score += 1
    elif latest['RSI'] > 70:
        score -= 1
    
    # MACD
    if latest['MACD'] > latest['MACD_Signal']:
        score += 1
    else:
        score -= 1
    
    # Moving Averages
    if latest['Close'] > latest['SMA_20']:
        score += 0.5
    if latest['Close'] > latest['SMA_50']:
        score += 0.5
    
    return score

def predict_price_movement(data: pd.DataFrame) -> dict:
    """Predict price movement for the next trading day."""
    if len(data) < 50:
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'predicted_high': None,
            'predicted_low': None
        }
    
    # Calculate various factors
    trend_strength = calculate_trend_strength(data)
    technical_score = calculate_technical_score(data)
    
    # Recent volatility
    recent_volatility = data['Close'].tail(20).std() / data['Close'].tail(20).mean()
    
    # Combine factors for final prediction
    combined_score = (trend_strength + technical_score) / 2
    
    # Calculate confidence (0 to 1)
    confidence = min(abs(combined_score) / 2, 1.0)
    
    # Determine direction
    if combined_score > 0.2:
        direction = 'UP'
    elif combined_score < -0.2:
        direction = 'DOWN'
    else:
        direction = 'NEUTRAL'
    
    # Predict price range
    last_price = data['Close'].iloc[-1]
    predicted_range = last_price * recent_volatility
    
    return {
        'direction': direction,
        'confidence': confidence,
        'predicted_high': last_price * (1 + predicted_range) if direction != 'DOWN' else last_price,
        'predicted_low': last_price * (1 - predicted_range) if direction != 'UP' else last_price
    }

@st.cache_data
def get_prediction(df: pd.DataFrame) -> dict:
    """Get price prediction with caching."""
    try:
        return predict_price_movement(df)
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'predicted_high': None,
            'predicted_low': None
        }
