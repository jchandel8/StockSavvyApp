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
    """Predict price movement for multiple timeframes."""
    if len(data) < 50:
        return {
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
                'timeframe': '3+ Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
        }
    
    last_price = data['Close'].iloc[-1]
    predictions = {}
    
    # Define timeframes and their parameters
    timeframes = {
        'short_term': {'name': '1 Week', 'window': 5, 'volatility_window': 10, 'weight': 0.4},
        'medium_term': {'name': '1 Month', 'window': 20, 'volatility_window': 30, 'weight': 0.3},
        'long_term': {'name': '3+ Months', 'window': 60, 'volatility_window': 90, 'weight': 0.3}
    }
    
    for timeframe, params in timeframes.items():
        # Calculate trend strength for the specific timeframe
        trend_data = data.tail(params['window'])
        trend_strength = calculate_trend_strength(trend_data)
        
        # Calculate technical score
        technical_score = calculate_technical_score(trend_data)
        
        # Calculate volatility for the specific timeframe
        volatility_data = data.tail(params['volatility_window'])
        volatility = volatility_data['Close'].std() / volatility_data['Close'].mean()
        
        # Combine factors with timeframe-specific weights
        combined_score = (trend_strength * params['weight'] + technical_score * (1 - params['weight']))
        
        # Calculate confidence (0 to 1)
        confidence = min(abs(combined_score) / 2, 1.0)
        
        # Determine direction
        if combined_score > 0.2:
            direction = 'UP'
        elif combined_score < -0.2:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # Calculate predicted range based on timeframe volatility
        predicted_range = last_price * volatility * (params['volatility_window'] / 10)
        
        predictions[timeframe] = {
            'timeframe': params['name'],
            'direction': direction,
            'confidence': confidence,
            'predicted_high': last_price * (1 + predicted_range) if direction != 'DOWN' else last_price,
            'predicted_low': last_price * (1 - predicted_range) if direction != 'UP' else last_price
        }
    
    return predictions

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
