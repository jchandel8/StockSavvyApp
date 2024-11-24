import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from utils.stock_data import is_crypto
from utils.technical_analysis import (
    calculate_gap_and_go_signals,
    calculate_trend_continuation,
    calculate_fibonacci_signals,
    calculate_weekly_trendline_break,
    calculate_mvrv_ratio
)

def calculate_prediction(data: pd.DataFrame, look_back: int = 60) -> dict:
    try:
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        # Prepare sequences
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make prediction
        last_sequence = scaled_data[-look_back:].reshape(1, -1)
        predicted_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
        
        return {
            'forecast': float(predicted_price[0][0]),
            'confidence': max(min(model.score(X, y), 1.0), 0.0)
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
        # Calculate base metrics
        last_price = data['Close'].iloc[-1]
        trend_strength = calculate_trend_strength(data)
        
        # Get predictions
        prediction_results = calculate_prediction(data)
        
        # Define timeframes
        timeframes = ['short_term', 'medium_term', 'long_term']
        periods = {'short_term': '1 Week', 'medium_term': '1 Month', 'long_term': '3 Months'}
        
        for timeframe in timeframes:
            predictions[timeframe] = {
                'timeframe': periods[timeframe],
                'direction': 'UP' if prediction_results.get('forecast', last_price) > last_price else 'DOWN',
                'confidence': prediction_results.get('confidence', 0.5),
                'predicted_high': prediction_results.get('forecast', last_price) * 1.02,
                'predicted_low': prediction_results.get('forecast', last_price) * 0.98,
                'forecast': prediction_results.get('forecast', last_price)
            }
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        
    return predictions

@st.cache_data
def get_prediction(df: pd.DataFrame, ticker: str) -> dict:
    """Get price prediction with caching."""
    try:
        return predict_price_movement(df, ticker)
    except Exception as e:
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
