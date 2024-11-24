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

def calculate_prediction(data: pd.DataFrame, timeframe: str = 'short_term', look_back: int = None) -> dict:
    # Set look_back periods based on timeframe
    if look_back is None:
        look_back = {
            'short_term': 30,    # 1 week prediction
            'medium_term': 60,   # 1 month prediction
            'long_term': 90      # 3 months prediction
        }.get(timeframe, 30)
    
    try:
        # Prepare data with specific look_back
        prices = np.array(data['Close']).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        # Prepare sequences with timeframe-specific look_back
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
        
        # Calculate volatility factor based on timeframe
        volatility = np.std(data['Close'].pct_change().dropna())
        volatility_factor = {
            'short_term': 1.0,
            'medium_term': 1.5,
            'long_term': 2.0
        }.get(timeframe, 1.0)
        
        # Adjust confidence based on timeframe and volatility
        base_confidence = max(min(model.score(X, y), 1.0), 0.0)
        adjusted_confidence = base_confidence * (1 / (1 + volatility * volatility_factor))
        
        return {
            'forecast': float(predicted_price[0][0]),
            'confidence': adjusted_confidence
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
        last_price = data['Close'].iloc[-1]
        trend_strength = calculate_trend_strength(data)
        
        timeframes = ['short_term', 'medium_term', 'long_term']
        periods = {'short_term': '1 Week', 'medium_term': '1 Month', 'long_term': '3 Months'}
        
        for timeframe in timeframes:
            # Get timeframe-specific prediction
            prediction_results = calculate_prediction(data, timeframe=timeframe)
            
            # Calculate volatility adjustment
            volatility = np.std(data['Close'].pct_change().dropna())
            volatility_factor = {'short_term': 1.0, 'medium_term': 1.5, 'long_term': 2.0}[timeframe]
            
            forecast = prediction_results.get('forecast', last_price)
            confidence = prediction_results.get('confidence', 0.5)
            
            # Adjust range based on timeframe
            range_factor = {'short_term': 0.02, 'medium_term': 0.04, 'long_term': 0.06}[timeframe]
            predicted_range = forecast * range_factor * (1 + volatility)
            
            predictions[timeframe] = {
                'timeframe': periods[timeframe],
                'direction': 'UP' if forecast > last_price else 'DOWN',
                'confidence': confidence,
                'predicted_high': forecast + predicted_range,
                'predicted_low': forecast - predicted_range,
                'forecast': forecast
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
