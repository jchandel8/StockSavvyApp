import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils.stock_data import is_crypto
from utils.technical_analysis import (
    calculate_gap_and_go_signals,
    calculate_trend_continuation,
    calculate_fibonacci_signals,
    calculate_weekly_trendline_break,
    calculate_mvrv_ratio
)

def calculate_lstm_prediction(data: pd.DataFrame, look_back: int = 60) -> dict:
    try:
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        # Prepare sequences
        x_train, y_train = [], []
        for i in range(look_back, len(scaled_data)):
            x_train.append(scaled_data[i-look_back:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build and train model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        # Make prediction
        x_test = []
        x_test.append(scaled_data[-look_back:])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        predicted_price = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        
        return {
            'lstm_forecast': float(predicted_price[0][0]),
            'confidence': 0.8
        }
    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")
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
        
        # Get LSTM predictions
        lstm_results = calculate_lstm_prediction(data)
        
        # Define timeframes
        timeframes = ['short_term', 'medium_term', 'long_term']
        periods = {'short_term': '1 Week', 'medium_term': '1 Month', 'long_term': '3 Months'}
        
        for timeframe in timeframes:
            predictions[timeframe] = {
                'timeframe': periods[timeframe],
                'direction': 'UP' if lstm_results.get('lstm_forecast', last_price) > last_price else 'DOWN',
                'confidence': lstm_results.get('confidence', 0.5),
                'predicted_high': lstm_results.get('lstm_forecast', last_price) * 1.02,
                'predicted_low': lstm_results.get('lstm_forecast', last_price) * 0.98,
                'lstm_forecast': lstm_results.get('lstm_forecast', last_price)
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