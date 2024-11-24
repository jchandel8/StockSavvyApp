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

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    # Calculate price changes
    delta = prices.diff()
    
    # Get gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series) -> tuple:
    # Calculate EMAs
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    
    # Calculate MACD and Signal Line
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> tuple:
    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = prices.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * 2)
    lower_band = middle_band - (std * 2)
    
    return upper_band, lower_band

def calculate_atr(data, period=14):
    """Calculate Average True Range."""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def build_lstm_model(X, y):
    """Build and train LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    return model

def calculate_prediction(data: pd.DataFrame, timeframe: str = 'short_term', look_back: int = None) -> dict:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        from sklearn.svm import SVR
        
        # Feature Engineering
        features = pd.DataFrame()
        
        # Technical Indicators
        features['rsi'] = calculate_rsi(data['Close'])
        features['macd'], features['signal'] = calculate_macd(data['Close'])
        features['bb_upper'], features['bb_lower'] = calculate_bollinger_bands(data['Close'])
        features['atr'] = calculate_atr(data[['High', 'Low', 'Close']])
        
        # Volume Analysis
        features['volume_sma'] = data['Volume'].rolling(window=20).mean()
        features['volume_std'] = data['Volume'].rolling(window=20).std()
        
        # Price Momentum
        features['price_momentum'] = data['Close'].pct_change(periods=5)
        features['price_acceleration'] = features['price_momentum'].diff()
        
        # Volatility
        features['volatility'] = data['Close'].pct_change().rolling(window=20).std()
        
        # Market Trend
        features['trend_strength'] = calculate_trend_strength(data)
        
        # Prepare data for models
        X = features.dropna().values
        y = data['Close'].shift(-1).dropna().values[:-1]
        
        if len(X) < 2 or len(y) < 2:
            return {}
        
        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(objective='reg:squarederror', random_state=42),
            'svr': SVR(kernel='rbf')
        }
        
        # Train and make predictions
        predictions = {}
        weights = {'rf': 0.4, 'xgb': 0.4, 'svr': 0.2}
        
        for name, model in models.items():
            try:
                model.fit(X[:-1], y)
                pred = model.predict(X[-1:])
                predictions[name] = pred[0]
            except Exception as e:
                st.error(f"Error in {name} model: {str(e)}")
                continue
        
        if not predictions:
            return {}
        
        # Weighted ensemble prediction
        final_prediction = sum(predictions[model] * weight 
                             for model, weight in weights.items()
                             if model in predictions)
        
        # Calculate confidence based on model agreement
        std_predictions = np.std(list(predictions.values()))
        max_std = np.std(data['Close'])
        confidence = 1 - (std_predictions / max_std) if max_std > 0 else 0.5
        
        return {
            'forecast': final_prediction,
            'confidence': confidence,
            'model_predictions': predictions
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
        
        timeframes = ['daily', 'short_term', 'medium_term', 'long_term']
        periods = {
            'daily': '1 Day',
            'short_term': '1 Week',
            'medium_term': '1 Month',
            'long_term': '3 Months'
        }
        
        for timeframe in timeframes:
            # Get timeframe-specific prediction
            prediction_results = calculate_prediction(data, timeframe=timeframe)
            
            # Calculate volatility adjustment
            volatility = np.std(data['Close'].pct_change().dropna())
            volatility_factor = {
                'daily': 0.5,
                'short_term': 1.0,
                'medium_term': 1.5,
                'long_term': 2.0
            }[timeframe]
            
            forecast = prediction_results.get('forecast', last_price)
            confidence = prediction_results.get('confidence', 0.5)
            
            # Adjust range based on timeframe
            range_factor = {
                'daily': 0.01,
                'short_term': 0.02,
                'medium_term': 0.04,
                'long_term': 0.06
            }[timeframe]
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
