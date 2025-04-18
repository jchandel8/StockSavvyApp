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
    """
    Advanced multi-model ensemble prediction system that combines LSTM deep learning,
    statistical methods, and technical analysis with time series adjustments.
    """
    # Set look_back periods based on timeframe if not provided
    if look_back <= 0:
        look_back = {
            'daily': 15,           # 1 day prediction
            'short_term': 30,      # 1 week prediction
            'medium_term': 60,     # 1 month prediction
            'long_term': 90,       # 3 months prediction
            'extended_term': 120   # 6 months prediction
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
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # 1. LSTM DEEP LEARNING MODEL
        # Prepare features for deep learning
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
        lstm_model = build_lstm_model((look_back, scaled_features.shape[1]))
        lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Make LSTM prediction
        last_sequence = scaled_features[-look_back:].reshape(1, look_back, scaled_features.shape[1])
        lstm_probability = lstm_model.predict(last_sequence)[0][0]
        lstm_direction = 'UP' if lstm_probability > 0.5 else 'DOWN'
        
        # 2. STATISTICAL TIME SERIES MODELS
        # Calculate statistical predictions using various methods
        statistical_predictions = []
        
        # 2.1 Moving Average forecast
        if len(data) >= 20:
            ma_window = min(20, len(data) // 4)  # Adaptive window size
            ma_prediction = data['Close'].rolling(window=ma_window).mean().iloc[-1]
            ma_direction = 'UP' if ma_prediction > current_price else 'DOWN'
            statistical_predictions.append((ma_prediction, 0.6, ma_direction))  # Medium weight
        
        # 2.2 Exponential Smoothing
        if len(data) >= 30:
            alpha = 0.3  # Smoothing factor
            smoothed = data['Close'].ewm(alpha=alpha, adjust=False).mean()
            ema_prediction = smoothed.iloc[-1]
            ema_direction = 'UP' if ema_prediction > current_price else 'DOWN'
            statistical_predictions.append((ema_prediction, 0.7, ema_direction))  # Medium-high weight
        
        # 2.3 Linear Regression Trend
        if len(data) >= 30:
            try:
                # Use last n days for regression
                n = min(60, len(data) // 2)
                X_trend = np.array(range(n)).reshape(-1, 1)
                y_trend = data['Close'].values[-n:]
                
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_trend, y_trend)
                
                # Predict forward steps based on timeframe
                forward_steps = {
                    'daily': 1,
                    'short_term': 5,
                    'medium_term': 20,
                    'long_term': 60,
                    'extended_term': 120
                }.get(timeframe, 5)
                
                # Predict
                next_point = np.array([[n + forward_steps]]) 
                lr_prediction = model.predict(next_point)[0]
                lr_direction = 'UP' if lr_prediction > current_price else 'DOWN'
                statistical_predictions.append((lr_prediction, 0.65, lr_direction))  # Medium-high weight
            except Exception as e:
                logger.warning(f"Linear regression prediction failed: {e}")
        
        # 3. TECHNICAL INDICATOR SIGNALS
        # Use technical indicators to predict future direction
        technical_votes = []
        
        # 3.1 RSI Signal
        rsi = features['rsi'].iloc[-1]
        if rsi < 30:  # Oversold
            technical_votes.append(('UP', 0.7))  # Strong reversal signal
        elif rsi > 70:  # Overbought
            technical_votes.append(('DOWN', 0.7))  # Strong reversal signal
        elif rsi > 50:  # Momentum up
            technical_votes.append(('UP', 0.55))
        elif rsi < 50:  # Momentum down
            technical_votes.append(('DOWN', 0.55))
        
        # 3.2 MACD Signal
        macd = features['macd'].iloc[-1]
        macd_signal = features['macd_signal'].iloc[-1]
        if macd > macd_signal and macd > 0:
            technical_votes.append(('UP', 0.65))
        elif macd < macd_signal and macd < 0:
            technical_votes.append(('DOWN', 0.65))
        
        # 3.3 Bollinger Band Signal
        bb_width = features['bbands_width'].iloc[-1]
        bb_position = features['bbands_pct_b'].iloc[-1]
        if bb_position < 0.1:  # Near lower band
            technical_votes.append(('UP', 0.6))  # Potential bounce
        elif bb_position > 0.9:  # Near upper band
            technical_votes.append(('DOWN', 0.6))  # Potential pullback
        
        # 3.4 Moving Average Crossovers
        sma20 = features['sma_20'].iloc[-1]
        sma50 = features['sma_50'].iloc[-1]
        sma200 = features['sma_200'].iloc[-1] if 'sma_200' in features else None
        
        if sma20 > sma50:
            technical_votes.append(('UP', 0.6))
        elif sma20 < sma50:
            technical_votes.append(('DOWN', 0.6))
            
        if sma200 is not None:
            if current_price > sma200:  # Above 200 SMA
                technical_votes.append(('UP', 0.55))
            else:
                technical_votes.append(('DOWN', 0.55))
        
        # 4. ENSEMBLE PREDICTION CALCULATION
        # Combine all models using weighted average
        
        # 4.1 Calculate statistical consensus
        stat_price = 0
        stat_weight_sum = 0
        stat_direction_votes = {'UP': 0, 'DOWN': 0}
        
        for pred, weight, direction in statistical_predictions:
            stat_price += pred * weight
            stat_weight_sum += weight
            stat_direction_votes[direction] += weight
        
        if stat_weight_sum > 0:
            stat_price = stat_price / stat_weight_sum
            stat_direction = 'UP' if stat_direction_votes['UP'] > stat_direction_votes['DOWN'] else 'DOWN'
        else:
            stat_price = current_price
            stat_direction = 'NEUTRAL'
        
        # 4.2 Calculate technical consensus
        tech_direction_votes = {'UP': 0, 'DOWN': 0}
        tech_weight_sum = 0
        
        for direction, weight in technical_votes:
            tech_direction_votes[direction] += weight
            tech_weight_sum += weight
        
        if tech_weight_sum > 0:
            tech_direction = 'UP' if tech_direction_votes['UP'] > tech_direction_votes['DOWN'] else 'DOWN'
            tech_confidence = max(tech_direction_votes['UP'], tech_direction_votes['DOWN']) / tech_weight_sum
        else:
            tech_direction = 'NEUTRAL'
            tech_confidence = 0.5
        
        # 4.3 Combine all models with adaptive weighting based on timeframe
        
        # For shorter timeframes, give more weight to technical analysis
        # For longer timeframes, give more weight to statistical models
        lstm_weight = 0.5  # Base LSTM weight
        stat_weight = 0.3  # Base statistical weight
        tech_weight = 0.2  # Base technical weight
        
        # Adjust weights based on timeframe
        if timeframe == 'daily':
            tech_weight = 0.35
            lstm_weight = 0.4
            stat_weight = 0.25
        elif timeframe == 'short_term':
            tech_weight = 0.3
            lstm_weight = 0.4
            stat_weight = 0.3
        elif timeframe == 'medium_term':
            tech_weight = 0.25
            lstm_weight = 0.45
            stat_weight = 0.3
        elif timeframe in ['long_term', 'extended_term']:
            tech_weight = 0.15
            lstm_weight = 0.45
            stat_weight = 0.4
        
        # Calculate direction consensus
        direction_scores = {
            'UP': lstm_weight * (1 if lstm_direction == 'UP' else 0) +
                  stat_weight * (1 if stat_direction == 'UP' else 0) +
                  tech_weight * (tech_direction_votes.get('UP', 0) / tech_weight_sum if tech_weight_sum > 0 else 0),
            'DOWN': lstm_weight * (1 if lstm_direction == 'DOWN' else 0) +
                    stat_weight * (1 if stat_direction == 'DOWN' else 0) +
                    tech_weight * (tech_direction_votes.get('DOWN', 0) / tech_weight_sum if tech_weight_sum > 0 else 0)
        }
        
        final_direction = 'UP' if direction_scores['UP'] > direction_scores['DOWN'] else 'DOWN'
        
        # Calculate forecast price
        # Weighted average of all model predictions
        if stat_weight_sum > 0:
            # For LSTM, convert probability to price change
            lstm_change_factor = (lstm_probability - 0.5) * 2  # Scale to [-1, 1]
            # Adjust the price change based on timeframe
            timeframe_factors = {
                'daily': 0.01,
                'short_term': 0.03,
                'medium_term': 0.05,
                'long_term': 0.08,
                'extended_term': 0.12
            }
            lstm_price_change = current_price * lstm_change_factor * timeframe_factors.get(timeframe, 0.03)
            lstm_price = current_price + lstm_price_change
            
            # Weighted combination of all predictions
            forecast_price = (
                lstm_weight * lstm_price +
                stat_weight * stat_price +
                tech_weight * (current_price * (1.02 if tech_direction == 'UP' else 0.98))
            ) / (lstm_weight + stat_weight + tech_weight)
        else:
            # Fallback to just LSTM prediction
            lstm_change_factor = (lstm_probability - 0.5) * 2
            forecast_price = current_price * (1 + (lstm_change_factor * 0.03))
        
        # Calculate confidence score
        # Base on agreement between models and strength of signals
        model_agreement = 0
        if lstm_direction == stat_direction:
            model_agreement += 0.2
        if lstm_direction == tech_direction:
            model_agreement += 0.2
        if stat_direction == tech_direction:
            model_agreement += 0.2
        
        # LSTM model accuracy
        model_predictions = lstm_model.predict(X)
        model_accuracy = np.mean((model_predictions > 0.5) == y)
        recent_accuracy = np.mean((model_predictions[-min(10, len(y)):] > 0.5) == y[-min(10, len(y)):])
        lstm_confidence = (model_accuracy + recent_accuracy) / 2
        
        # Technical confidence based on strength of signals
        # Calculate trend alignment
        trend_aligned = (features['sma_20'].iloc[-1] > features['sma_50'].iloc[-1]) == (final_direction == 'UP')
        trend_factor = 1.2 if trend_aligned else 0.8
        
        # Calculate volume confirmation
        volume_trend = features['volume_trend'].iloc[-1]
        volume_confirms = (volume_trend > 1) == (final_direction == 'UP')
        volume_factor = 1.2 if volume_confirms else 0.8
        
        # Calculate volatility adjustment
        volatility = features['volatility_index'].iloc[-1]
        volatility_factor = 1 / (1 + volatility * 2)  # Lower confidence in high volatility
        
        # Combine all confidence factors
        base_confidence = (
            lstm_weight * lstm_confidence +
            stat_weight * 0.7 +  # Default statistical confidence
            tech_weight * tech_confidence
        ) / (lstm_weight + stat_weight + tech_weight)
        
        # Apply adjustments
        adjusted_confidence = (
            base_confidence * 
            (1 + model_agreement) * 
            trend_factor * 
            volume_factor * 
            volatility_factor
        )
        
        # Cap and scale confidence appropriately
        # For longer timeframes, reduce confidence
        timeframe_confidence_factors = {
            'daily': 1.0,
            'short_term': 0.95,
            'medium_term': 0.9,
            'long_term': 0.85,
            'extended_term': 0.8
        }
        
        final_confidence = min(adjusted_confidence * timeframe_confidence_factors.get(timeframe, 1.0), 0.95)
        
        # For very weak signals, mark as neutral with low confidence
        if final_confidence < 0.3:
            final_direction = 'NEUTRAL'
            final_confidence = 0.0
        
        return {
            'forecast': float(forecast_price),
            'confidence': float(final_confidence),
            'direction': final_direction
        }
    except Exception as e:
        logger.error(f"Error in prediction calculation: {str(e)}")
        return {
            'forecast': float(current_price),
            'confidence': 0.0,
            'direction': 'NEUTRAL'
        }

def calculate_trend_strength(data: pd.DataFrame) -> float:
    """Calculate the strength of the current trend."""
    if len(data) < 10:  # Use a smaller window for limited data
        return 0.0
        
    try:
        # Adapt window size to available data
        window_size = min(20, max(3, len(data) // 4))
        
        # Calculate price momentum using pandas operations
        returns = pd.Series(data['Close']).pct_change()
        momentum_series = returns.rolling(window=window_size).mean()
        
        # Safe handling of NaN values
        valid_momentum = pd.Series(momentum_series[pd.notna(momentum_series)])
        momentum = float(valid_momentum.iloc[-1]) if len(valid_momentum) > 0 else 0.0
        
        # Calculate trend consistency
        direction_changes = (returns.shift(-1) * returns < 0).sum()
        consistency = 1 - (direction_changes / len(returns)) if len(returns) > 0 else 0.5
        
        # Combine factors
        score = (momentum + consistency) / 2
        return max(min(abs(score), 1.0), 0.0)
    except Exception as e:
        print(f"Error in calculate_trend_strength: {str(e)}")
        return 0.0

def generate_simplified_predictions(df: pd.DataFrame) -> dict:
    """
    Generate simplified predictions for all timeframes when there is limited data.
    This uses basic technical indicators and simple statistical methods to make predictions.
    """
    predictions = {}
    
    # Ensure we have some data
    if df is None or df.empty or len(df) < 2:
        return {
            'daily': {'timeframe': '1 Day', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                     'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'short_term': {'timeframe': '1 Week', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                          'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'medium_term': {'timeframe': '1 Month', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                           'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'long_term': {'timeframe': '3 Months', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                         'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'extended_term': {'timeframe': '6 Months', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                            'predicted_high': None, 'predicted_low': None, 'forecast': None}
        }
    
    try:
        # Current price and historical data
        current_price = df['Close'].iloc[-1]
        
        # 1. Simple Moving Average Analysis
        # Adaptive window size based on available data
        sma_window = min(max(3, len(df) // 4), 10)
        sma = df['Close'].rolling(window=sma_window).mean().iloc[-1] if len(df) >= sma_window else current_price
        
        # 2. Calculate average daily return
        daily_returns = df['Close'].pct_change().dropna()
        mean_return = daily_returns.mean() if len(daily_returns) > 0 else 0
        
        # 3. Simple trend analysis
        try:
            # Use simple linear regression on the available data points
            from sklearn.linear_model import LinearRegression
            X = np.array(range(len(df))).reshape(-1, 1)
            y = df['Close'].values
            model = LinearRegression()
            model.fit(X, y)
            trend_slope = model.coef_[0]
            
            # Normalize slope to a percentage of current price
            normalized_slope = trend_slope / current_price if current_price > 0 else 0
        except Exception:
            # Fallback to simple up/down trend detection
            if len(df) >= 3:
                trend_slope = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / len(df)
                normalized_slope = trend_slope / current_price if current_price > 0 else 0
            else:
                normalized_slope = 0
        
        # 4. Simple volatility estimation
        volatility = daily_returns.std() if len(daily_returns) > 1 else 0.01
        
        # 5. Technical indicators (simplified)
        # RSI (if enough data)
        rsi = None
        if len(df) >= 6:
            try:
                up_days = daily_returns[daily_returns > 0].sum()
                down_days = abs(daily_returns[daily_returns < 0].sum())
                rs = up_days / down_days if down_days > 0 else 1
                rsi = 100 - (100 / (1 + rs))
            except Exception:
                rsi = 50  # Neutral RSI
        else:
            rsi = 50
            
        # Price relative to recent high/low
        high_low_ratio = None
        if len(df) >= 5:
            recent_high = df['High'].iloc[-5:].max()
            recent_low = df['Low'].iloc[-5:].min()
            price_range = recent_high - recent_low
            if price_range > 0:
                high_low_ratio = (current_price - recent_low) / price_range
            else:
                high_low_ratio = 0.5
        else:
            high_low_ratio = 0.5
        
        # 6. Generate predictions for each timeframe
        timeframes = [
            ('daily', '1 Day', 1),
            ('short_term', '1 Week', 5),
            ('medium_term', '1 Month', 21),
            ('long_term', '3 Months', 63),
            ('extended_term', '6 Months', 126)
        ]
        
        for tf_key, tf_name, days_ahead in timeframes:
            # Direction based on trend and indicators
            direction = 'UP' if normalized_slope > 0 else ('DOWN' if normalized_slope < 0 else 'NEUTRAL')
            
            # Adjust direction based on RSI if available
            if rsi is not None:
                if rsi < 30:  # Oversold
                    direction = 'UP'  # Potential reversal
                elif rsi > 70:  # Overbought
                    direction = 'DOWN'  # Potential reversal
                    
            # Confidence based on consistency and available data
            confidence_base = min(0.5 + (len(df) / 100), 0.7)  # Higher confidence with more data, max 0.7
            
            # Adjust confidence based on timeframe (lower for longer timeframes)
            tf_factor = 1 - (days_ahead / 200)  # Ranges from ~0.99 for daily to ~0.37 for 6-months
            confidence = confidence_base * tf_factor
            
            # Adjust confidence based on indicators
            # If RSI and trend direction align, increase confidence
            if rsi is not None:
                if (rsi > 50 and direction == 'UP') or (rsi < 50 and direction == 'DOWN'):
                    confidence *= 1.2
                else:
                    confidence *= 0.8
                    
            # Calculate price prediction using simple projection
            # Base forecast on trend with increasing uncertainty for longer timeframes
            price_change_factor = normalized_slope * days_ahead
            
            # Scale the factor for realism (can be adjusted based on volatility)
            scale_factor = min(1.0, 10 * volatility)  # Limit extreme projections
            price_change_factor *= scale_factor
            
            forecast = current_price * (1 + price_change_factor)
            
            # Calculate range based on volatility and timeframe
            # Longer timeframes have wider ranges
            volatility_factor = volatility * np.sqrt(days_ahead)
            range_width = current_price * volatility_factor * (1 + days_ahead/100)
            
            # Ensure minimum range width for visual clarity
            min_range = current_price * 0.01 * days_ahead/5
            range_width = max(range_width, min_range)
            
            predictions[tf_key] = {
                'timeframe': tf_name,
                'direction': direction,
                'confidence': min(confidence, 0.9),  # Cap at 0.9
                'predicted_high': forecast + range_width,
                'predicted_low': forecast - range_width,
                'forecast': forecast
            }
            
        return predictions
        
    except Exception as e:
        logger.error(f"Error in simplified prediction: {str(e)}")
        
        # Return default values with neutral prediction
        return {
            'daily': {'timeframe': '1 Day', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                     'predicted_high': current_price * 1.01, 'predicted_low': current_price * 0.99, 
                     'forecast': current_price},
            'short_term': {'timeframe': '1 Week', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                          'predicted_high': current_price * 1.03, 'predicted_low': current_price * 0.97,
                          'forecast': current_price},
            'medium_term': {'timeframe': '1 Month', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                           'predicted_high': current_price * 1.05, 'predicted_low': current_price * 0.95,
                           'forecast': current_price},
            'long_term': {'timeframe': '3 Months', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                         'predicted_high': current_price * 1.08, 'predicted_low': current_price * 0.92,
                         'forecast': current_price},
            'extended_term': {'timeframe': '6 Months', 'direction': 'NEUTRAL', 'confidence': 0.5, 
                            'predicted_high': current_price * 1.12, 'predicted_low': current_price * 0.88,
                            'forecast': current_price}
        }

def predict_price_movement(data: pd.DataFrame, ticker: str) -> dict:
    """Generate price predictions for multiple timeframes using ensemble of advanced models."""
    if data is None or len(data) < 50:
        return {
            'daily': {'timeframe': '1 Day', 'direction': 'NEUTRAL', 'confidence': 0.0,
                    'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'short_term': {'timeframe': '1 Week', 'direction': 'NEUTRAL', 'confidence': 0.0,
                          'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'medium_term': {'timeframe': '1 Month', 'direction': 'NEUTRAL', 'confidence': 0.0,
                           'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'long_term': {'timeframe': '3 Months', 'direction': 'NEUTRAL', 'confidence': 0.0,
                         'predicted_high': None, 'predicted_low': None, 'forecast': None},
            'extended_term': {'timeframe': '6 Months', 'direction': 'NEUTRAL', 'confidence': 0.0,
                            'predicted_high': None, 'predicted_low': None, 'forecast': None}
        }
    
    predictions = {}
    try:
        last_price = data['Close'].iloc[-1]
        trend_strength = calculate_trend_strength(data)
        
        # Enhanced timeframes with adjusted look-back periods
        timeframes = ['daily', 'short_term', 'medium_term', 'long_term', 'extended_term']
        periods = {
            'daily': '1 Day',
            'short_term': '1 Week',
            'medium_term': '1 Month',
            'long_term': '3 Months',
            'extended_term': '6 Months'
        }
        
        # Adjust look-back periods for different timeframes
        look_back_periods = {
            'daily': 15,        # For 1-day prediction
            'short_term': 30,   # For 1-week prediction
            'medium_term': 60,  # For 1-month prediction
            'long_term': 90,    # For 3-month prediction
            'extended_term': 120 # For 6-month prediction
        }
        
        # Calculate seasonal patterns (if enough data)
        seasonal_factors = {}
        if len(data) >= 365:  # At least a year of data
            try:
                # Calculate month-of-year seasonality
                data['month'] = pd.DatetimeIndex(data.index).month
                monthly_returns = data.groupby('month')['Close'].pct_change().mean()
                
                # Get current month and calculate expected seasonal effect
                current_month = pd.DatetimeIndex([pd.Timestamp.now()])[0].month
                next_months = [(current_month + i - 1) % 12 + 1 for i in range(1, 7)]  # Next 6 months
                
                for i, timeframe in enumerate(['daily', 'short_term', 'medium_term', 'long_term', 'extended_term']):
                    # For daily, use current month seasonality
                    if timeframe == 'daily':
                        seasonal_factors[timeframe] = monthly_returns.get(current_month, 0)
                    # For others, use average of relevant future months
                    else:
                        months_to_consider = next_months[:i+1]  # More months for longer timeframes
                        seasonal_impact = sum(monthly_returns.get(m, 0) for m in months_to_consider) / len(months_to_consider)
                        seasonal_factors[timeframe] = seasonal_impact
            except Exception as e:
                logger.warning(f"Error calculating seasonality: {str(e)}")
                # Set neutral seasonal factors if calculation fails
                for timeframe in timeframes:
                    seasonal_factors[timeframe] = 0
        else:
            # Not enough data for seasonal analysis
            for timeframe in timeframes:
                seasonal_factors[timeframe] = 0
        
        # Market regime factors (bull/bear/sideways)
        market_regime = 'neutral'
        regime_factor = 1.0
        
        # Detect current market regime
        if len(data) > 50:
            sma_200 = data['Close'].rolling(window=min(200, len(data)//2)).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=min(50, len(data)//5)).mean().iloc[-1]
            recent_volatility = data['Close'].pct_change().rolling(window=20).std().iloc[-1]
            historic_volatility = data['Close'].pct_change().rolling(window=min(100, len(data)//3)).std().iloc[-1]
            
            # Bull market: price above SMAs, recent volatility lower than historic
            if data['Close'].iloc[-1] > sma_200 and data['Close'].iloc[-1] > sma_50 and sma_50 > sma_200:
                market_regime = 'bull'
                regime_factor = 1.2  # More aggressive upside predictions
            # Bear market: price below SMAs, recent volatility higher than historic
            elif data['Close'].iloc[-1] < sma_200 and data['Close'].iloc[-1] < sma_50 and sma_50 < sma_200:
                market_regime = 'bear'
                regime_factor = 0.8  # More conservative/negative predictions
            # Else neutral/sideways
        
        for timeframe in timeframes:
            # Get prediction for current timeframe
            look_back = look_back_periods[timeframe]
            prediction_results = calculate_prediction(data, timeframe=timeframe, look_back=look_back)
            
            # Skip if prediction doesn't meet confidence threshold
            if not prediction_results:
                continue
                
            forecast = prediction_results.get('forecast', last_price)
            confidence = prediction_results.get('confidence', 0.5)
            direction = prediction_results.get('direction', 'NEUTRAL')
            
            # Apply seasonal adjustments
            seasonal_impact = seasonal_factors.get(timeframe, 0)
            forecast = forecast * (1 + seasonal_impact * 0.2)  # Scale seasonal effect
            
            # Apply market regime adjustments
            if market_regime == 'bull' and direction == 'UP':
                confidence = min(confidence * 1.2, 0.95)  # Increase confidence in upward predictions in bull markets
            elif market_regime == 'bear' and direction == 'DOWN':
                confidence = min(confidence * 1.2, 0.95)  # Increase confidence in downward predictions in bear markets
            
            # Calculate volatility-adjusted range with improved factors
            volatility = data['Close'].rolling(window=min(20, len(data)//3)).std().iloc[-1] / last_price
            
            # Range factors increase with time horizon and are adjusted for market regime
            base_range_factors = {
                'daily': 0.01,
                'short_term': 0.03,
                'medium_term': 0.05,
                'long_term': 0.08,
                'extended_term': 0.12
            }
            
            # Adjust range based on market regime
            volatility_multiplier = 1.5 if market_regime == 'bear' else (0.8 if market_regime == 'bull' else 1.0)
            range_factor = base_range_factors[timeframe] * regime_factor
            predicted_range = last_price * range_factor * (1 + volatility * volatility_multiplier)
            
            # Ensure predictions make sense for the timeframe
            # Longer timeframes should have wider ranges
            if timeframe in ['long_term', 'extended_term'] and predicted_range < last_price * 0.05:
                predicted_range = last_price * 0.05  # Minimum 5% range for long term predictions
            
            # Construct final prediction
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
                'predicted_low': None,
                'forecast': None
            },
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'medium_term': {
                'timeframe': '1 Month',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'long_term': {
                'timeframe': '3 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'extended_term': {
                'timeframe': '6 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            }
        }
    
    # Check if we have minimum required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        logger.warning(f"Missing required columns for prediction: {[col for col in required_columns if col not in df.columns]}")
        return {
            'daily': {
                'timeframe': '1 Day',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'medium_term': {
                'timeframe': '1 Month',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'long_term': {
                'timeframe': '3 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'extended_term': {
                'timeframe': '6 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            }
        }
    
    # Check for sufficient data length - but use what we have with fallback methods
    if len(df) < 50:  # Not enough data for full model, use simplified prediction
        logger.warning(f"Limited data length for prediction: {len(df)} data points - using simplified model")
        return generate_simplified_predictions(df)
    
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
            'daily': {
                'timeframe': '1 Day',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'medium_term': {
                'timeframe': '1 Month',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'long_term': {
                'timeframe': '3 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            },
            'extended_term': {
                'timeframe': '6 Months',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None,
                'forecast': None
            }
        }
