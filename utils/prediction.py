import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
from typing import Dict, List, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from utils.stock_data import is_crypto
from utils.technical_analysis import (
    calculate_gap_and_go_signals,
    calculate_trend_continuation,
    calculate_fibonacci_signals,
    calculate_weekly_trendline_break,
    calculate_mvrv_ratio
)

def calculate_arima_prediction(data: pd.DataFrame, is_crypto: bool = False, steps: int = 30) -> dict:
    try:
        # Prepare data
        prices = data['Close'].values
        
        # Different parameters for crypto vs stocks
        if is_crypto:
            # Use shorter window for crypto due to higher volatility
            prices = prices[-500:]  # Use last 500 data points
            p_values = range(0, 3)  # Reduced AR terms
            q_values = range(0, 2)  # Reduced MA terms
        else:
            p_values = range(0, 6)
            q_values = range(0, 4)
        
        # Perform Augmented Dickey-Fuller test for stationarity
        adf_result = adfuller(prices)
        d = 1 if adf_result[1] > 0.05 else 0
        
        best_aic = float('inf')
        best_params = (1, d, 1)
        
        # Grid search with more stable parameters for crypto
        for p in p_values:
            for q in q_values:
                try:
                    model = ARIMA(prices, order=(p, d, q))
                    model_fit = model.fit(method='css' if is_crypto else 'mle')  # Use CSS method for crypto
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_params = (p, d, q)
                except:
                    continue
        
        # Fit final model with best parameters
        final_model = ARIMA(prices, order=best_params)
        final_fit = final_model.fit(method='css' if is_crypto else 'mle')
        
        # Make predictions
        forecast = final_fit.forecast(steps=steps)
        confidence_intervals = final_fit.get_forecast(steps=steps).conf_int()
        
        return {
            'forecast': forecast,
            'lower_bound': confidence_intervals[:,0],
            'upper_bound': confidence_intervals[:,1],
            'model_aic': final_fit.aic,
            'model_order': best_params
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return {}

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

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

def analyze_fundamental_factors(ticker: str) -> float:
    """Analyze fundamental factors for price prediction."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Financial health score
        score = 0
        
        # Revenue growth
        if 'revenueGrowth' in info and info['revenueGrowth'] > 0:
            score += info['revenueGrowth']
            
        # Profit margins
        if 'profitMargins' in info and info['profitMargins'] > 0:
            score += info['profitMargins']
            
        # Debt to equity
        if 'debtToEquity' in info and info['debtToEquity'] < 100:
            score += 0.5
            
        # Return on equity
        if 'returnOnEquity' in info and info['returnOnEquity'] > 0:
            score += info['returnOnEquity']
            
        return min(score, 1.0)  # Normalize to 0-1
    except:
        return 0.5  # Neutral score on error

def analyze_market_cycle(data: pd.DataFrame) -> float:
    """Analyze market cycle based on price patterns and momentum."""
    try:
        # Calculate momentum indicators using pandas operations
        price_series = pd.Series(data['Close'])
        returns = price_series.pct_change()
        momentum = returns.rolling(window=20).mean()
        volatility = returns.rolling(window=20).std()
        
        # Safe handling of latest values
        valid_momentum = momentum[pd.notna(momentum)]
        valid_volatility = volatility[pd.notna(volatility)]
        
        latest_momentum = float(pd.Series(valid_momentum).iloc[-1]) if len(valid_momentum) > 0 else 0.0
        latest_volatility = float(pd.Series(valid_volatility).iloc[-1]) if len(valid_volatility) > 0 else 0.0
        volatility_mean = float(valid_volatility.mean()) if len(valid_volatility) > 0 else 0.0
        
        if latest_momentum > 0 and latest_volatility < volatility_mean:
            return 0.8  # Upward trend with low volatility
        elif latest_momentum > 0:
            return 0.6  # Upward trend with high volatility
        elif latest_momentum < 0 and latest_volatility > volatility_mean:
            return 0.2  # Downward trend with high volatility
        else:
            return 0.4  # Downward trend with low volatility
    except:
        return 0.5

def analyze_volume_profile(data: pd.DataFrame) -> float:
    try:
        if len(data) < 10:
            return 0.5
            
        # Calculate volume-weighted price levels
        close_prices = np.array(data['Close'].values)
        volumes = np.array(data['Volume'].values)
        
        # Create price bins (10 equal-sized bins)
        min_price = float(np.min(close_prices))
        max_price = float(np.max(close_prices))
        price_range = np.linspace(min_price, max_price, 11)
        current_price = float(close_prices[-1])
        
        # Find which bin contains the current price
        current_bin = np.digitize(current_price, price_range) - 1
        
        # Calculate volume profile
        volume_profile = np.zeros(10)
        for i in range(len(close_prices)):
            bin_idx = np.digitize(close_prices[i], price_range) - 1
            if 0 <= bin_idx < 10:  # Ensure valid bin index
                volume_profile[bin_idx] += volumes[i]
                
        # Get volume for current price level
        current_level_volume = volume_profile[current_bin] if 0 <= current_bin < 10 else np.mean(volume_profile)
        mean_volume = np.mean(volume_profile[volume_profile > 0])  # Consider only bins with volume
        
        # Calculate ratio
        volume_ratio = current_level_volume / mean_volume if mean_volume > 0 else 1.0
        return min(float(volume_ratio), 1.0)
    except Exception as e:
        print(f"Error in analyze_volume_profile: {str(e)}")
        return 0.5

def calculate_support_resistance(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate support and resistance levels."""
    try:
        # Use price pivots to identify support/resistance
        high = data['High'].iloc[-20:]
        low = data['Low'].iloc[-20:]
        close = data['Close'].iloc[-20:]
        
        resistance = max(high)
        support = min(low)
        
        return {
            'support': support,
            'resistance': resistance
        }
    except:
        current_price = data['Close'].iloc[-1]
        return {
            'support': current_price * 0.95,
            'resistance': current_price * 1.05
        }

def calculate_dynamic_range(last_price: float, volatility: float, 
                          support_resistance: Dict[str, float], 
                          window: int) -> float:
    """Calculate dynamic price range based on volatility and S/R levels."""
    base_range = volatility * window / 10
    
    # Adjust range based on proximity to support/resistance
    support_distance = abs(last_price - support_resistance['support']) / last_price
    resistance_distance = abs(support_resistance['resistance'] - last_price) / last_price
    
    # Reduce range if near support/resistance
    if min(support_distance, resistance_distance) < 0.02:
        base_range *= 0.5
        
    return base_range

def determine_direction(score: float, is_crypto: bool = False) -> str:
    """Determine price direction based on combined score with crypto-specific thresholds."""
    if is_crypto:
        if score > 0.60:  # Higher threshold for crypto
            return 'UP'
        elif score < 0.40:  # Lower threshold for crypto
            return 'DOWN'
    else:
        if score > 0.55:
            return 'UP'
        elif score < 0.45:
            return 'DOWN'
    return 'NEUTRAL'

def calculate_confidence(score: float, volatility: float) -> float:
    """Calculate prediction confidence."""
    if pd.isna(score) or pd.isna(volatility):
        return 0.1  # Return minimum confidence instead of NaN
        
    base_confidence = abs(score - 0.5) * 2.5  # Increased sensitivity
    volatility_factor = max(1 - volatility * 1.5, 0)  # Less penalty for volatility
    confidence = min(base_confidence * volatility_factor, 1.0)
    return max(confidence, 0.1)  # Minimum confidence of 10%

def analyze_economic_factors(ticker: str) -> float:
    """Analyze economic factors for a given stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        score = 0.5  # Start neutral
        
        # Sector performance
        if 'sector' in info:
            sector_perf = stock.info.get('sector_performance', 0)
            score += sector_perf * 0.2
        
        # Market cap stability
        if 'marketCap' in info and 'enterpriseValue' in info:
            market_ratio = info['marketCap'] / info['enterpriseValue']
            if 0.8 <= market_ratio <= 1.2:
                score += 0.1
        
        # Institutional ownership
        if 'institutionalOwnership' in info:
            inst_own = info['institutionalOwnership']
            score += min(inst_own * 0.3, 0.2)
        
        return min(max(score, 0), 1)  # Normalize to 0-1
    except:
        return 0.5

def calculate_target(last_price: float, predicted_range: float, 
                    target_type: str, support_resistance: Dict[str, float]) -> float:
    """Calculate price targets considering support/resistance."""
    if target_type == 'high':
        raw_target = last_price * (1 + predicted_range)
        return min(raw_target, support_resistance['resistance'])
    else:
        raw_target = last_price * (1 - predicted_range)
        return max(raw_target, support_resistance['support'])

def predict_price_movement(data: pd.DataFrame, ticker: str) -> dict:
    """Predict price movement for multiple timeframes with enhanced analysis."""
    if data is None or len(data) < 50:
        return {
            'short_term': {'timeframe': '1 Week', 'direction': 'NEUTRAL', 'confidence': 0.0,
                          'predicted_high': None, 'predicted_low': None},
            'medium_term': {'timeframe': '1 Month', 'direction': 'NEUTRAL', 'confidence': 0.0,
                           'predicted_high': None, 'predicted_low': None},
            'long_term': {'timeframe': '3 Months', 'direction': 'NEUTRAL', 'confidence': 0.0,
                         'predicted_high': None, 'predicted_low': None}
        }
    
    # Calculate base metrics
    last_price = data['Close'].iloc[-1]
    trend_strength = calculate_trend_strength(data)
    predictions = {}
    
    # Add new factors
    fundamental_score = analyze_fundamental_factors(ticker)
    market_cycle_score = analyze_market_cycle(data)
    volume_profile_score = analyze_volume_profile(data)
    support_resistance = calculate_support_resistance(data)
    
    # Define timeframes and their parameters based on asset type
    timeframes = {
        'short_term': {
            'name': '24 Hours',
            'window': min(24, len(data)),
            'volatility_window': min(48, len(data)),
            'weight': 0.4
        },
        'medium_term': {
            'name': '7 Days',
            'window': min(168, len(data)),
            'volatility_window': min(336, len(data)),
            'weight': 0.3
        },
        'long_term': {
            'name': '30 Days',
            'window': min(720, len(data)),
            'volatility_window': min(1440, len(data)),
            'weight': 0.3
        }
    } if is_crypto(ticker) else {
        'short_term': {
            'name': '1 Week',
            'window': min(5, len(data)),
            'volatility_window': min(10, len(data)),
            'weight': 0.4
        },
        'medium_term': {
            'name': '1 Month',
            'window': min(20, len(data)),
            'volatility_window': min(30, len(data)),
            'weight': 0.3
        },
        'long_term': {
            'name': '3 Months',
            'window': min(60, len(data)),
            'volatility_window': min(60, len(data)),
            'weight': 0.3
        }
    }
    
    for timeframe, params in timeframes.items():
        # Before calculating volatility, check for valid data
        window_data = data.tail(params['window'])
        if len(window_data) < params['window']:
            continue

        technical_score = calculate_trend_strength(window_data)
        
        # Before volatility calculation, ensure we have enough data
        volatility_data = data.tail(params['volatility_window'])
        if len(volatility_data) < params['volatility_window']:
            continue

        # Add null check for volatility calculation
        volatility = volatility_data['Close'].std() / volatility_data['Close'].mean()
        if pd.isna(volatility):
            volatility = data['Close'].std() / data['Close'].mean()  # Use full dataset if window fails
        
        # Calculate additional signals
        gap_and_go = calculate_gap_and_go_signals(window_data).iloc[-1]
        trend_continuation = calculate_trend_continuation(window_data).iloc[-1]
        fibonacci = calculate_fibonacci_signals(window_data).iloc[-1]
        weekly_trendline = calculate_weekly_trendline_break(window_data).iloc[-1]
        
        # Count agreeing signals
        signal_agreement = sum([
            gap_and_go,
            trend_continuation,
            fibonacci,
            weekly_trendline
        ]) / 4.0  # Normalize to 0-1
        
        # Calculate economic factors
        economic_score = analyze_economic_factors(ticker)
        
        # Combine all factors with updated weights based on asset type
        if is_crypto(ticker):
            combined_score = (
                trend_strength * 0.25 +                # Technical trend (increased weight)
                technical_score * 0.25 +               # Price momentum (increased weight)
                market_cycle_score * 0.20 +            # Market conditions
                volume_profile_score * 0.15 +          # Volume analysis
                signal_agreement * 0.15                # Technical signals
            )
            
            # Add MVRV ratio adjustment for crypto
            mvrv_ratio = calculate_mvrv_ratio(data).iloc[-1]
            if not pd.isna(mvrv_ratio):
                if mvrv_ratio > 3.0:  # Overvalued
                    combined_score *= 0.8
                elif mvrv_ratio < 1.0:  # Undervalued
                    combined_score *= 1.2
        else:
            combined_score = (
                trend_strength * 0.20 +                # Technical trend
                technical_score * 0.20 +               # Price momentum
                fundamental_score * 0.20 +             # Company health
                market_cycle_score * 0.15 +            # Market conditions
                volume_profile_score * 0.10 +          # Volume analysis
                signal_agreement * 0.15                # Technical signals
            )
        
        # Add economic factor adjustment
        if fundamental_score > 0.7:  # Strong company fundamentals
            combined_score *= 1.1    # Boost prediction confidence
        elif fundamental_score < 0.3:  # Weak fundamentals
            combined_score *= 0.9    # Reduce prediction confidence
        
        # Adjust predicted range based on support/resistance
        predicted_range = calculate_dynamic_range(
            last_price,
            volatility,
            support_resistance,
            params['volatility_window']
        )
        
        # Update predictions dictionary
        predictions[timeframe] = {
            'timeframe': params['name'],
            'direction': determine_direction(combined_score),
            'confidence': calculate_confidence(combined_score, volatility),
            'predicted_high': calculate_target(last_price, predicted_range, 'high', support_resistance),
            'predicted_low': calculate_target(last_price, predicted_range, 'low', support_resistance)
        }
    
    # Add LSTM predictions
    try:
        lstm_results = calculate_lstm_prediction(data, is_crypto=is_crypto(ticker))
        if lstm_results:
            for timeframe in predictions:
                predictions[timeframe].update(lstm_results)
    except Exception as e:
        st.error(f"Error adding LSTM predictions: {str(e)}")

    return predictions
        if arima_results:
            for timeframe in predictions:
                predictions[timeframe].update({
                    'arima_forecast': arima_results['arima_forecast'],
                    'arima_lower': arima_results['arima_forecast'] * 0.95,
                    'arima_upper': arima_results['arima_forecast'] * 1.05,
                    'model_order': arima_results['model_order']
                })
    except Exception as e:
        st.error(f"Error adding ARIMA predictions: {str(e)}")
        
    return predictions
        try:
            arima_results = calculate_arima_prediction(data)
            if arima_results:
                for timeframe in predictions:
        except Exception as e:
            st.error(f"Error in ARIMA prediction: {str(e)}")
            arima_results = {}
                tf_params = timeframes[timeframe]
                forecast_idx = min(tf_params['window'], len(arima_results['forecast'])-1)
                
                # Adjust predictions using ARIMA forecast
                arima_direction = 'UP' if arima_results['forecast'][forecast_idx] > last_price else 'DOWN'
                if arima_direction == predictions[timeframe]['direction']:
                    predictions[timeframe]['confidence'] *= 1.2  # Boost confidence if ARIMA agrees
                
                # Update predicted high/low with ARIMA bounds
                predictions[timeframe]['arima_forecast'] = float(arima_results['forecast'][forecast_idx])
                predictions[timeframe]['arima_lower'] = float(arima_results['lower_bound'][forecast_idx])
                predictions[timeframe]['arima_upper'] = float(arima_results['upper_bound'][forecast_idx])
    except Exception as e:
        st.error(f"Error incorporating ARIMA predictions: {str(e)}")
    
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
