import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
from typing import Dict, List, Tuple
from utils.technical_analysis import (
    calculate_gap_and_go_signals,
    calculate_trend_continuation,
    calculate_fibonacci_signals,
    calculate_weekly_trendline_break
)

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
    """Analyze volume profile for price levels."""
    try:
        # Calculate volume-weighted price levels
        close_series = pd.Series(data['Close'].values)
        volume_series = pd.Series(data['Volume'].values)
        
        # Create price bins
        bins = pd.qcut(close_series, q=10, duplicates='drop')
        volume_profile = volume_series.groupby(bins).mean()
        
        # Get the current price level
        current_price = float(close_series.iloc[-1])
        current_bin = pd.qcut([current_price], q=10, duplicates='drop', labels=False)[0]
        
        # Calculate volume metrics
        mean_volume = float(volume_profile.mean())
        
        # Handle the case where the current bin exists in volume profile
        try:
            current_level_volume = float(volume_profile.iloc[current_bin])
        except (IndexError, KeyError):
            current_level_volume = mean_volume
            
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

def determine_direction(score: float) -> str:
    """Determine price direction based on combined score."""
    if score > 0.55:  # Lower threshold for upward movement
        return 'UP'
    elif score < 0.45:  # Higher threshold for downward movement
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
    if len(data) < 50:
        return {
            'short_term': {
                'timeframe': '1 Week',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_high': None,
                'predicted_low': None
            }
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
    
    # Define timeframes and their parameters
    timeframes = {
        'short_term': {'name': '1 Week', 'window': 5, 'volatility_window': 10, 'weight': 0.4},
        'medium_term': {'name': '1 Month', 'window': 20, 'volatility_window': 30, 'weight': 0.3},
        'long_term': {'name': '3 Months', 'window': 60, 'volatility_window': 60, 'weight': 0.3}
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
        
        # Combine all factors with updated weights
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
