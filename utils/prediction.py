import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
from typing import Dict, List, Tuple

def calculate_trend_strength(data: pd.DataFrame) -> float:
    """Calculate the strength of the current trend."""
    if len(data) < 20:
        return 0.0
        
    try:
        # Calculate price momentum
        returns = data['Close'].pct_change()
        momentum = float(returns.rolling(window=20).mean().iloc[-1])
        
        # Calculate trend consistency
        direction_changes = (returns[1:] * returns[:-1].values < 0).sum()
        consistency = 1 - (direction_changes / len(returns))
        
        # Combine factors
        score = (momentum + consistency) / 2
        return max(min(abs(score), 1.0), 0.0)
    except:
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
        # Calculate momentum indicators
        returns = data['Close'].pct_change()
        momentum = returns.rolling(window=20).mean()
        volatility = returns.rolling(window=20).std()
        
        # Determine cycle phase
        latest_momentum = momentum.iloc[-1]
        latest_volatility = volatility.iloc[-1]
        
        if latest_momentum > 0 and latest_volatility < volatility.mean():
            return 0.8  # Upward trend with low volatility
        elif latest_momentum > 0:
            return 0.6  # Upward trend with high volatility
        elif latest_momentum < 0 and latest_volatility > volatility.mean():
            return 0.2  # Downward trend with high volatility
        else:
            return 0.4  # Downward trend with low volatility
    except:
        return 0.5

def analyze_volume_profile(data: pd.DataFrame) -> float:
    """Analyze volume profile for price levels."""
    try:
        # Calculate volume-weighted price levels
        price_levels = pd.qcut(data['Close'], q=10)
        volume_profile = data.groupby(price_levels)['Volume'].mean()
        
        # Current price level's volume ratio
        current_price = data['Close'].iloc[-1]
        current_level_volume = volume_profile[price_levels.iloc[-1]]
        avg_volume = volume_profile.mean()
        
        volume_ratio = current_level_volume / avg_volume
        return min(float(volume_ratio), 1.0)
    except:
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
    if score > 0.6:
        return 'UP'
    elif score < 0.4:
        return 'DOWN'
    return 'NEUTRAL'

def calculate_confidence(score: float, volatility: float) -> float:
    """Calculate prediction confidence."""
    base_confidence = abs(score - 0.5) * 2  # Convert to 0-1 scale
    volatility_factor = max(1 - volatility * 2, 0)  # Lower confidence with high volatility
    return min(base_confidence * volatility_factor, 1.0)

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
        # Calculate technical score
        window_data = data.tail(params['window'])
        technical_score = calculate_trend_strength(window_data)
        
        # Calculate volatility
        volatility_data = data.tail(params['volatility_window'])
        volatility = volatility_data['Close'].std() / volatility_data['Close'].mean()
        
        # Combine all factors
        combined_score = (
            trend_strength * 0.3 +
            technical_score * 0.3 +
            fundamental_score * 0.2 +
            market_cycle_score * 0.1 +
            volume_profile_score * 0.1
        )
        
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
