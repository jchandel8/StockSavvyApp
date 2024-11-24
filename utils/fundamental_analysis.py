import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=86400)
def get_fundamental_metrics(ticker: str) -> dict:
    """Get fundamental analysis metrics for a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = {
            'Market Cap': format_market_cap(info.get('marketCap', 0)),
            'P/E Ratio': round(info.get('forwardPE', 0), 2),
            'EPS': round(info.get('trailingEps', 0), 2),
            'ROE': f"{round(info.get('returnOnEquity', 0) * 100, 2)}%",
            'Debt to Equity': round(info.get('debtToEquity', 0), 2),
            'Current Ratio': round(info.get('currentRatio', 0), 2),
            'Profit Margin': f"{round(info.get('profitMargins', 0) * 100, 2)}%",
            'Beta': round(info.get('beta', 0), 2)
        }
        
        return metrics
    except Exception as e:
        st.error(f"Error fetching fundamental metrics: {str(e)}")
        return {}

def format_market_cap(market_cap: float) -> str:
    """Format market cap in billions/millions."""
    if market_cap >= 1e9:
        return f"${round(market_cap/1e9, 2)}B"
    elif market_cap >= 1e6:
        return f"${round(market_cap/1e6, 2)}M"
    else:
        return f"${round(market_cap, 2)}"

def analyze_fundamentals(metrics: dict) -> dict:
    """Analyze fundamental metrics and provide insights."""
    analysis = {'strengths': [], 'weaknesses': [], 'summary': ''}
    
    if not metrics:
        return analysis
    
    # P/E Analysis
    pe = metrics.get('P/E Ratio', 0)
    if 0 < pe < 15:
        analysis['strengths'].append('Attractive P/E ratio indicating possible undervaluation')
    elif pe > 30:
        analysis['weaknesses'].append('High P/E ratio might indicate overvaluation')
    
    # Debt Analysis
    debt_equity = metrics.get('Debt to Equity', 0)
    if debt_equity < 1:
        analysis['strengths'].append('Healthy debt levels')
    elif debt_equity > 2:
        analysis['weaknesses'].append('High debt levels pose risk')
    
    # Current Ratio Analysis
    current_ratio = metrics.get('Current Ratio', 0)
    if current_ratio > 1.5:
        analysis['strengths'].append('Strong liquidity position')
    elif current_ratio < 1:
        analysis['weaknesses'].append('Potential liquidity concerns')
    
    return analysis
