import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.prediction import calculate_prediction

def backtest_prediction_model(df: pd.DataFrame, initial_investment: float) -> dict:
    st.info("This backtest is based on the 1-day forecasts made by our model. For UP predictions, it simulates buying at open price and selling at the day's achieved high price if profitable. For DOWN predictions, it simulates shorting at open price and covering at the day's achieved low price if profitable.")
    
    history = []
    portfolio_value = initial_investment
    total_trades = 0
    profitable_trades = 0
    total_profit = 0
    
    # Use sliding window for predictions
    window_size = 30
    
    for i in range(window_size, len(df)-1):
        # Get data for current and next day
        historical_data = df.iloc[:i]
        next_day = df.iloc[i+1]
        
        # Get prediction for next day
        prediction = calculate_prediction(historical_data, timeframe='daily')
        if not prediction:
            continue
        
        predicted_direction = 'UP' if prediction.get('forecast', 0) > df['Close'].iloc[i] else 'DOWN'
        
        # Simulate trade based on prediction
        trade_profit = 0
        trade_taken = False
        
        if predicted_direction == 'UP':
            # Long trade: Buy at open, sell at high if profitable
            if next_day['High'] > next_day['Open']:
                trade_profit = ((next_day['High'] - next_day['Open']) / next_day['Open']) * portfolio_value
                trade_taken = True
                
        else:  # predicted_direction == 'DOWN'
            # Short trade: Sell at open, buy at low if profitable
            if next_day['Low'] < next_day['Open']:
                trade_profit = ((next_day['Open'] - next_day['Low']) / next_day['Open']) * portfolio_value
                trade_taken = True
        
        if trade_taken:
            total_trades += 1
            if trade_profit > 0:
                profitable_trades += 1
                total_profit += trade_profit
            portfolio_value += trade_profit
        
        # Record history
        history.append({
            'date': df.index[i+1],
            'portfolio_value': portfolio_value,
            'predicted_direction': predicted_direction,
            'open_price': next_day['Open'],
            'high_price': next_day['High'],
            'low_price': next_day['Low'],
            'trade_profit': trade_profit if trade_taken else 0,
            'trade_taken': trade_taken
        })
    
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'final_value': portfolio_value,
        'total_return': ((portfolio_value - initial_investment) / initial_investment) * 100,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'accuracy': win_rate,
        'history': pd.DataFrame(history)
    }

def create_backtest_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    # Portfolio value
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['portfolio_value'],
        name='Portfolio Value',
        line=dict(color='#FF4B4B', width=2)
    ))
    
    # Add trade markers
    profitable_trades = history[history['trade_profit'] > 0]
    losing_trades = history[history['trade_profit'] < 0]
    
    if len(profitable_trades) > 0:
        fig.add_trace(go.Scatter(
            x=profitable_trades['date'],
            y=profitable_trades['portfolio_value'],
            mode='markers',
            name='Profitable Trade',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if len(losing_trades) > 0:
        fig.add_trace(go.Scatter(
            x=losing_trades['date'],
            y=losing_trades['portfolio_value'],
            mode='markers',
            name='Losing Trade',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title='Backtest Results - Daily Prediction Performance',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return fig
