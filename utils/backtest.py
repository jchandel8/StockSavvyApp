import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.prediction import calculate_prediction
import logging

logger = logging.getLogger(__name__)

def validate_trade(prediction, confidence, market_conditions, current_price):
    """Validate if a trade should be taken based on multiple factors."""
    return (
        confidence > 0.7 and  # High confidence requirement
        abs(prediction['forecast'] - current_price) / current_price > 0.01 and  # Minimum price movement
        prediction['direction'] == market_conditions.get('trend', 'NEUTRAL')  # Trend alignment
    )

def backtest_prediction_model(df: pd.DataFrame, initial_investment: float) -> dict:
    """Run backtesting simulation with realistic trading conditions."""
    try:
        st.info("Running backtest simulation with transaction costs and risk management...")
        
        # Constants for trading simulation
        TRANSACTION_COST = 0.001  # 0.1% per trade
        SLIPPAGE = 0.001  # 0.1% slippage
        POSITION_SIZE = 0.2  # 20% of portfolio per trade
        STOP_LOSS = 0.02  # 2% stop loss
        TAKE_PROFIT = 0.04  # 4% take profit
        
        # Initialize tracking variables
        portfolio_value = initial_investment
        history = []
        total_trades = 0
        profitable_trades = 0
        
        # Use 20-day window for predictions
        window = 20
        
        # Run simulation
        for i in range(window, len(df)-1):
            current_data = df.iloc[:i]
            next_day = df.iloc[i+1]
            
            # Get prediction
            pred = calculate_prediction(current_data, timeframe='daily')
            if not pred:
                continue
                
            # Only trade if confidence is high enough
            if pred['confidence'] < 0.7:
                continue
                
            # Calculate position size
            position_size = portfolio_value * POSITION_SIZE
            
            # Calculate entry price with slippage
            entry_price = float(next_day['Open'])
            entry_price *= (1 + SLIPPAGE) if pred['direction'] == 'UP' else (1 - SLIPPAGE)
            
            # Calculate exit levels
            stop_loss_price = entry_price * (1 - STOP_LOSS) if pred['direction'] == 'UP' else entry_price * (1 + STOP_LOSS)
            take_profit_price = entry_price * (1 + TAKE_PROFIT) if pred['direction'] == 'UP' else entry_price * (1 - TAKE_PROFIT)
            
            # Initialize trade variables
            shares = position_size / entry_price
            exit_price = entry_price
            trade_profit = 0
            trade_taken = False
            
            # Simulate trade execution
            if pred['direction'] == 'UP':
                if next_day['Low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    trade_taken = True
                elif next_day['High'] >= take_profit_price:
                    exit_price = take_profit_price
                    trade_taken = True
                else:
                    exit_price = float(next_day['Close'])
                    trade_taken = True
                    
                if trade_taken:
                    trade_profit = (exit_price - entry_price) * shares
                    
            else:  # SHORT trade
                if next_day['High'] >= stop_loss_price:
                    exit_price = stop_loss_price
                    trade_taken = True
                elif next_day['Low'] <= take_profit_price:
                    exit_price = take_profit_price
                    trade_taken = True
                else:
                    exit_price = float(next_day['Close'])
                    trade_taken = True
                    
                if trade_taken:
                    trade_profit = (entry_price - exit_price) * shares
            
            # Apply transaction costs
            if trade_taken:
                trade_profit -= position_size * TRANSACTION_COST * 2  # Entry and exit costs
                portfolio_value += trade_profit
                
                total_trades += 1
                if trade_profit > 0:
                    profitable_trades += 1
                
                # Record trade
                history.append({
                    'date': df.index[i+1],
                    'portfolio_value': float(portfolio_value),
                    'predicted_direction': pred['direction'],
                    'prediction_confidence': float(pred['confidence']),
                    'open_price': float(next_day['Open']),
                    'high_price': float(next_day['High']),
                    'low_price': float(next_day['Low']),
                    'close_price': float(next_day['Close']),
                    'position_size': float(position_size),
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'trade_profit': float(trade_profit),
                    'trade_taken': True
                })
        
        # Calculate performance metrics
        history_df = pd.DataFrame(history)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((portfolio_value - initial_investment) / initial_investment) * 100
        
        return {
            'final_value': portfolio_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'history': history_df
        }
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        return {
            'final_value': initial_investment,
            'total_return': 0,
            'total_trades': 0,
            'win_rate': 0,
            'history': pd.DataFrame()
        }

def create_backtest_chart(history: pd.DataFrame) -> go.Figure:
    """Create visualization of backtest results."""
    fig = go.Figure()
    
    # Portfolio value line
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
        title='Backtest Results',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return fig