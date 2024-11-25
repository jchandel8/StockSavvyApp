import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.prediction import calculate_prediction

def backtest_prediction_model(df: pd.DataFrame, initial_investment: float) -> dict:
    st.info("This backtest simulates realistic trading conditions including transaction costs, slippage, and risk management.")
    
    # Constants for trading simulation
    TRANSACTION_COST = 0.001  # 0.1% per trade
    SLIPPAGE = 0.001  # 0.1% slippage
    MAX_POSITION_SIZE = 0.2  # Maximum 20% of portfolio per trade
    STOP_LOSS = 0.02  # 2% stop loss
    TAKE_PROFIT = 0.04  # 4% take profit
    
    history = []
    portfolio_value = initial_investment
    total_trades = 0
    profitable_trades = 0
    total_profit = 0
    max_drawdown = 0
    peak_value = initial_investment
    
    # Use sliding window for predictions
    window_size = 30
    
    for i in range(window_size, len(df)-1):
        # Get data for current and next day
        historical_data = df.iloc[:i]
        next_day = df.iloc[i+1]
        current_day = df.iloc[i]
        
        # Get prediction for next day
        prediction = calculate_prediction(historical_data, timeframe='daily')
        if not prediction:
            continue
        
        forecast = prediction.get('forecast', 0)
        confidence = prediction.get('confidence', 0)
        
        # Only trade if confidence is high enough
        if confidence < 0.6:
            continue
            
        predicted_direction = 'UP' if forecast > current_day['Close'] else 'DOWN'
        
        # Calculate position size based on confidence and portfolio value
        position_size = min(MAX_POSITION_SIZE * confidence, MAX_POSITION_SIZE) * portfolio_value
        
        # Calculate entry price with slippage
        entry_price = next_day['Open'] * (1 + SLIPPAGE) if predicted_direction == 'UP' else next_day['Open'] * (1 - SLIPPAGE)
        
        # Calculate stop loss and take profit levels
        stop_loss_price = entry_price * (1 - STOP_LOSS) if predicted_direction == 'UP' else entry_price * (1 + STOP_LOSS)
        take_profit_price = entry_price * (1 + TAKE_PROFIT) if predicted_direction == 'UP' else entry_price * (1 - TAKE_PROFIT)
        
        # Simulate trade based on prediction
        trade_profit = 0
        trade_taken = False
        exit_price = entry_price  # Default to entry price
        
        if predicted_direction == 'UP':
            # Long trade simulation with realistic exit conditions
            shares = position_size / entry_price
            transaction_cost = position_size * TRANSACTION_COST * 2  # Entry and exit costs
            
            # Check if stop loss was hit first
            if next_day['Low'] <= stop_loss_price:
                exit_price = stop_loss_price
                trade_profit = (exit_price - entry_price) * shares - transaction_cost
                trade_taken = True
            # Check if take profit was hit
            elif next_day['High'] >= take_profit_price:
                exit_price = take_profit_price
                trade_profit = (exit_price - entry_price) * shares - transaction_cost
                trade_taken = True
            # Otherwise, close at end of day
            else:
                exit_price = next_day['Close']
                trade_profit = (exit_price - entry_price) * shares - transaction_cost
                trade_taken = True
                
        else:  # predicted_direction == 'DOWN'
            # Short trade simulation with realistic exit conditions
            shares = position_size / entry_price
            transaction_cost = position_size * TRANSACTION_COST * 2  # Entry and exit costs
            
            # Check if stop loss was hit first
            if next_day['High'] >= stop_loss_price:
                exit_price = stop_loss_price
                trade_profit = (entry_price - exit_price) * shares - transaction_cost
                trade_taken = True
            # Check if take profit was hit
            elif next_day['Low'] <= take_profit_price:
                exit_price = take_profit_price
                trade_profit = (entry_price - exit_price) * shares - transaction_cost
                trade_taken = True
            # Otherwise, close at end of day
            else:
                exit_price = next_day['Close']
                trade_profit = (entry_price - exit_price) * shares - transaction_cost
                trade_taken = True
        
        if trade_taken:
            total_trades += 1
            portfolio_value += trade_profit
            
            if trade_profit > 0:
                profitable_trades += 1
                total_profit += trade_profit
            
            # Update maximum drawdown
            peak_value = max(peak_value, portfolio_value)
            drawdown = (peak_value - portfolio_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
        
        # Record history with enhanced metrics
        history.append({
            'date': df.index[i+1],
            'portfolio_value': portfolio_value,
            'predicted_direction': predicted_direction,
            'prediction_confidence': confidence,
            'position_size': position_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'trade_profit': trade_profit if trade_taken else 0,
            'trade_taken': trade_taken,
            'drawdown': drawdown if trade_taken else 0
        })
    
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
    profit_factor = abs(total_profit / (initial_investment - portfolio_value)) if portfolio_value < initial_investment else float('inf')
    
    return {
        'final_value': portfolio_value,
        'total_return': ((portfolio_value - initial_investment) / initial_investment) * 100,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'accuracy': win_rate,
        'max_drawdown': max_drawdown * 100,
        'profit_factor': profit_factor,
        'avg_profit_per_trade': avg_profit_per_trade,
        'sharpe_ratio': (total_profit / initial_investment) / (df['Close'].pct_change().std() * (252 ** 0.5)) if total_trades > 0 else 0,
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
