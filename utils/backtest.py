import pandas as pd
import plotly.graph_objects as go
from utils.prediction import calculate_prediction

def backtest_prediction_model(df: pd.DataFrame, initial_investment: float) -> dict:
    history = []
    portfolio_value = initial_investment
    correct_predictions = 0
    total_predictions = 0
    total_trades = 0
    profitable_trades = 0
    
    # Use sliding window to simulate historical predictions
    window_size = 60
    position = None
    entry_price = 0
    
    for i in range(window_size, len(df)-1):
        historical_data = df.iloc[:i]
        prediction = calculate_prediction(historical_data)
        
        current_price = df['Close'].iloc[i]
        next_price = df['Close'].iloc[i+1]
        
        # Get predicted high/low prices
        pred_high = prediction.get('predicted_high', current_price * 1.02)
        pred_low = prediction.get('predicted_low', current_price * 0.98)
        confidence = prediction.get('confidence', 0.5)
        
        # Position sizing based on confidence
        position_size = portfolio_value * min(confidence * 0.8, 0.8)  # Max 80% of portfolio
        
        # Trading logic
        if position is None:  # No position
            if current_price < pred_low:  # Buy signal
                position = 'long'
                entry_price = current_price
                shares = position_size / current_price
                total_trades += 1
            elif current_price > pred_high:  # Short signal
                position = 'short'
                entry_price = current_price
                shares = position_size / current_price
                total_trades += 1
        else:  # In position
            # Take profit or stop loss logic
            if position == 'long':
                if current_price >= pred_high or current_price <= pred_low * 0.95:  # Take profit or stop loss
                    profit = shares * (current_price - entry_price)
                    portfolio_value += profit
                    if profit > 0:
                        profitable_trades += 1
                        correct_predictions += 1
                    position = None
            else:  # Short position
                if current_price <= pred_low or current_price >= pred_high * 1.05:  # Take profit or stop loss
                    profit = shares * (entry_price - current_price)
                    portfolio_value += profit
                    if profit > 0:
                        profitable_trades += 1
                        correct_predictions += 1
                    position = None
        
        total_predictions += 1
        history.append({
            'date': df.index[i],
            'portfolio_value': portfolio_value,
            'predicted_high': pred_high,
            'predicted_low': pred_low,
            'actual_price': current_price,
            'position': position,
            'confidence': confidence
        })
    
    return {
        'final_value': portfolio_value,
        'total_return': ((portfolio_value - initial_investment) / initial_investment) * 100,
        'accuracy': (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': (profitable_trades / total_trades * 100) if total_trades > 0 else 0,
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
    
    # Predicted ranges
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['predicted_high'],
        name='Predicted High',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['predicted_low'],
        name='Predicted Low',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Backtest Results - Portfolio Value and Predictions',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return fig
