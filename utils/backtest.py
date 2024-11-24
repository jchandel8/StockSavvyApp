import pandas as pd
import plotly.graph_objects as go
from utils.prediction import calculate_prediction

def backtest_prediction_model(df: pd.DataFrame, initial_investment: float) -> dict:
    history = []
    portfolio_value = initial_investment
    cash = initial_investment
    position = None
    shares = 0
    
    # Trading performance metrics
    total_trades = 0
    profitable_trades = 0
    total_profit = 0
    total_loss = 0
    
    # Transaction costs
    commission = 0.001  # 0.1% per trade
    
    # Use sliding window for predictions
    window_size = 30  # For daily predictions
    
    for i in range(window_size, len(df)-1):
        # Get historical data up to current point
        historical_data = df.iloc[:i]
        current_price = df['Close'].iloc[i]
        next_price = df['Close'].iloc[i+1]
        
        # Get prediction for next day
        prediction = calculate_prediction(historical_data, timeframe='daily')
        if not prediction:
            continue
            
        predicted_high = prediction.get('forecast', 0) * 1.02
        predicted_low = prediction.get('forecast', 0) * 0.98
        confidence = prediction.get('confidence', 0)
        
        # Trading logic
        if position is None:  # No position
            if current_price < predicted_low and confidence > 0.6:
                # Buy signal with high confidence
                position_size = cash * 0.95  # Use 95% of available cash
                shares = position_size * (1 - commission) / current_price
                position = 'long'
                cash -= position_size
                total_trades += 1
                entry_price = current_price
        else:  # In position
            position_value = shares * current_price
            
            # Calculate profit/loss
            if position == 'long':
                if current_price >= predicted_high:  # Take profit
                    profit = shares * (current_price * (1 - commission) - entry_price)
                    cash += shares * current_price * (1 - commission)
                    if profit > 0:
                        profitable_trades += 1
                        total_profit += profit
                    else:
                        total_loss += abs(profit)
                    position = None
                    shares = 0
                elif current_price <= predicted_low * 0.95:  # Stop loss
                    loss = shares * (current_price * (1 - commission) - entry_price)
                    cash += shares * current_price * (1 - commission)
                    total_loss += abs(loss)
                    position = None
                    shares = 0
        
        # Calculate current portfolio value
        portfolio_value = cash + (shares * current_price if shares > 0 else 0)
        
        # Record history
        history.append({
            'date': df.index[i],
            'portfolio_value': portfolio_value,
            'predicted_high': predicted_high,
            'predicted_low': predicted_low,
            'actual_price': current_price,
            'position': position,
            'confidence': confidence
        })
    
    # Calculate final metrics
    total_pnl = total_profit - total_loss
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit_per_trade = total_profit / profitable_trades if profitable_trades > 0 else 0
    avg_loss_per_trade = total_loss / (total_trades - profitable_trades) if (total_trades - profitable_trades) > 0 else 0
    
    return {
        'final_value': portfolio_value,
        'total_return': ((portfolio_value - initial_investment) / initial_investment) * 100,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'accuracy': win_rate,  # Same as win rate in this case
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_loss_per_trade': avg_loss_per_trade,
        'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
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
    
    # Actual price
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['actual_price'],
        name='Stock Price',
        line=dict(color='#4B4BFF', width=1)
    ))
    
    # Add buy/sell markers
    long_entries = history[history['position'] == 'long'].index
    if len(long_entries) > 0:
        fig.add_trace(go.Scatter(
            x=history['date'][long_entries],
            y=history['actual_price'][long_entries],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    fig.update_layout(
        title='Backtest Results - Portfolio Value and Trading Signals',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return fig
