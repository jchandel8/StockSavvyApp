import pandas as pd
import plotly.graph_objects as go
from utils.prediction import calculate_prediction

def backtest_prediction_model(df: pd.DataFrame, initial_investment: float) -> dict:
    history = []
    portfolio_value = initial_investment
    correct_predictions = 0
    total_predictions = 0
    
    # Use sliding window to simulate historical predictions
    window_size = 60  # Same as our prediction model's look_back
    for i in range(window_size, len(df)-1):
        historical_data = df.iloc[:i]
        prediction = calculate_prediction(historical_data)
        actual_return = (df['Close'].iloc[i+1] - df['Close'].iloc[i]) / df['Close'].iloc[i]
        
        # Update portfolio based on prediction
        if prediction.get('forecast', 0) > df['Close'].iloc[i]:
            portfolio_value *= (1 + actual_return)
            if actual_return > 0:
                correct_predictions += 1
        else:
            portfolio_value *= (1 - actual_return)
            if actual_return < 0:
                correct_predictions += 1
        
        total_predictions += 1
        history.append({
            'date': df.index[i],
            'portfolio_value': portfolio_value,
            'predicted_direction': 'UP' if prediction.get('forecast', 0) > df['Close'].iloc[i] else 'DOWN',
            'actual_direction': 'UP' if actual_return > 0 else 'DOWN'
        })
    
    return {
        'final_value': portfolio_value,
        'total_return': ((portfolio_value - initial_investment) / initial_investment) * 100,
        'accuracy': (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0,
        'history': pd.DataFrame(history)
    }

def create_backtest_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['portfolio_value'],
        name='Portfolio Value',
        line=dict(color='#FF4B4B', width=2)
    ))
    
    fig.update_layout(
        title='Backtest Results - Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_dark',
        height=500
    )
    
    return fig
