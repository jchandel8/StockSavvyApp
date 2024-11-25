import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def create_stock_chart(df, indicators):
    """Create an interactive stock chart with technical indicators."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # Add Moving Averages if available
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='blue', width=1)
        ), row=1, col=1)

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='orange', width=1)
        ), row=1, col=1)

    # Add Bollinger Bands if available
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ), row=1, col=1)

    # Volume chart
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='rgb(158,202,225)'
    ), row=2, col=1)

    fig.update_layout(
        title_text="Stock Price Chart",
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
