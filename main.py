import streamlit as st
import pandas as pd
from utils.stock_data import get_stock_data, get_stock_info, search_stocks, is_crypto
from utils.technical_analysis import calculate_indicators, generate_signals
from utils.fundamental_analysis import get_fundamental_metrics, analyze_fundamentals, format_market_cap, format_number

def get_direction_indicator(direction: str) -> str:
    colors = {
        'UP': '#22c55e',  # Green
        'DOWN': '#ef4444',  # Red
        'NEUTRAL': '#6b7280'  # Gray
    }
    return f'''
        <div style="display: inline-flex; align-items: center; gap: 8px;">
            <div style="
                width: 10px;
                height: 10px;
                background: {colors[direction]};
                border-radius: 50%;
                box-shadow: 0 0 4px {colors[direction]};
                animation: pulse 2s infinite;
            "></div>
            <span>{direction}</span>
            <style>
                @keyframes pulse {{
                    0% {{ box-shadow: 0 0 0 0 {colors[direction]}66; }}
                    70% {{ box-shadow: 0 0 0 6px {colors[direction]}00; }}
                    100% {{ box-shadow: 0 0 0 0 {colors[direction]}00; }}
                }}
            </style>
        </div>
    '''
# Imports are already defined above
from utils.news_service import get_news, format_news_sentiment
from utils.prediction import get_prediction
from components.chart import create_stock_chart
from components.signals import display_signals, display_technical_summary

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# Load custom CSS
with open('styles/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Title and search
st.title("Stock Analysis Platform")

# Create placeholder for search input
search_container = st.container()
search_input = search_container.empty()

# Search box with auto-complete
search_query = search_input.text_input("Search for a stock", value="", key="stock_search")
stock_suggestions = search_stocks(search_query)

# Show real-time suggestions
if stock_suggestions and search_query:
    suggestion_container = st.container()
    with suggestion_container:
        for stock in stock_suggestions:
            if st.button(f"{stock['name']} ({stock['symbol']}) - {stock['exchange']}", key=f"btn_{stock['symbol']}"):
                search_query = stock['symbol']
                ticker = stock['symbol']
                break
    ticker = search_query
else:
    ticker = search_query

if ticker:
    # Load data
    df = get_stock_data(ticker)
    if not df.empty:
        info = get_stock_info(ticker)
        st.header(f"{info['name']} ({ticker})")
        
        # Stock price and info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", 
                     f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):.2f}%")
        with col2:
            st.metric("Market Cap", format_market_cap(info['market_cap']))
        with col3:
            st.metric("Sector", info['sector'])
        
        # Technical Analysis
        st.subheader("Technical Analysis")
        is_crypto = is_crypto(ticker)
        df = calculate_indicators(df, is_crypto=is_crypto)
        
        if is_crypto:
            # Add crypto-specific metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("24h Volume", format_market_cap(info.get('volume_24h', 0)))
            with col2:
                st.metric("24h Change", f"{info.get('price_change_24h', 0):.2f}%")
            with col3:
                st.metric("Circulating Supply", format_number(info.get('circulating_supply', 0)))
            with col4:
                st.metric("Market Dominance", f"{info.get('market_dominance', 0):.2f}%")
        
        fig = create_stock_chart(df, None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading Signals and Predictions
        signals = generate_signals(df)
        display_signals(signals)
        display_technical_summary(df)
        
        # Price Predictions
        st.subheader("Price Predictions")
        predictions = get_prediction(df, ticker)
        
        timeframe_tabs = st.tabs(['Short-term', 'Medium-term', 'Long-term'])
        
        for tab, (timeframe, pred) in zip(timeframe_tabs, predictions.items()):
            with tab:
                st.write(f"**{pred['timeframe']} Forecast**")
                pred_cols = st.columns(4)
                
                with pred_cols[0]:
                    st.markdown(get_direction_indicator(pred['direction']), unsafe_allow_html=True)
                
                with pred_cols[1]:
                    st.metric("Confidence", f"{pred['confidence']*100:.1f}%")
                
                with pred_cols[2]:
                    if pred['predicted_high']:
                        st.metric("Predicted High", f"${pred['predicted_high']:.2f}")
                
                with pred_cols[3]:
                    if pred['predicted_low']:
                        st.metric("Predicted Low", f"${pred['predicted_low']:.2f}")
        
        # Fundamental Analysis
        st.subheader("Fundamental Analysis")
        metrics = get_fundamental_metrics(ticker)
        analysis = analyze_fundamentals(metrics)
        
        cols = st.columns(4)
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 4]:
                st.metric(metric, value)
        
        # Fundamental Analysis Summary
        if analysis['strengths'] or analysis['weaknesses']:
            st.subheader("Fundamental Analysis Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Strengths**")
                for strength in analysis['strengths']:
                    st.write(f"✅ {strength}")
            with col2:
                st.write("**Weaknesses**")
                for weakness in analysis['weaknesses']:
                    st.write(f"⚠️ {weakness}")
        
        # News Section
        st.subheader("Latest News")
        news = get_news(ticker)
        for article in news:
            sentiment_label, sentiment_color = format_news_sentiment(article['sentiment'])
            with st.expander(f"{article['title']} ({sentiment_label})"):
                if article.get('image_url'):
                    st.image(article['image_url'], use_container_width=True)
                st.write(article['summary'])
                st.write(f"Source: {article['source']} | [Read More]({article['url']})")
    else:
        st.error("Unable to fetch stock data. Please check the ticker symbol.")
