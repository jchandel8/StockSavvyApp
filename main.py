import streamlit as st
import pandas as pd
from utils.stock_data import get_stock_data, get_stock_info, search_stocks
from utils.technical_analysis import calculate_indicators, generate_signals
from utils.fundamental_analysis import get_fundamental_metrics, analyze_fundamentals
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

# Search box with auto-complete
search_query = st.text_input("Search for a stock", value="", key="stock_search")
stock_suggestions = search_stocks(search_query)

if stock_suggestions:
    # Create a list of formatted options for the selectbox
    options = [f"{stock['symbol']} - {stock['name']} ({stock['exchange']})" for stock in stock_suggestions]
    selected = st.selectbox("Select a stock", options, key="stock_select")
    
    # Extract the ticker from the selected option
    ticker = selected.split(" - ")[0] if selected else ""
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
            st.metric("Market Cap", info['market_cap'])
        with col3:
            st.metric("Sector", info['sector'])
        
        # Technical Analysis
        st.subheader("Technical Analysis")
        df = calculate_indicators(df)
        fig = create_stock_chart(df, None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading Signals and Predictions
        signals = generate_signals(df)
        display_signals(signals)
        display_technical_summary(df)
        
        # Price Predictions
        st.subheader("Price Predictions")
        prediction = get_prediction(df)
        
        pred_cols = st.columns(4)
        with pred_cols[0]:
            direction_color = {
                'UP': '🟢',
                'DOWN': '🔴',
                'NEUTRAL': '⚪'
            }.get(prediction['direction'], '⚪')
            st.metric("Predicted Direction", f"{direction_color} {prediction['direction']}")
        
        with pred_cols[1]:
            st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
        
        with pred_cols[2]:
            if prediction['predicted_high']:
                st.metric("Predicted High", f"${prediction['predicted_high']:.2f}")
        
        with pred_cols[3]:
            if prediction['predicted_low']:
                st.metric("Predicted Low", f"${prediction['predicted_low']:.2f}")
        
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
                    st.image(article['image_url'], use_column_width=True)
                st.write(article['summary'])
                st.write(f"Source: {article['source']} | [Read More]({article['url']})")
    else:
        st.error("Unable to fetch stock data. Please check the ticker symbol.")
