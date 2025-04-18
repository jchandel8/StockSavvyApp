import streamlit as st
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.stock_data import get_stock_data, get_stock_info, search_stocks, is_crypto, format_crypto_symbol
from utils.technical_analysis import generate_signals
from utils.fundamental_analysis import get_fundamental_metrics, analyze_fundamentals, format_market_cap
from utils.prediction import get_prediction
from utils.news_service import get_news, format_news_sentiment
from utils.backtest import backtest_prediction_model, create_backtest_chart
from components.chart import create_stock_chart
from components.signals import display_signals, display_technical_summary

# Page configuration
st.set_page_config(
    page_title="StockSavvy - Advanced Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already done
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_error = None

# Apply custom CSS
try:
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Custom styling could not be loaded: {str(e)}")

# Header with logo and search
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="logo-container">
            <div class="logo">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #0f172a;">
                    <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>
                    <polyline points="16 7 22 7 22 13"></polyline>
                </svg>
            </div>
            <div>
                <h1 style="margin: 0; font-size: 1.25rem; font-weight: bold;">StockSavvy</h1>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Advanced Technical & Fundamental Analysis</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for search and filters
with st.sidebar:
    st.markdown("""
    <div class="logo-container" style="margin-bottom: 2rem;">
        <div class="logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #0f172a;">
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>
                <polyline points="16 7 22 7 22 13"></polyline>
            </svg>
        </div>
        <div>
            <h1 style="margin: 0; font-size: 1.25rem; font-weight: bold;">StockSavvy</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #94a3b8; margin-bottom: 0.5rem;'>Search stocks and crypto</p>", unsafe_allow_html=True)
    search_query = st.text_input("", value="", placeholder="Enter symbol (e.g., AAPL) or name", label_visibility="collapsed")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #94a3b8; margin-bottom: 0.5rem;'>Quick Filters</p>", unsafe_allow_html=True)
    filter_options = st.multiselect(
        "",
        ["Stocks", "Crypto", "ETFs", "Forex"],
        default=["Stocks"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #94a3b8; margin-bottom: 0.5rem;'>Featured Assets</p>", unsafe_allow_html=True)
    if st.button("AAPL - Apple Inc."):
        search_query = "AAPL"
    if st.button("MSFT - Microsoft Corp."):
        search_query = "MSFT"
    if st.button("GOOGL - Alphabet Inc."):
        search_query = "GOOGL"
    if st.button("BTC-USD - Bitcoin"):
        search_query = "BTC-USD"
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #94a3b8; margin-bottom: 0.5rem;'>Time Period</p>", unsafe_allow_html=True)
    time_period = st.select_slider(
        "",
        options=["1D", "1W", "1M", "3M", "6M", "1Y", "5Y", "MAX"],
        value="1Y",
        label_visibility="collapsed"
    )

if search_query:
    with st.spinner('🔍 Searching financial markets...'):
        search_results = search_stocks(search_query)
        if search_results:
            st.markdown("""
                <div style="background: rgba(38, 39, 48, 0.3); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; font-size: 1.2rem; color: #00d2ff;">Search Results</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Format the options with icons and better formatting
            options = []
            for r in search_results:
                icon = "🪙" if r.get('exchange', '').lower() in ['crypto', 'binance', 'coinbase'] else "📈"
                option = f"{icon} {r['symbol']} - {r['name']} ({r['exchange']})"
                options.append(option)
            
            # Use a container for the select box with custom styles
            select_container = st.container()
            with select_container:
                selected = st.selectbox(
                    "Select an asset to analyze",
                    options,
                    format_func=lambda x: x,
                    key="asset_selector"
                )
                
                ticker = selected.split(' - ')[0].strip().split(' ')[-1] if selected else None
        else:
            st.markdown("""
                <div style="background: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 10px; margin: 20px 0;">
                    <h3 style="margin-top: 0; font-size: 1.2rem; color: #ff4b4b;">No Results Found</h3>
                    <p>We couldn't find any matches for your search. Please try:</p>
                    <ul>
                        <li>Checking for typos</li>
                        <li>Using a stock symbol (e.g., AAPL, MSFT, GOOG)</li>
                        <li>Using a crypto symbol (e.g., BTC-USD, ETH-USD)</li>
                        <li>Searching for a company name (e.g., Apple, Microsoft)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('last_error'):
                st.error(st.session_state['last_error'])
                
            ticker = None
else:
    # Display featured assets when no search is performed
    st.markdown("""
        <div style="background: rgba(58, 123, 213, 0.1); padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h3 style="margin-top: 0; font-size: 1.2rem; color: #3a7bd5;">Featured Assets</h3>
            <p>Search for any stock or cryptocurrency above, or select one of these popular options:</p>
        </div>
    """, unsafe_allow_html=True)
    
    featured_col1, featured_col2, featured_col3, featured_col4 = st.columns(4)
    
    with featured_col1:
        if st.button("📱 Apple (AAPL)"):
            ticker = "AAPL"
    with featured_col2:
        if st.button("🪙 Bitcoin (BTC-USD)"):
            ticker = "BTC-USD"
    with featured_col3:
        if st.button("🚗 Tesla (TSLA)"):
            ticker = "TSLA"
    with featured_col4:
        if st.button("📊 S&P 500 (SPY)"):
            ticker = "SPY"
    
    # Set ticker to None by default
    if 'ticker' not in locals():
        ticker = None

if ticker:
    try:
        # Format crypto symbols correctly
        if is_crypto(ticker):
            ticker = format_crypto_symbol(ticker)
            
        # Get stock/crypto data and info
        df = get_stock_data(ticker)
        info = get_stock_info(ticker)
        
        if not df.empty:
            # Create an attractive header for the asset
            asset_type = "Cryptocurrency" if is_crypto(ticker) else "Stock"
            icon = "🪙" if is_crypto(ticker) else "📈"
            
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, rgba(58, 123, 213, 0.2), rgba(0, 210, 255, 0.1)); 
                            padding: 20px; border-radius: 12px; margin: 15px 0; 
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="font-size: 2.5rem; margin-right: 15px;">{icon}</div>
                        <div>
                            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">{info.get('name', ticker)}</h1>
                            <p style="margin: 0; color: rgba(255, 255, 255, 0.7); font-size: 1.1rem;">
                                {asset_type} • {ticker} • {info.get('exchange', 'Market')}
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Calculate price metrics
            current_price = df['Close'].iloc[-1]
            price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            price_color = "#00d2ff" if price_change >= 0 else "#ff4b4b"
            
            # Create a modern metrics section with 4 cards
            st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)
            metrics_container = st.container()
            
            with metrics_container:
                # First row of metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Current Price</p>
                            <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">${current_price:.2f}</h2>
                            <p style="margin: 0; font-size: 1rem; color: {price_color}; font-weight: 500;">
                                {price_change:+.2f}% {' ↑' if price_change >= 0 else ' ↓'}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    market_cap = format_market_cap(info.get('market_cap', 0))
                    st.markdown(f"""
                        <div class="metric-card">
                            <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Market Cap</p>
                            <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">{market_cap}</h2>
                            <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.5);">
                                {asset_type} value
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if is_crypto(ticker):
                        volume = f"${info.get('volume_24h', 0):,.0f}"
                        volume_label = "24h Volume"
                        volume_desc = "Trading volume"
                    else:
                        volume = f"{info.get('pe_ratio', 0):.2f}"
                        volume_label = "P/E Ratio"
                        volume_desc = "Price to earnings"
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">{volume_label}</p>
                            <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">{volume}</h2>
                            <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.5);">
                                {volume_desc}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # 52 week range or equivalent metric
                    if len(df) > 200:  # Enough data for yearly analysis
                        year_high = df['High'].iloc[-252:].max()
                        year_low = df['Low'].iloc[-252:].min()
                        pct_from_high = ((current_price - year_high) / year_high) * 100
                        
                        st.markdown(f"""
                            <div class="metric-card">
                                <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">52 Week Range</p>
                                <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">${year_low:.2f} - ${year_high:.2f}</h2>
                                <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.5);">
                                    {pct_from_high:.1f}% from high
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # For assets with limited data
                        high = df['High'].max()
                        low = df['Low'].min()
                        data_period = f"{len(df)} day" if len(df) == 1 else f"{len(df)} days"
                        
                        st.markdown(f"""
                            <div class="metric-card">
                                <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">{data_period} Range</p>
                                <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">${low:.2f} - ${high:.2f}</h2>
                                <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.5);">
                                    Price range
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Technical Analysis Tab
            tabs = st.tabs(["Technical Analysis", "Price Prediction", "News", "Backtesting"])
            
            with tabs[0]:
                # Display interactive chart
                fig = create_stock_chart(df, {})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trading signals
                signals = generate_signals(df)
                display_signals(signals)
                display_technical_summary(df)
                
            with tabs[1]:
                # Create modern prediction header
                st.markdown("""
                    <div style="background: rgba(38, 39, 48, 0.3); padding: 20px; border-radius: 12px; margin: 15px 0;">
                        <h2 style="margin-top: 0; font-size: 1.5rem; font-weight: 700; margin-bottom: 10px;">
                            <span style="color: #00d2ff;">AI</span> Price Prediction Model
                        </h2>
                        <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.7);">
                            Our advanced ensemble models combine deep learning, technical analysis, and market sentiment.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show loading state while predictions are being generated
                with st.spinner("⚙️ Analyzing market data and generating AI predictions..."):
                    predictions = get_prediction(df, ticker)
                
                # Create timeframe selection tabs
                timeframe_labels = {
                    'daily': '1 Day',
                    'short_term': '1 Week',
                    'medium_term': '1 Month',
                    'long_term': '3 Months',
                    'extended_term': '6 Months'
                }
                
                prediction_container = st.container()
                
                with prediction_container:
                    # Create selection buttons for timeframes
                    st.markdown("""
                        <style>
                        div.timeframe-selector {
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                            margin: 20px 0;
                        }
                        div.timeframe-button {
                            padding: 8px 15px;
                            background: rgba(38, 39, 48, 0.7);
                            border-radius: 8px;
                            cursor: pointer;
                            text-align: center;
                            transition: all 0.3s ease;
                            border: 1px solid rgba(255, 255, 255, 0.05);
                        }
                        div.timeframe-button:hover {
                            background: rgba(58, 123, 213, 0.2);
                            transform: translateY(-2px);
                        }
                        div.timeframe-button.active {
                            background: rgba(58, 123, 213, 0.4);
                            border: 1px solid #3a7bd5;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Initialize timeframe in session state if not already done
                    if 'selected_timeframe' not in st.session_state:
                        st.session_state.selected_timeframe = 'short_term'
                    
                    # Create the timeframe selector UI
                    timeframe_html = '<div class="timeframe-selector">'
                    for tf, label in timeframe_labels.items():
                        active_class = "active" if st.session_state.selected_timeframe == tf else ""
                        timeframe_html += f'<div class="timeframe-button {active_class}" onclick="this.dataset.clicked=\'{tf}\'" data-timeframe="{tf}">{label}</div>'
                    timeframe_html += '</div>'
                    
                    # Display the timeframe selector
                    st.markdown(timeframe_html, unsafe_allow_html=True)
                    
                    # Create buttons for selecting timeframes
                    cols = st.columns(len(timeframe_labels))
                    for i, (tf, label) in enumerate(timeframe_labels.items()):
                        with cols[i]:
                            if st.button(label, key=f"tf_{tf}"):
                                st.session_state.selected_timeframe = tf
                    
                    # Display the selected timeframe's prediction
                    selected_tf = st.session_state.selected_timeframe
                    
                    if selected_tf in predictions:
                        pred = predictions[selected_tf]
                        
                        # Create the prediction card
                        direction_icon = "↗️" if pred['direction'] == "UP" else ("↘️" if pred['direction'] == "DOWN" else "↔️")
                        direction_color = "#00d2ff" if pred['direction'] == "UP" else ("#ff4b4b" if pred['direction'] == "DOWN" else "#cccccc")
                        confidence_val = pred.get('confidence', 0) * 100
                        confidence_bg = "rgba(0, 210, 255, 0.2)" if pred['direction'] == "UP" else ("rgba(255, 75, 75, 0.2)" if pred['direction'] == "DOWN" else "rgba(204, 204, 204, 0.2)")
                        
                        # Calculate forecast change values
                        current_price = df['Close'].iloc[-1]
                        forecast_val = pred.get('forecast')
                        change = ((forecast_val - current_price) / current_price) * 100 if forecast_val is not None else 0
                        change_str = f"{change:+.2f}%" if forecast_val is not None else "N/A"
                        
                        # Prediction card header
                        st.markdown(f"""
                            <div style="background: {confidence_bg}; padding: 25px; border-radius: 12px; margin: 20px 0; 
                                        border: 1px solid rgba(255, 255, 255, 0.1);">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                                    <h3 style="margin: 0; font-size: 1.6rem; font-weight: 700;">
                                        {pred['timeframe']} Forecast
                                    </h3>
                                    <div style="background: rgba(0, 0, 0, 0.2); padding: 5px 15px; border-radius: 20px;">
                                        <span style="color: {direction_color}; font-weight: 600; font-size: 1.1rem;">
                                            {direction_icon} {pred['direction']}
                                        </span>
                                    </div>
                                </div>
                        """, unsafe_allow_html=True)
                        
                        # Create prediction metrics grid
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Forecast price
                            forecast_display = f"${forecast_val:.2f}" if forecast_val is not None else "N/A"
                            st.markdown(f"""
                                <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; height: 100%;">
                                    <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Target Price</p>
                                    <h2 style="margin: 5px 0; font-size: 1.8rem; font-weight: 700; color: {direction_color};">
                                        {forecast_display}
                                    </h2>
                                    <p style="margin: 0; font-size: 0.9rem; color: {direction_color};">
                                        {change_str} from current
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Confidence
                            conf_gradient = "linear-gradient(90deg, rgba(0, 210, 255, 0.1) 0%, rgba(0, 210, 255, 0.3) " + str(confidence_val) + "%, rgba(0, 0, 0, 0.1) " + str(confidence_val) + "%)"
                            if pred['direction'] == "DOWN":
                                conf_gradient = "linear-gradient(90deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0.3) " + str(confidence_val) + "%, rgba(0, 0, 0, 0.1) " + str(confidence_val) + "%)"
                            
                            st.markdown(f"""
                                <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; height: 100%;">
                                    <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Model Confidence</p>
                                    <h2 style="margin: 5px 0; font-size: 1.8rem; font-weight: 700;">{confidence_val:.1f}%</h2>
                                    <div style="width: 100%; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden; margin-top: 8px;">
                                        <div style="width: {confidence_val}%; height: 100%; background: {direction_color};"></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Range prediction
                        st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # High prediction
                            high_val = pred.get('predicted_high')
                            high_display = f"${high_val:.2f}" if high_val is not None else "N/A"
                            high_change = ((high_val - current_price) / current_price) * 100 if high_val is not None else 0
                            high_change_str = f"{high_change:+.2f}%" if high_val is not None else "N/A"
                            
                            st.markdown(f"""
                                <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px;">
                                    <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Potential High</p>
                                    <h2 style="margin: 5px 0; font-size: 1.6rem; font-weight: 700; color: #00d2ff;">{high_display}</h2>
                                    <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.5);">{high_change_str}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Low prediction
                            low_val = pred.get('predicted_low')
                            low_display = f"${low_val:.2f}" if low_val is not None else "N/A"
                            low_change = ((low_val - current_price) / current_price) * 100 if low_val is not None else 0
                            low_change_str = f"{low_change:+.2f}%" if low_val is not None else "N/A"
                            
                            st.markdown(f"""
                                <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px;">
                                    <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Potential Low</p>
                                    <h2 style="margin: 5px 0; font-size: 1.6rem; font-weight: 700; color: #ff4b4b;">{low_display}</h2>
                                    <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.5);">{low_change_str}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        # Close the prediction card
                        st.markdown("</div>", unsafe_allow_html=True)
                            
                        # Methodology explanation
                        with st.expander("🧠 How our prediction model works"):
                            st.markdown("""
                                <div style="padding: 10px; font-size: 0.9rem; color: rgba(255, 255, 255, 0.8);">
                                    <p>Our price prediction system combines multiple advanced algorithms:</p>
                                    <ul>
                                        <li><strong>Deep Learning:</strong> Optimized LSTM neural networks analyze historical patterns</li>
                                        <li><strong>Technical Analysis:</strong> Comprehensive indicators including RSI, MACD, Bollinger Bands</li>
                                        <li><strong>Statistical Models:</strong> Time series forecasting with regression and ARIMA models</li>
                                        <li><strong>Sentiment Analysis:</strong> News and social media sentiment integration</li>
                                        <li><strong>Market Regime Detection:</strong> Adapts to different volatility environments</li>
                                    </ul>
                                    <p>The model automatically adjusts weightings based on market conditions and past performance.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Prediction data not available for {timeframe_labels.get(selected_tf, selected_tf)} timeframe.")
                
            with tabs[2]:
                # Modern news section header
                st.markdown("""
                    <div style="background: rgba(38, 39, 48, 0.3); padding: 20px; border-radius: 12px; margin: 15px 0;">
                        <h2 style="margin-top: 0; font-size: 1.5rem; font-weight: 700; margin-bottom: 10px;">
                            <span style="color: #00d2ff;">Market</span> News & Sentiment
                        </h2>
                        <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.7);">
                            Latest news articles and sentiment analysis for this asset
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Get news with loading state
                with st.spinner("📰 Gathering latest market news..."):
                    news = get_news(ticker)
                
                if news:
                    # Add filter options
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("<p style='margin-bottom: 5px; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);'>Filter by sentiment:</p>", unsafe_allow_html=True)
                    with col2:
                        sort_option = st.selectbox(
                            "Sort by:",
                            ["Latest First", "Highest Sentiment", "Lowest Sentiment"],
                            label_visibility="collapsed"
                        )
                    
                    # Sort news articles based on selection
                    if sort_option == "Highest Sentiment":
                        news = sorted(news, key=lambda x: x.get('sentiment', 0), reverse=True)
                    elif sort_option == "Lowest Sentiment":
                        news = sorted(news, key=lambda x: x.get('sentiment', 0))
                    # Default is already latest first
                    
                    # Display articles in a modern card format
                    for article in news:
                        # Safely get values with defaults
                        title = article.get('title', 'No title available')
                        summary = article.get('summary', 'No summary available')
                        source = article.get('source', 'Unknown source')
                        sentiment_val = article.get('sentiment', 0)
                        date = article.get('date', 'Recent')
                        
                        # Format sentiment
                        sentiment, color = format_news_sentiment(sentiment_val)
                        
                        # Determine sentiment icon
                        if sentiment_val > 0.2:
                            sentiment_icon = "📈"
                        elif sentiment_val < -0.2:
                            sentiment_icon = "📉"
                        else:
                            sentiment_icon = "📊"
                        
                        # Generate tag categories based on content analysis
                        tags = []
                        if "price" in title.lower() or "price" in summary.lower():
                            tags.append("Price Action")
                        if "earnings" in title.lower() or "earnings" in summary.lower():
                            tags.append("Earnings")
                        if "analysis" in title.lower() or "analyst" in title.lower() or "rating" in title.lower():
                            tags.append("Analysis")
                        if "market" in title.lower() or "index" in title.lower():
                            tags.append("Market")
                        if len(tags) == 0:
                            tags.append("News")
                        
                        # Create tag HTML
                        tags_html = ""
                        for tag in tags[:2]:  # Limit to 2 tags
                            tags_html += f'<span style="background: rgba(0, 0, 0, 0.3); padding: 3px 8px; border-radius: 12px; font-size: 0.7rem; margin-right: 5px;">{tag}</span>'
                        
                        # Display article card
                        st.markdown(
                            f"""
                            <div style='background: rgba(38, 39, 48, 0.7); padding: 20px; border-radius: 12px; margin: 15px 0; 
                                        border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                                        transition: transform 0.2s ease-in-out;'
                                 onmouseover="this.style.transform='translateY(-5px)'"
                                 onmouseout="this.style.transform='translateY(0)'"
                                 class="news-article">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <div style="display: flex; gap: 10px; align-items: center;">
                                        {tags_html}
                                    </div>
                                    <div style="background: {color}25; color: {color}; padding: 3px 10px; border-radius: 12px; display: flex; align-items: center; font-size: 0.8rem;">
                                        {sentiment_icon} {sentiment}
                                    </div>
                                </div>
                                <h3 style="margin: 10px 0; font-size: 1.2rem; font-weight: 600;">{title}</h3>
                                <p style="margin: 10px 0; color: rgba(255, 255, 255, 0.8); font-size: 0.95rem;">{summary}</p>
                                <div style="display: flex; justify-content: space-between; margin-top: 15px; font-size: 0.8rem; color: rgba(255, 255, 255, 0.6);">
                                    <span>{source}</span>
                                    <span>{date}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Add a "Load more news" button at the bottom
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔄 Load More News", key="load_more_news"):
                            st.info("Functionality to load additional news will be implemented in the next update.")
                else:
                    # Display empty state with suggestions
                    st.markdown("""
                        <div style="background: rgba(38, 39, 48, 0.5); padding: 30px; border-radius: 12px; margin: 25px 0; text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 20px;">📰</div>
                            <h3 style="margin: 10px 0; font-size: 1.3rem; font-weight: 600;">No News Available</h3>
                            <p style="color: rgba(255, 255, 255, 0.7); margin: 15px 0;">
                                We couldn't find any recent news articles for this asset.
                            </p>
                            <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.9rem;">
                                Try searching for a different stock or cryptocurrency, or check back later for updates.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with tabs[3]:
                # Modern backtesting header
                st.markdown("""
                    <div style="background: rgba(38, 39, 48, 0.3); padding: 20px; border-radius: 12px; margin: 15px 0;">
                        <h2 style="margin-top: 0; font-size: 1.5rem; font-weight: 700; margin-bottom: 10px;">
                            <span style="color: #00d2ff;">Strategy</span> Backtesting
                        </h2>
                        <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.7);">
                            Test our AI prediction model with historical data to see how it would have performed
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create a nice configuration panel
                st.markdown("""
                    <div style="background: rgba(38, 39, 48, 0.7); padding: 20px; border-radius: 12px; margin: 20px 0; 
                                border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                        <h3 style="margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 600;">Backtest Configuration</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Configuration inputs in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    initial_investment = st.number_input(
                        "Initial Investment ($)", 
                        min_value=1000, 
                        value=10000, 
                        step=1000,
                        help="Starting capital for the backtest"
                    )
                
                with col2:
                    risk_percentage = st.slider(
                        "Risk per Trade (%)", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=2.0, 
                        step=0.5,
                        help="Percentage of portfolio risked on each trade"
                    )
                
                with col3:
                    strategy = st.selectbox(
                        "Trading Strategy",
                        ["AI Predictions", "MACD Crossover", "RSI Reversals", "Bollinger Bands"],
                        index=0,
                        help="Strategy to use for backtesting"
                    )
                
                # Add a run button with better styling
                st.markdown("""
                    <style>
                    div.stButton > button {
                        width: 100%;
                        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
                        height: 3em;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Run Backtest Simulation", key="run_backtest_button"):
                    with st.spinner("⚙️ Running backtest simulation... This may take a moment"):
                        # Pass in risk parameter (though not used currently in the model)
                        backtest_results = backtest_prediction_model(df, initial_investment)  # risk_percentage would be used in an upgraded version
                    
                    # Display results in a modern container
                    st.markdown("""
                        <div style="background: rgba(38, 39, 48, 0.7); padding: 25px; border-radius: 12px; margin: 25px 0; 
                                    border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                            <h3 style="margin: 0 0 20px 0; font-size: 1.3rem; font-weight: 600;">Backtest Results</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display backtest metrics in cards
                    metric_cols = st.columns(4)
                    
                    # Format the metrics with improved styling
                    final_value = backtest_results['final_value']
                    initial = initial_investment
                    profit = final_value - initial
                    profit_pct = (profit / initial) * 100
                    profit_color = "#00d2ff" if profit >= 0 else "#ff4b4b"
                    
                    with metric_cols[0]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Final Portfolio Value</p>
                                <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">${final_value:,.2f}</h2>
                                <p style="margin: 0; font-size: 1rem; color: {profit_color}; font-weight: 500;">
                                    {profit:+,.2f} ({profit_pct:+.2f}%)
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        total_return = backtest_results['total_return']
                        return_color = "#00d2ff" if total_return >= 0 else "#ff4b4b"
                        st.markdown(f"""
                            <div class="metric-card">
                                <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Total Return</p>
                                <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">{total_return:.1f}%</h2>
                                <p style="margin: 0; font-size: 1rem; color: {return_color}; font-weight: 500;">
                                    {' ↑' if total_return >= 0 else ' ↓'}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        win_rate = backtest_results['win_rate']
                        # Color coding for win rate
                        wr_color = "#00d2ff"
                        if win_rate < 40:
                            wr_color = "#ff4b4b"
                        elif win_rate < 50:
                            wr_color = "#ff9e00"
                            
                        st.markdown(f"""
                            <div class="metric-card">
                                <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Win Rate</p>
                                <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">{win_rate:.1f}%</h2>
                                <div style="width: 100%; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden;">
                                    <div style="width: {win_rate}%; height: 100%; background: {wr_color};"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        total_trades = backtest_results['total_trades']
                        profitable_trades = int((win_rate / 100) * total_trades)
                        
                        st.markdown(f"""
                            <div class="metric-card">
                                <p style="margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Trades</p>
                                <h2 style="margin: 10px 0; font-size: 1.8rem; font-weight: 700;">{total_trades}</h2>
                                <p style="margin: 0; font-size: 1rem; color: rgba(255, 255, 255, 0.7); font-weight: 500;">
                                    {profitable_trades} profitable
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display backtest chart with improved styling
                    if len(backtest_results['history']) > 0:
                        st.markdown("""
                            <div style="background: rgba(38, 39, 48, 0.5); padding: 20px; border-radius: 12px; margin: 25px 0; border: 1px solid rgba(255, 255, 255, 0.05);">
                                <h3 style="margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 600;">Portfolio Performance</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig = create_backtest_chart(backtest_results['history'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display trades table with modern styling
                        if len(backtest_results['history']) > 0:
                            trades_df = backtest_results['history'][backtest_results['history']['trade_taken']]
                            
                            # Calculate profit percentage
                            trades_df['profit_percentage'] = (trades_df['trade_profit'] / trades_df['open_price']) * 100
                            
                            # Format the table data
                            trades_table = pd.DataFrame({
                                'Date': trades_df['date'],
                                'Prediction': trades_df['predicted_direction'],
                                'Portfolio Value Before': trades_df['portfolio_value'].shift(1).fillna(initial_investment).map('${:,.2f}'.format),
                                'Opening Price': trades_df['open_price'].map('${:,.2f}'.format),
                                'Entry Price': trades_df['entry_price'].map('${:,.2f}'.format),
                                'Exit Price': trades_df['exit_price'].map('${:,.2f}'.format),
                                'Profit($)': trades_df['trade_profit'].map('${:,.2f}'.format),
                                'Profit(%)': trades_df.apply(lambda x: '{:,.2f}%'.format((x['trade_profit'] / x['position_size'] * 100) if x['position_size'] > 0 else 0), axis=1),
                                'Portfolio Value After': trades_df['portfolio_value'].map('${:,.2f}'.format)
                            })
                            
                            st.markdown("""
                                <div style="background: rgba(38, 39, 48, 0.5); padding: 20px; border-radius: 12px; margin: 25px 0; border: 1px solid rgba(255, 255, 255, 0.05);">
                                    <h3 style="margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 600;">Trading History</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.dataframe(
                                trades_table,
                                use_container_width=True,
                                column_config={
                                    "Profit($)": st.column_config.NumberColumn(
                                        "Profit($)",
                                        format="$%.2f",
                                    ),
                                    "Profit(%)": st.column_config.NumberColumn(
                                        "Profit(%)",
                                        format="%.2f%%",
                                    ),
                                }
                            )
                            
                            # Add a download button for the trade history
                            csv = trades_table.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Trading History",
                                csv,
                                f"{ticker}_backtest_results.csv",
                                "text/csv",
                                key="download-backtest-csv"
                            )
                else:
                    # Show informational message when no backtest has been run
                    st.markdown("""
                        <div style="background: rgba(38, 39, 48, 0.5); padding: 30px; border-radius: 12px; margin: 25px 0; text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 20px;">📊</div>
                            <h3 style="margin: 10px 0; font-size: 1.3rem; font-weight: 600;">Run a Backtest Simulation</h3>
                            <p style="color: rgba(255, 255, 255, 0.7); margin: 15px 0 25px 0;">
                                Test how our AI prediction model would have performed with historical data.
                                Configure your parameters and click the button above to start.
                            </p>
                            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin: 0 auto; max-width: 600px;">
                                <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; text-align: left; width: 45%;">
                                    <h4 style="margin: 0 0 10px 0; font-size: 1rem; color: #00d2ff;">How It Works</h4>
                                    <p style="font-size: 0.9rem; margin: 0; color: rgba(255, 255, 255, 0.7);">
                                        Our backtesting engine simulates trades using historical data to evaluate strategy performance.
                                    </p>
                                </div>
                                <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; text-align: left; width: 45%;">
                                    <h4 style="margin: 0 0 10px 0; font-size: 1rem; color: #00d2ff;">What You'll See</h4>
                                    <p style="font-size: 0.9rem; margin: 0; color: rgba(255, 255, 255, 0.7);">
                                        Performance metrics, equity curve, and detailed trade history with profit/loss analysis.
                                    </p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                        
        else:
            st.error("No data available for this symbol.")
            
    except Exception as e:
        st.error(f"Error analyzing asset: {str(e)}")