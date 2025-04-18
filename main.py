import streamlit as st
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.stock_data import get_stock_data, get_stock_info, search_stocks, is_crypto, format_crypto_symbol
from utils.technical_analysis import generate_signals, calculate_indicators
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

# Function to generate sample stock data for design purposes
def generate_sample_data(ticker="AAPL"):
    # Generate sample data for demonstration purposes
    dates = pd.date_range(start='2024-01-01', end='2025-03-31', freq='D')
    np.random.seed(42)
    
    # Generate price data
    base_price = 180
    price = base_price + np.cumsum(np.random.normal(0, 1, len(dates))) * 0.5
    
    # Ensure the final price is 196.98 as in the screenshots
    price = price * (196.98 / price[-1])
    
    # Generate volume data
    volume = np.random.normal(100000000, 20000000, len(dates))
    volume = np.abs(volume)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': price - np.random.normal(0, 1, len(dates)),
        'High': price + np.random.normal(1, 0.5, len(dates)),
        'Low': price - np.random.normal(1, 0.5, len(dates)),
        'Close': price,
        'Volume': volume
    })
    
    # Sample info with proper company names based on ticker
    company_names = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com, Inc.',
        'META': 'Meta Platforms, Inc.',
        'TSLA': 'Tesla, Inc.',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'NFLX': 'Netflix, Inc.',
        'DIS': 'The Walt Disney Company'
    }
    
    info = {
        'name': company_names.get(ticker, f'{ticker} Inc.'),
        'market_cap': 2710000000000,  # $2.71T
        'pe_ratio': 28.92,
        'exchange': 'NASDAQ'
    }
    
    return df, info

# Initialize session state if not already done
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_error = None
    st.session_state.ticker = "AAPL"  # Default ticker
    st.session_state.tab = "Technical Analysis"  # Default tab
    st.session_state.backtest_strategy = "Momentum"
    st.session_state.backtest_risk = "Medium"
    st.session_state.backtest_capital = 10000
    st.session_state.backtest_start = datetime.datetime(2024, 1, 1).date()
    st.session_state.backtest_end = datetime.datetime(2025, 3, 31).date()

# Apply custom CSS for styling
def load_css_file(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        return True
    except Exception as e:
        st.warning(f"Error loading CSS from {file_path}: {str(e)}")
        return False

# Load main styling
main_css_loaded = load_css_file('styles/style.css')

# Load prediction specific styling with improved error handling
prediction_css_loaded = load_css_file('styles/prediction.css')

# Apply inline CSS as fallback for prediction styling if the file can't be loaded
if not prediction_css_loaded:
    st.markdown("""
    <style>
    /* Fallback styles for prediction cards */
    .prediction-header { display: flex !important; justify-content: space-between !important; align-items: center !important; margin-bottom: 1rem !important; }
    .prediction-title { font-size: 1.125rem !important; font-weight: 600 !important; margin: 0 !important; }
    
    .up-badge { display: inline-block !important; background-color: rgba(16, 185, 129, 0.2) !important; color: #10b981 !important; 
               border-radius: 9999px !important; padding: 0.25rem 0.75rem !important; font-size: 0.75rem !important; font-weight: 500 !important; }
    .down-badge { display: inline-block !important; background-color: rgba(239, 68, 68, 0.2) !important; color: #ef4444 !important; 
                border-radius: 9999px !important; padding: 0.25rem 0.75rem !important; font-size: 0.75rem !important; font-weight: 500 !important; }
    .neutral-badge { display: inline-block !important; background-color: rgba(148, 163, 184, 0.2) !important; color: #94a3b8 !important; 
                   border-radius: 9999px !important; padding: 0.25rem 0.75rem !important; font-size: 0.75rem !important; font-weight: 500 !important; }
    
    .progress-label { display: flex !important; justify-content: space-between !important; margin-bottom: 0.25rem !important; font-size: 0.875rem !important; }
    .progress-bar { height: 0.5rem !important; background-color: rgba(255, 255, 255, 0.1) !important; border-radius: 9999px !important; overflow: hidden !important; }
    .progress-value { height: 100% !important; background-color: #10b981 !important; }
    
    .range-slider { width: 100% !important; height: 0.25rem !important; background-color: rgba(255, 255, 255, 0.1) !important; 
                  border-radius: 9999px !important; position: relative !important; margin: 1rem 0 !important; }
    .range-slider-indicator { position: absolute !important; width: 0.5rem !important; height: 0.5rem !important; background-color: #10b981 !important; 
                            border-radius: 50% !important; top: -0.125rem !important; transform: translateX(-50%) !important; }
    </style>
    """, unsafe_allow_html=True)

# Header section with logo and search
st.markdown("""
<div class="header-container">
    <div class="logo-container">
        <div class="logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>
                <polyline points="16 7 22 7 22 13"></polyline>
            </svg>
        </div>
        <div>
            <h1 style="margin: 0; font-size: 1.25rem; font-weight: bold;">StockSavvy</h1>
            <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Advanced Technical & Fundamental Analysis</p>
        </div>
    </div>
    <div class="search-container">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
""", unsafe_allow_html=True)

search_query = st.text_input("", value="", placeholder="Search stocks and crypto...", label_visibility="collapsed")

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

# Set default ticker to render the app
ticker = None

# Check if a ticker is already in session state
if 'ticker' in st.session_state:
    ticker = st.session_state.ticker

# Process the search query
if search_query:
    # If search is performed, update the view
    with st.spinner('🔍 Searching financial markets...'):
        search_results = search_stocks(search_query)
        if search_results:
            # Display search results
            options = []
            symbols = []
            
            for r in search_results:
                icon = "🪙" if r.get('exchange', '').lower() in ['crypto', 'binance', 'coinbase'] else "📈"
                option_text = f"{icon} {r['symbol']} - {r['name']} ({r['exchange']})"
                options.append(option_text)
                symbols.append(r['symbol'])
            
            # Create the selectbox for results
            selected_idx = st.selectbox(
                "Select an asset to analyze",
                range(len(options)),
                format_func=lambda i: options[i],
                key="asset_selector"
            )
            
            # Set the selected ticker to the corresponding symbol
            if 0 <= selected_idx < len(symbols):
                # Only update if it's a new ticker
                new_ticker = symbols[selected_idx]
                if new_ticker != ticker:
                    ticker = new_ticker
                    st.session_state.ticker = new_ticker
                    st.rerun()
        else:
            st.error("No results found. Try another search term.")

if ticker:
    # Format crypto symbols correctly
    if is_crypto(ticker):
        ticker = format_crypto_symbol(ticker)
    
    # Get real stock/crypto data using Alpha Vantage API
    df = get_stock_data(ticker)
    info = get_stock_info(ticker)
    
    # Add the current date as a Date column if needed
    if 'Date' not in df.columns:
        df['Date'] = df.index
    
    # Stock symbol and name header
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="stock-header">
            <div class="stock-symbol">
                {ticker}
            </div>
            <div>
                <h2 class="stock-name">{info.get('name', ticker)}</h2>
                <span class="stock-exchange">Major Exchange</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics section
    current_price = df['Close'].iloc[-1]
    
    # Calculate price change over last day
    if len(df) > 1:
        prev_price = df['Close'].iloc[-2]
        price_change_pct = ((current_price - prev_price) / prev_price) * 100
    else:
        price_change_pct = 0
        
    # Determine if price is up or down
    price_change_sign = "+" if price_change_pct >= 0 else ""
    price_change_color = "#10b981" if price_change_pct >= 0 else "#ef4444"
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Current Price<span class="info-icon" title="Current trading price">ⓘ</span></div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="price-change" style="color: {price_change_color}">{price_change_sign}{price_change_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get market cap from stock info or calculate it
    market_cap = info.get('market_cap', 0)
    if market_cap == 0 and 'shares_outstanding' in info:
        market_cap = current_price * info.get('shares_outstanding', 0)
        
    # Format market cap
    if market_cap >= 1e12:  # Trillion
        market_cap_formatted = f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:  # Billion
        market_cap_formatted = f"${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:  # Million
        market_cap_formatted = f"${market_cap/1e6:.2f}M"
    else:
        market_cap_formatted = f"${market_cap:,.0f}"
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Market Cap</div>
            <div class="metric-value">{market_cap_formatted}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get P/E ratio
    pe_ratio = info.get('pe_ratio', info.get('trailingPE', '-'))
    pe_ratio = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else '-'
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">P/E Ratio</div>
            <div class="metric-value">{pe_ratio}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab navigation
    tab_titles = ["Technical Analysis", "Price Prediction", "News", "Backtesting"]
    tabs = st.tabs(tab_titles)
    
    # Technical Analysis Tab
    with tabs[0]:
        col1, col2 = st.columns([7, 3])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; margin-bottom: 1rem;">Stock Price Chart</h3>
            """, unsafe_allow_html=True)
            
            # Create price chart using plotly
            df_with_indicators = calculate_indicators(df)
            
            # Make sure we have a Date column for plotting
            if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df['Date'] = df.index
                df_with_indicators['Date'] = df.index
            
            # Create subplot with 2 rows
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Add price chart
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], 
                    y=df['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#10b981', width=2)
                ),
                row=1, col=1
            )
            
            # Add SMA lines
            if 'SMA_20' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'], 
                        y=df_with_indicators['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#3b82f6', width=1.5, dash='dash')
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'], 
                        y=df_with_indicators['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='#f59e0b', width=1.5, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Add volume chart
            fig.add_trace(
                go.Bar(
                    x=df['Date'], 
                    y=df['Volume'],
                    name='Volume',
                    marker=dict(color='#3b82f6', opacity=0.5)
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#1a2234'
                ),
                xaxis2=dict(
                    showgrid=True,
                    gridcolor='#1a2234',
                    showticklabels=True
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#1a2234'
                ),
                yaxis2=dict(
                    showgrid=True,
                    gridcolor='#1a2234'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Technical Analysis Summary panel
            st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; margin-bottom: 1rem;">Technical Analysis Summary</h3>
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <span>Current Price</span>
                    <span style="font-weight: 600;">$196.98</span>
                </div>
                
                <div style="height: 1px; background-color: #1a2234; margin: 1rem 0;"></div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <div style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.25rem;">RSI</div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-weight: 600;">40.93</span>
                            <span class="badge-yellow badge">Neutral</span>
                        </div>
                    </div>
                    
                    <div>
                        <div style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.25rem;">MACD</div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-weight: 600;">-7.78</span>
                            <span class="badge-red badge">Bearish</span>
                        </div>
                    </div>
                </div>
                
                <div style="height: 1px; background-color: #1a2234; margin: 1rem 0;"></div>
                
                <h4 style="font-weight: 500; margin-bottom: 0.5rem;">Buy Signals</h4>
                <div style="display: flex; align-items: center; gap: 0.5rem; color: #94a3b8; margin-bottom: 1rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <span style="font-size: 0.875rem;">No buy signals detected</span>
                </div>
                
                <h4 style="font-weight: 500; margin-bottom: 0.5rem;">Sell Signals</h4>
                <div style="display: flex; align-items: center; gap: 0.5rem; color: #94a3b8; margin-bottom: 1rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <span style="font-size: 0.875rem;">No sell signals detected</span>
                </div>
            </div>
            
            <div class="card">
                <h3 style="margin-top: 0; margin-bottom: 1rem;">Moving Averages</h3>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span>SMA 20</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 600;">$204.56</span>
                        <span class="badge-red badge">Below</span>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span>SMA 50</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 600;">$210.32</span>
                        <span class="badge-red badge">Below</span>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>SMA 200</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 600;">$190.45</span>
                        <span class="badge-green badge">Above</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Price Prediction Tab
    with tabs[1]:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; margin-bottom: 0.5rem;">Price Prediction Model</h3>
            <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem;">Our advanced price prediction model utilizes a combination of machine learning techniques, technical indicators, seasonal patterns, and market regime analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get price predictions for this stock
        predictions = get_prediction(df, ticker)
        
        # Display prediction cards in columns
        # First row - 1 Day and 1 Week
        col1, col2 = st.columns(2)
        
        # 1 Day and 1 Week forecast
        with col1:
            # Get 1-day prediction data
            one_day_pred = predictions.get("short_term", {})
            pred_price = one_day_pred.get("price", current_price * 1.01)  # Default to 1% up if missing
            confidence = one_day_pred.get("confidence", 50.0)
            direction = one_day_pred.get("direction", "UP")
            
            # Calculate percentage change
            price_change = ((pred_price - current_price) / current_price) * 100
            price_change_color = "#10b981" if price_change >= 0 else "#ef4444"
            price_change_sign = "+" if price_change >= 0 else ""
            
            # Direction badge
            if direction == "UP":
                direction_badge = "up-badge"
                direction_text = "UP"
            elif direction == "DOWN":
                direction_badge = "down-badge"
                direction_text = "DOWN"
            else:
                direction_badge = "neutral-badge"
                direction_text = "NEUTRAL"
                
            # Calculate high and low predictions
            pred_high = one_day_pred.get("high", pred_price * 1.01)
            pred_low = one_day_pred.get("low", pred_price * 0.99)
            
            # Create the card with dynamic data
            one_day_card = f"""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">1 Day Forecast</h4>
                    <span class="{direction_badge}">{direction_text}</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>{confidence:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: {confidence}%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">${pred_price:.2f} <span style="color: {price_change_color}; font-size: 0.75rem;">({price_change_sign}{price_change:.2f}%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">${pred_high:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">${pred_low:.2f}</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>${pred_low:.2f}</span>
                        <span>${pred_high:.2f}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(one_day_card, unsafe_allow_html=True)
            
            # 1 Week Forecast Card
            # Get 1-week prediction data
            one_week_pred = predictions.get("medium_term", {})
            pred_price_week = one_week_pred.get("price", current_price * 1.02)  # Default to 2% up if missing
            confidence_week = one_week_pred.get("confidence", 50.0)
            direction_week = one_week_pred.get("direction", "UP")
            
            # Calculate percentage change
            price_change_week = ((pred_price_week - current_price) / current_price) * 100
            price_change_color_week = "#10b981" if price_change_week >= 0 else "#ef4444"
            price_change_sign_week = "+" if price_change_week >= 0 else ""
            
            # Direction badge
            if direction_week == "UP":
                direction_badge_week = "up-badge"
                direction_text_week = "UP"
            elif direction_week == "DOWN":
                direction_badge_week = "down-badge"
                direction_text_week = "DOWN"
            else:
                direction_badge_week = "neutral-badge"
                direction_text_week = "NEUTRAL"
                
            # Calculate high and low predictions
            pred_high_week = one_week_pred.get("high", pred_price_week * 1.02)
            pred_low_week = one_week_pred.get("low", pred_price_week * 0.98)
            
            # Create the card with dynamic data
            one_week_card = f"""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">1 Week Forecast</h4>
                    <span class="{direction_badge_week}">{direction_text_week}</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>{confidence_week:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: {confidence_week}%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">${pred_price_week:.2f} <span style="color: {price_change_color_week}; font-size: 0.75rem;">({price_change_sign_week}{price_change_week:.2f}%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">${pred_high_week:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">${pred_low_week:.2f}</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>${pred_low_week:.2f}</span>
                        <span>${pred_high_week:.2f}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(one_week_card, unsafe_allow_html=True)
        
        # 1 Month and 3 Months forecast
        with col2:
            # Get 1-month prediction data
            one_month_pred = predictions.get("long_term", {})
            pred_price_month = one_month_pred.get("price", current_price * 1.03)  # Default to 3% up if missing
            confidence_month = one_month_pred.get("confidence", 60.0)
            direction_month = one_month_pred.get("direction", "UP")
            
            # Calculate percentage change
            price_change_month = ((pred_price_month - current_price) / current_price) * 100
            price_change_color_month = "#10b981" if price_change_month >= 0 else "#ef4444"
            price_change_sign_month = "+" if price_change_month >= 0 else ""
            
            # Direction badge
            if direction_month == "UP":
                direction_badge_month = "up-badge"
                direction_text_month = "UP"
            elif direction_month == "DOWN":
                direction_badge_month = "down-badge"
                direction_text_month = "DOWN"
            else:
                direction_badge_month = "neutral-badge"
                direction_text_month = "NEUTRAL"
                
            # Calculate high and low predictions
            pred_high_month = one_month_pred.get("high", pred_price_month * 1.04)
            pred_low_month = one_month_pred.get("low", pred_price_month * 0.96)
            
            # Create the card with dynamic data
            one_month_card = f"""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">1 Month Forecast</h4>
                    <span class="{direction_badge_month}">{direction_text_month}</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>{confidence_month:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: {confidence_month}%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">${pred_price_month:.2f} <span style="color: {price_change_color_month}; font-size: 0.75rem;">({price_change_sign_month}{price_change_month:.2f}%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">${pred_high_month:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">${pred_low_month:.2f}</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>${pred_low_month:.2f}</span>
                        <span>${pred_high_month:.2f}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(one_month_card, unsafe_allow_html=True)
            
            # 3 Months Forecast Card
            # Get 3-months prediction data
            three_month_pred = predictions.get("extended_term", {})
            pred_price_3m = three_month_pred.get("price", current_price * 1.05)  # Default to 5% up if missing
            confidence_3m = three_month_pred.get("confidence", 55.0)
            direction_3m = three_month_pred.get("direction", "UP")
            
            # Calculate percentage change
            price_change_3m = ((pred_price_3m - current_price) / current_price) * 100
            price_change_color_3m = "#10b981" if price_change_3m >= 0 else "#ef4444"
            price_change_sign_3m = "+" if price_change_3m >= 0 else ""
            
            # Direction badge
            if direction_3m == "UP":
                direction_badge_3m = "up-badge"
                direction_text_3m = "UP"
            elif direction_3m == "DOWN":
                direction_badge_3m = "down-badge"
                direction_text_3m = "DOWN"
            else:
                direction_badge_3m = "neutral-badge"
                direction_text_3m = "NEUTRAL"
                
            # Calculate high and low predictions
            pred_high_3m = three_month_pred.get("high", pred_price_3m * 1.07)
            pred_low_3m = three_month_pred.get("low", pred_price_3m * 0.93)
            
            # Create the card with dynamic data
            three_months_card = f"""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">3 Months Forecast</h4>
                    <span class="{direction_badge_3m}">{direction_text_3m}</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>{confidence_3m:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: {confidence_3m}%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">${pred_price_3m:.2f} <span style="color: {price_change_color_3m}; font-size: 0.75rem;">({price_change_sign_3m}{price_change_3m:.2f}%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">${pred_high_3m:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">${pred_low_3m:.2f}</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>${pred_low_3m:.2f}</span>
                        <span>${pred_high_3m:.2f}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(three_months_card, unsafe_allow_html=True)
        
        # 6 Months forecast
        col1, col2 = st.columns([1, 1])
        with col1:
            # Get 6-months prediction data
            six_month_pred = predictions.get("long_extended_term", {})
            pred_price_6m = six_month_pred.get("price", current_price * 1.07)  # Default to 7% up if missing
            confidence_6m = six_month_pred.get("confidence", 51.9)
            direction_6m = six_month_pred.get("direction", "UP")
            
            # Calculate percentage change
            price_change_6m = ((pred_price_6m - current_price) / current_price) * 100
            price_change_color_6m = "#10b981" if price_change_6m >= 0 else "#ef4444"
            price_change_sign_6m = "+" if price_change_6m >= 0 else ""
            
            # Direction badge
            if direction_6m == "UP":
                direction_badge_6m = "up-badge"
                direction_text_6m = "UP"
            elif direction_6m == "DOWN":
                direction_badge_6m = "down-badge"
                direction_text_6m = "DOWN"
            else:
                direction_badge_6m = "neutral-badge"
                direction_text_6m = "NEUTRAL"
                
            # Calculate high and low predictions
            pred_high_6m = six_month_pred.get("high", pred_price_6m * 1.10)
            pred_low_6m = six_month_pred.get("low", pred_price_6m * 0.90)
            
            # Create the card with dynamic data
            six_months_card = f"""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">6 Months Forecast</h4>
                    <span class="{direction_badge_6m}">{direction_text_6m}</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>{confidence_6m:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: {confidence_6m}%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">${pred_price_6m:.2f} <span style="color: {price_change_color_6m}; font-size: 0.75rem;">({price_change_sign_6m}{price_change_6m:.2f}%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">${pred_high_6m:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">${pred_low_6m:.2f}</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>${pred_low_6m:.2f}</span>
                        <span>${pred_high_6m:.2f}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(six_months_card, unsafe_allow_html=True)
    
    # News Tab
    with tabs[2]:
        # Sample news data for design purposes
        news_data = [
            {
                "title": "TSMC Confident Amid Trump Tariff Turmoil: CEO CC Wei Says No Change In Customer Behavior",
                "subtitle": "Despite Nvidia H20 Chip Curbs - NVIDIA (NASDAQ:NVDA), Apple (NASDAQ:AAPL)",
                "content": "Taiwan Semiconductor Manufacturing Co. Ltd. TSM remains optimistic despite ongoing geopolitical tensions. CEO CC Wei stated that customer behavior has not changed, even amid recent challenges such as the Nvidia Corporation NVDA H20 chip facing clamdown and tariff-related uncertainties.",
                "source": "Benzinga",
                "sentiment": "Neutral",
                "date": "4/17/2025"
            },
            {
                "title": "Top Analyst Reports for Apple, Philip Morris & Sony",
                "subtitle": "",
                "content": "Today's Research Daily features new research reports on 16 major stocks, including Bank of Apple Inc. (AAPL), Philip Morris International Inc. (PM) and Sony Group Corp. (SONY), as well as a micro-cap stock Vaso Corp. (VASO).",
                "source": "Zacks Commentary",
                "sentiment": "Neutral",
                "date": "4/17/2025"
            },
            {
                "title": "Google Scores Partial Win In Antitrust Case But Faces Setback On Publisher Tools",
                "subtitle": "Alphabet (NASDAQ:GOOGL)",
                "content": "US Department of Justice wins antitrust case against Google, ruling it maintained a monopoly in ad tech through anticompetitive practices. Google faces appeals after court dismisses some antitrust claims but supports the Justice Department's push for a breakup.",
                "source": "Benzinga",
                "sentiment": "Neutral",
                "date": "4/17/2025"
            },
            {
                "title": "Temu And Shein To Raise Prices Amid US Tariffs, Slash Ad Spend On Apple, Meta",
                "subtitle": "PDD Holdings (NASDAQ:PDD)",
                "content": "Temu and Shein to raise prices starting April 26, shifting tariff burden onto consumers as U.S. tariffs on Chinese goods increase. Temu cuts U.S. ad spend and sees sharp decline in App Store rankings, while facing impact of 145% tariff on Chinese imports.",
                "source": "Benzinga",
                "sentiment": "Negative",
                "date": "4/17/2025"
            }
        ]
        
        for article in news_data:
            # Determine sentiment badge class
            if article["sentiment"] == "Positive":
                badge_class = "badge-green"
            elif article["sentiment"] == "Negative":
                badge_class = "badge-red"
            else:
                badge_class = "badge-neutral"
            
            st.markdown(f"""
            <div class="card">
                <h3 style="margin-top: 0; margin-bottom: 0.5rem;">{article["title"]}</h3>
                <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">{article["subtitle"]}</p>
                
                <p style="margin-bottom: 1rem;">{article["content"]}</p>
                
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge badge-blue">Source: {article["source"]}</span>
                        <span class="badge {badge_class}">Sentiment: {article["sentiment"]}</span>
                    </div>
                    <a href="#" style="color: #3b82f6; text-decoration: none; font-size: 0.875rem; display: flex; align-items: center; gap: 0.25rem;">
                        Read more
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                            <polyline points="12 5 19 12 12 19"></polyline>
                        </svg>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Backtesting Tab
    with tabs[3]:
        col1, col2 = st.columns([4, 8])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; margin-bottom: 1rem;">Backtest Parameters</h3>
                <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem;">Configure your strategy parameters</p>
            """, unsafe_allow_html=True)
            
            # Strategy selection
            strategy = st.selectbox(
                "Strategy",
                options=["Momentum", "Mean Reversion", "Trend Following", "Breakout", "Value"],
                index=0
            )
            
            # Time period
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<label>Time Period</label>", unsafe_allow_html=True)
            col1a, col1b = st.columns(2)
            with col1a:
                start_date = st.date_input("", value=datetime.datetime(2024, 1, 1).date(), label_visibility="collapsed")
            with col1b:
                end_date = st.date_input("", value=datetime.datetime(2025, 3, 31).date(), label_visibility="collapsed")
            
            # Initial capital
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<label>Initial Capital</label>", unsafe_allow_html=True)
            initial_capital = st.number_input("", value=10000, min_value=1000, max_value=1000000, step=1000, label_visibility="collapsed")
            
            # Risk level slider
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='display: flex; justify-content: space-between;'><label>Risk Level</label><span>Medium</span></div>", unsafe_allow_html=True)
            risk_level = st.slider("", min_value=0, max_value=100, value=50, label_visibility="collapsed")
            
            # Run backtest button
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            run_backtest = st.button("Run Backtest")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0; margin-bottom: 1rem;">Backtest Results</h3>
                <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem;">Strategy performance vs. benchmark</p>
            """, unsafe_allow_html=True)
            
            # Create a sample backtest chart
            dates = pd.date_range(start='2024-01-01', end='2025-03-31', freq='MS')
            
            strategy_returns = [10000]
            benchmark_returns = [10000]
            
            # Generate some sample return data
            for i in range(1, len(dates)):
                strategy_returns.append(strategy_returns[i-1] * (1 + np.random.normal(0.04, 0.05)))
                benchmark_returns.append(benchmark_returns[i-1] * (1 + np.random.normal(0.02, 0.03)))
            
            # Create the plot using Plotly
            fig = go.Figure()
            
            # Strategy line
            fig.add_trace(go.Scatter(
                x=dates,
                y=strategy_returns,
                mode='lines+markers',
                name='Momentum Strategy',
                line=dict(color='#10b981', width=2),
                marker=dict(size=6)
            ))
            
            # Benchmark line
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_returns,
                mode='lines+markers',
                name='S&P 500',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6)
            ))
            
            # Update the layout
            fig.update_layout(
                height=300,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#1a2234',
                    tickformat='%Y-%m'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#1a2234',
                    tickprefix='$',
                    tickformat=','
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics in a grid layout
            col2a, col2b, col2c, col2d = st.columns(4)
            
            with col2a:
                st.markdown("""
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Total Return</div>
                    <div style="font-weight: 600; font-size: 1.25rem;">75.0%</div>
                    <div style="color: #10b981; font-size: 0.75rem;">vs. 30.0%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2b:
                st.markdown("""
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Annualized Return</div>
                    <div style="font-weight: 600; font-size: 1.25rem;">56.3%</div>
                    <div style="color: #10b981; font-size: 0.75rem;">vs. 22.5%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2c:
                st.markdown("""
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Sharpe Ratio</div>
                    <div style="font-weight: 600; font-size: 1.25rem;">1.85</div>
                    <div style="color: #10b981; font-size: 0.75rem;">vs. 1.25</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2d:
                st.markdown("""
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Max Drawdown</div>
                    <div style="font-weight: 600; font-size: 1.25rem;">-8.3%</div>
                    <div style="color: #ef4444; font-size: 0.75rem;">vs. -4.5%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
else:
    # Display welcome message when no ticker is selected
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto; display: block; margin-bottom: 1.5rem;">
            <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>
            <polyline points="16 7 22 7 22 13"></polyline>
        </svg>
        <h2 style="margin-bottom: 1rem; color: #f8fafc; font-weight: 600;">Welcome to StockSavvy</h2>
        <p style="color: #94a3b8; max-width: 24rem; margin: 0 auto;">
            Search for any stock or cryptocurrency to view comprehensive analysis, price predictions, and trading signals.
        </p>
    </div>
    """, unsafe_allow_html=True)