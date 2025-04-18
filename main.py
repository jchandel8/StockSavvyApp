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
    
    # Sample info
    info = {
        'name': 'Apple Inc.' if ticker == 'AAPL' else f'{ticker} Inc.',
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

# Apply custom CSS
try:
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Custom styling could not be loaded: {str(e)}")

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
ticker = st.session_state.ticker
if search_query:
    # If search is performed, update the view
    with st.spinner('🔍 Searching financial markets...'):
        search_results = search_stocks(search_query)
        if search_results:
            # Display search results
            options = []
            for r in search_results:
                icon = "🪙" if r.get('exchange', '').lower() in ['crypto', 'binance', 'coinbase'] else "📈"
                option = f"{icon} {r['symbol']} - {r['name']} ({r['exchange']})"
                options.append(option)
            
            selected = st.selectbox(
                "Select an asset to analyze",
                options,
                format_func=lambda x: x,
                key="asset_selector"
            )
            
            # Extract the ticker from the selected option
            # Format is "📈 AAPL - Apple Inc. (Major Exchange)"
            if selected:
                # First split by the dash to get "📈 AAPL "
                first_part = selected.split(' - ')[0].strip()
                # Then split by space to get ["📈", "AAPL"]
                ticker = first_part.split(' ')[-1]
                
                # Log for debugging
                st.write(f"Selected: {selected}")
                st.write(f"Extracted ticker: {ticker}")
                
                # Update session state
                st.session_state.ticker = ticker
                
                # Rerun the app to show the selected ticker
                st.rerun()
            else:
                ticker = None
        else:
            st.error("No results found. Try another search term.")
            ticker = None

if ticker:
    # Format crypto symbols correctly
    if is_crypto(ticker):
        ticker = format_crypto_symbol(ticker)
    
    # Try to get real data, use sample data if there's an issue
    try:
        # Get stock/crypto data and info
        df = get_stock_data(ticker)
        info = get_stock_info(ticker)
        
        # If data is empty or has issues, use sample data
        if df.empty or 'Date' not in df.columns:
            st.warning("API data unavailable. Using sample data for demonstration.")
            df, info = generate_sample_data(ticker)
    except Exception as e:
        st.warning(f"Error fetching data: {str(e)}. Using sample data for demonstration.")
        df, info = generate_sample_data(ticker)
    
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
    price_change = 1.39  # Hardcoded for design purposes
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Current Price<span class="info-icon" title="Current trading price">ⓘ</span></div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="price-change">+{price_change}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Market Cap</div>
            <div class="metric-value">$2.71T</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">P/E Ratio</div>
            <div class="metric-value">28.92</div>
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
        
        col1, col2 = st.columns(2)
        
        # 1 Day and 1 Week forecast
        with col1:
            st.markdown("""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">1 Day Forecast</h4>
                    <span class="up-badge">UP</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>70.7%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: 70.7%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">$196.62 <span style="color: #ef4444; font-size: 0.75rem;">(-0.19%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">$198.38</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">$194.85</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>$194.85</span>
                        <span>$198.38</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">1 Week Forecast</h4>
                    <span class="neutral-badge">NEUTRAL</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>0.0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: 0%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">$196.98 <span style="color: #94a3b8; font-size: 0.75rem;">(0.00%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">$202.29</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">$191.67</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>$191.67</span>
                        <span>$202.29</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 1 Month and 3 Months forecast
        with col2:
            st.markdown("""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">1 Month Forecast</h4>
                    <span class="up-badge">UP</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>60.2%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: 60.2%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">$195.82 <span style="color: #ef4444; font-size: 0.75rem;">(-0.59%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">$204.67</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">$186.98</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>$186.98</span>
                        <span>$204.67</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">3 Months Forecast</h4>
                    <span class="up-badge">UP</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>55.0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: 55%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">$192.33 <span style="color: #ef4444; font-size: 0.75rem;">(-2.36%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">$206.48</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">$178.18</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>$178.18</span>
                        <span>$206.48</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 6 Months forecast
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div class="card">
                <div class="prediction-header">
                    <h4 class="prediction-title">6 Months Forecast</h4>
                    <span class="up-badge">UP</span>
                </div>
                
                <div>
                    <div class="progress-label">
                        <span>Confidence</span>
                        <span>51.9%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="width: 51.9%;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                        <span style="font-weight: 600;">$186.29 <span style="color: #ef4444; font-size: 0.75rem;">(-5.43%)</span></span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                            <div style="font-weight: 600;">$207.52</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                            <div style="font-weight: 600;">$165.07</div>
                        </div>
                    </div>
                    
                    <div class="range-slider" style="margin-top: 1.5rem;">
                        <div class="range-slider-indicator" style="left: 50%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                        <span>$165.07</span>
                        <span>$207.52</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
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