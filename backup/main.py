import streamlit as st
import pandas as pd
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
    page_title="Stock Analysis Platform",
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

# Title and search
st.title("Stock Analysis Platform")
st.markdown("### Advanced Technical & Fundamental Analysis")

# Create search box with autocomplete
search_query = st.text_input("Search stocks and crypto", value="", placeholder="Enter symbol (e.g., AAPL) or name")

if search_query:
    with st.spinner('Searching...'):
        search_results = search_stocks(search_query)
        if search_results:
            options = [f"{r['symbol']} - {r['name']} ({r['exchange']})" for r in search_results]
            selected = st.selectbox("Select Asset", options)
            ticker = selected.split(' - ')[0] if selected else None
        else:
            if st.session_state.get('last_error'):
                st.error(st.session_state['last_error'])
            else:
                st.warning("No matching assets found. Try entering a valid stock symbol (e.g., AAPL) or cryptocurrency (e.g., BTC-USD)")
            ticker = None
else:
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
            # Display asset info header
            st.header(info.get('name', ticker))
            
            # Layout columns for metrics
            col1, col2, col3 = st.columns(3)
            
            current_price = df['Close'].iloc[-1]
            price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
            with col2:
                st.metric("Market Cap", format_market_cap(info.get('market_cap', 0)))
            with col3:
                if is_crypto(ticker):
                    st.metric("24h Volume", f"${info.get('volume_24h', 0):,.0f}")
                else:
                    st.metric("P/E Ratio", f"{info.get('pe_ratio', 0):.2f}")
            
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
                st.markdown("### Price Prediction Model")
                st.markdown("Our advanced price prediction model utilizes a combination of machine learning techniques, technical indicators, seasonal patterns, and market regime analysis.")
                
                # Show loading state while predictions are being generated
                with st.spinner("Generating price predictions..."):
                    predictions = get_prediction(df, ticker)
                
                # Display the predictions in order
                timeframe_order = ['daily', 'short_term', 'medium_term', 'long_term', 'extended_term']
                for timeframe in timeframe_order:
                    if timeframe in predictions:
                        pred = predictions[timeframe]
                        st.subheader(f"{pred['timeframe']} Forecast")
                        
                        # Create two rows for better layout
                        top_cols = st.columns(3)
                        bottom_cols = st.columns(2)
                        
                        # Top row: Direction, Confidence, Forecast Price
                        with top_cols[0]:
                            direction_color = "green" if pred['direction'] == "UP" else ("red" if pred['direction'] == "DOWN" else "gray")
                            st.markdown(f"<h4 style='color:{direction_color};'>Direction: {pred['direction']}</h4>", unsafe_allow_html=True)
                        
                        with top_cols[1]:
                            confidence_val = pred.get('confidence', 0) * 100
                            conf_color = "green" if confidence_val > 70 else ("orange" if confidence_val > 50 else "gray")
                            st.markdown(f"<h4 style='color:{conf_color};'>Confidence: {confidence_val:.1f}%</h4>", unsafe_allow_html=True)
                        
                        with top_cols[2]:
                            forecast_val = pred.get('forecast')
                            if forecast_val is not None:
                                current_price = df['Close'].iloc[-1]
                                change = ((forecast_val - current_price) / current_price) * 100
                                change_str = f" ({change:+.2f}%)"
                                forecast_display = f"${forecast_val:.2f}{change_str}"
                            else:
                                forecast_display = "N/A"
                            st.markdown(f"<h4>Target Price: {forecast_display}</h4>", unsafe_allow_html=True)
                        
                        # Bottom row: Range (High and Low)
                        with bottom_cols[0]:
                            high_val = pred.get('predicted_high')
                            high_display = f"${high_val:.2f}" if high_val is not None else "N/A"
                            st.metric("Predicted High", high_display)
                        
                        with bottom_cols[1]:
                            low_val = pred.get('predicted_low')
                            low_display = f"${low_val:.2f}" if low_val is not None else "N/A"
                            st.metric("Predicted Low", low_display)
                        
                        # Add separator between timeframes
                        st.markdown("---")
                
            with tabs[2]:
                news = get_news(ticker)
                if news:
                    for article in news:
                        # Safely get values with defaults
                        title = article.get('title', 'No title available')
                        summary = article.get('summary', 'No summary available')
                        source = article.get('source', 'Unknown source')
                        sentiment_val = article.get('sentiment', 0)
                        
                        # Format sentiment
                        sentiment, color = format_news_sentiment(sentiment_val)
                        
                        # Display article
                        st.markdown(
                            f"""
                            <div style='padding: 10px; border-left: 5px solid {color}; margin: 10px 0;'>
                                <h4>{title}</h4>
                                <p>{summary}</p>
                                <small>Source: {source} | Sentiment: {sentiment}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No news articles found for this asset.")
            
            with tabs[3]:
                st.subheader("Strategy Backtesting")
                initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000)
                
                if st.button("Run Backtest"):
                    backtest_results = backtest_prediction_model(df, initial_investment)
                    
                    # Display backtest metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Final Portfolio Value", f"${backtest_results['final_value']:,.2f}")
                    with metric_cols[1]:
                        st.metric("Total Return", f"{backtest_results['total_return']:.1f}%")
                    with metric_cols[2]:
                        st.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
                    with metric_cols[3]:
                        st.metric("Total Trades", backtest_results['total_trades'])
                    
                    # Display backtest chart
                    if len(backtest_results['history']) > 0:
                        fig = create_backtest_chart(backtest_results['history'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display trades table
                        if len(backtest_results['history']) > 0:
                            trades_df = backtest_results['history'][backtest_results['history']['trade_taken']]
                            
                            # Calculate profit percentage
                            trades_df['profit_percentage'] = (trades_df['trade_profit'] / trades_df['open_price']) * 100
                            
                            # Format the table data
                            # Format trades table with proper price fields and calculations
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
                            
                            st.subheader("Trading History")
                            st.dataframe(trades_table, use_container_width=True)
                        
        else:
            st.error("No data available for this symbol.")
            
    except Exception as e:
        st.error(f"Error analyzing asset: {str(e)}")