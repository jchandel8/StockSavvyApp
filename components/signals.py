import streamlit as st
import pandas as pd

def get_signal_indicator(signal_type: str) -> None:
    """Display a colored indicator based on signal type using native Streamlit components"""
    if signal_type == 'buy':
        st.success("", icon="🔼")
    elif signal_type == 'sell':
        st.error("", icon="🔽")
    else:
        st.info("", icon="⚪")
        
def get_signal_badge(signal_type: str, text: str) -> None:
    """Display a colored badge with text using native Streamlit components"""
    if signal_type == 'buy':
        st.success(text)
    elif signal_type == 'sell':
        st.error(text)
    else:
        st.info(text)

def display_signals(signals: dict):
    col1, col2 = st.columns(2)
    
    status_messages = {
        'loading': "Loading signals...",
        'no_data': "No data available",
        'missing_data': "Insufficient data for signal generation",
        'invalid_data': "Invalid data detected",
        'error': "Error generating signals",
    }
    
    if signals.get('status') in status_messages and signals['status'] != 'complete':
        message = status_messages[signals['status']]
        with col1:
            st.subheader("Buy Signals")
            st.info(message)
        with col2:
            st.subheader("Sell Signals")
            st.info(message)
        return
    
    with col1:
        st.subheader("Buy Signals")
        if signals.get('buy_signals'):
            for signal in signals['buy_signals']:
                st.success(signal, icon="🔼")
        else:
            st.info("No buy signals detected", icon="ℹ️")
    
    with col2:
        st.subheader("Sell Signals")
        if signals.get('sell_signals'):
            for signal in signals['sell_signals']:
                st.error(signal, icon="🔽")
        else:
            st.info("No sell signals detected", icon="ℹ️")

def display_technical_summary(df):
    st.subheader("Technical Analysis Summary")
    
    try:
        # Define a safe getter function for handling NaN values and exceptions
        def safe_get(df, column, index=-1, default=None):
            try:
                value = df[column].iloc[index]
                return default if pd.isna(value) else value
            except (IndexError, KeyError, AttributeError, ValueError):
                return default
        
        # Get values safely with defaults
        current_price = safe_get(df, 'Close', default=0.0)
        rsi = safe_get(df, 'RSI', default=50.0)  # Neutral RSI default
        macd = safe_get(df, 'MACD', default=0.0)
        signal = safe_get(df, 'MACD_Signal', default=0.0)
        sma20 = safe_get(df, 'SMA20', default=0.0)
        sma50 = safe_get(df, 'SMA50', default=0.0)
        sma200 = safe_get(df, 'SMA200', default=0.0)
        
        # Display current price in larger font
        st.metric("Current Price", f"${current_price:.2f}")
        
        # Add a divider
        st.markdown("---")
        
        # Display RSI and MACD in a single row
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**RSI**")
            if rsi is not None:
                # Determine RSI status with native color coding
                if rsi < 30:
                    st.success(f"{rsi:.2f} - Oversold", icon="🔼")
                elif rsi > 70:
                    st.error(f"{rsi:.2f} - Overbought", icon="🔽")
                else:
                    st.info(f"{rsi:.2f} - Neutral", icon="⚖️")
            else:
                st.info("RSI: N/A")
        
        with col2:
            st.write("**MACD**")
            if macd is not None and signal is not None:
                # Determine MACD status with native color coding
                if macd > signal:
                    st.success(f"{macd:.2f} - Bullish", icon="🔼")
                else:
                    st.error(f"{macd:.2f} - Bearish", icon="🔽")
            else:
                st.info("MACD: N/A")
        
        # Add a divider
        st.markdown("---")
        
        # Moving Averages section
        st.subheader("Moving Averages")
        
        # Display moving averages
        if current_price and sma20:
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMA 20")
            with col2:
                if current_price > sma20:
                    st.success(f"${sma20:.2f} (Above)", icon="🔼")
                else:
                    st.error(f"${sma20:.2f} (Below)", icon="🔽")
        
        if current_price and sma50:
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMA 50")
            with col2:
                if current_price > sma50:
                    st.success(f"${sma50:.2f} (Above)", icon="🔼")
                else:
                    st.error(f"${sma50:.2f} (Below)", icon="🔽")
        
        if current_price and sma200:
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMA 200")
            with col2:
                if current_price > sma200:
                    st.success(f"${sma200:.2f} (Above)", icon="🔼")
                else:
                    st.error(f"${sma200:.2f} (Below)", icon="🔽")
    
    except Exception as e:
        st.error(f"Error displaying technical summary: {str(e)}")
        st.metric("Current Price", "Error")
        
        col1, col2 = st.columns(2)
        with col1:
            st.error("RSI: Error")
        with col2:
            st.error("MACD: Error")
