import streamlit as st

def display_signals(signals: dict):
    """Display trading signals and explanations."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Buy Signals")
        if signals['buy_signals']:
            for signal in signals['buy_signals']:
                st.markdown(f"🟢 {signal}")
        else:
            st.write("No buy signals detected")
    
    with col2:
        st.subheader("Sell Signals")
        if signals['sell_signals']:
            for signal in signals['sell_signals']:
                st.markdown(f"🔴 {signal}")
        else:
            st.write("No sell signals detected")

def display_technical_summary(df):
    """Display technical analysis summary."""
    st.subheader("Technical Analysis Summary")
    
    # Latest values
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['MACD_Signal'].iloc[-1]
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with cols[1]:
        rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "⚪"
        st.metric("RSI", f"{rsi:.2f} {rsi_color}")
    
    with cols[2]:
        macd_signal = "🟢" if macd > signal else "🔴"
        st.metric("MACD", f"{macd:.2f} {macd_signal}")
