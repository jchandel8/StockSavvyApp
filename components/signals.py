import streamlit as st

def get_signal_html(signal_type: str) -> str:
    colors = {
        'buy': '#00ff00',
        'sell': '#ff0000',
        'neutral': '#808080'
    }
    return f'''
        <div style="
            display: inline-block;
            width: 12px;
            height: 12px;
            background: {colors[signal_type]};
            border-radius: 50%;
            box-shadow: 0 0 8px {colors[signal_type]};
            margin-right: 8px;
            vertical-align: middle;
        "></div>
    '''

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
        if signals['buy_signals']:
            for signal in signals['buy_signals']:
                st.markdown(get_signal_html('buy') + signal, unsafe_allow_html=True)
        else:
            st.write("No buy signals detected")
    
    with col2:
        st.subheader("Sell Signals")
        if signals['sell_signals']:
            for signal in signals['sell_signals']:
                st.markdown(get_signal_html('sell') + signal, unsafe_allow_html=True)
        else:
            st.write("No sell signals detected")

def display_technical_summary(df):
    st.subheader("Technical Analysis Summary")
    
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['MACD_Signal'].iloc[-1]
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with cols[1]:
        rsi_type = 'buy' if rsi < 30 else 'sell' if rsi > 70 else 'neutral'
        st.markdown(get_signal_html(rsi_type) + f"RSI: {rsi:.2f}", unsafe_allow_html=True)
    
    with cols[2]:
        macd_type = 'buy' if macd > signal else 'sell'
        st.markdown(get_signal_html(macd_type) + f"MACD: {macd:.2f}", unsafe_allow_html=True)
