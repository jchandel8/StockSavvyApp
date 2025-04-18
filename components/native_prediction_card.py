import streamlit as st

def create_native_prediction_card(title, direction, confidence, current_price, pred_price, pred_high, pred_low):
    """
    Create a styled prediction card using native Streamlit components
    """
    with st.container():
        # Use native components for everything
        st.subheader(title)
        
        # Direction badge
        if direction == "UP":
            st.markdown(f"<span style='background-color: #10b981; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;'>{direction}</span>", unsafe_allow_html=True)
        elif direction == "DOWN":
            st.markdown(f"<span style='background-color: #ef4444; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;'>{direction}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='background-color: #94a3b8; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;'>{direction}</span>", unsafe_allow_html=True)
        
        # Confidence meter
        st.write("Confidence:")
        st.progress(confidence/100.0)
        st.write(f"{confidence:.1f}%")
        
        # Price change
        price_change = ((pred_price - current_price) / current_price) * 100
        price_change_sign = "+" if price_change >= 0 else ""
        
        # Target price with color coding
        if price_change >= 0:
            st.markdown(f"**Target Price:** ${pred_price:.2f} <span style='color: #10b981; font-size: 0.8rem;'>({price_change_sign}{price_change:.2f}%)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Target Price:** ${pred_price:.2f} <span style='color: #ef4444; font-size: 0.8rem;'>({price_change_sign}{price_change:.2f}%)</span>", unsafe_allow_html=True)
        
        # Predicted high and low in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("Predicted High")
            st.write(f"${pred_high:.2f}")
        
        with col2:
            st.write("Predicted Low")
            st.write(f"${pred_low:.2f}")
        
        # Add a separator at the end
        st.markdown("---")