import streamlit as st

def create_prediction_card(title, direction, confidence, current_price, pred_price, pred_high, pred_low):
    """
    Create a styled prediction card with proper formatting
    """
    # Calculate percentage change
    price_change = ((pred_price - current_price) / current_price) * 100
    price_change_sign = "+" if price_change >= 0 else ""
    
    # Set colors based on direction
    if direction == "UP":
        direction_color = "#10b981"  # green
        badge_color = "background-color: #10b981;"
    elif direction == "DOWN":
        direction_color = "#ef4444"  # red
        badge_color = "background-color: #ef4444;"
    else:
        direction_color = "#94a3b8"  # neutral/gray
        badge_color = "background-color: #94a3b8;"
    
    price_change_color = "#10b981" if price_change >= 0 else "#ef4444"
    
    # Create the card HTML (avoiding nested string formatting issues)
    card_html = f"""
    <div style="background-color: #1a2234; border-radius: 0.5rem; padding: 1rem; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; font-size: 1rem; font-weight: 600; color: #f5f9fc;">{title}</h4>
            <span style="{badge_color} color: white; font-size: 0.75rem; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">{direction}</span>
        </div>
        
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #94a3b8; font-size: 0.875rem;">Confidence</span>
                <span style="color: #f5f9fc; font-size: 0.875rem;">{confidence:.1f}%</span>
            </div>
            <div style="background-color: #2a3447; height: 0.5rem; border-radius: 0.25rem; overflow: hidden;">
                <div style="background-color: {direction_color}; height: 100%; width: {confidence}%;"></div>
            </div>
        </div>
        
        <div style="margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #94a3b8; font-size: 0.875rem;">Target Price</span>
                <span style="font-weight: 600; color: #f5f9fc;">${pred_price:.2f} <span style="color: {price_change_color}; font-size: 0.75rem;">({price_change_sign}{price_change:.2f}%)</span></span>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div>
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted High</div>
                    <div style="font-weight: 600; color: #f5f9fc;">${pred_high:.2f}</div>
                </div>
                <div>
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Predicted Low</div>
                    <div style="font-weight: 600; color: #f5f9fc;">${pred_low:.2f}</div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; position: relative; height: 0.25rem; background-color: #2a3447; border-radius: 0.125rem;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0.75rem; height: 0.75rem; background-color: {direction_color}; border-radius: 50%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                <span>${pred_low:.2f}</span>
                <span>${pred_high:.2f}</span>
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)