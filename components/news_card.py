import streamlit as st
import datetime

def create_news_card(title, subtitle, content, source, sentiment, date):
    """
    Create a news card using native Streamlit components
    """
    with st.container():
        st.markdown(f"#### {title}")
        if subtitle:
            st.markdown(f"<span style='color: #94a3b8; font-size: 0.9rem;'>{subtitle}</span>", unsafe_allow_html=True)
        
        st.write(content)
        
        # Source and sentiment badges
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Source badge
            st.markdown(f"<span style='background-color: #3b82f6; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;'>Source: {source}</span>", unsafe_allow_html=True)
        
        with col2:
            # Sentiment badge with appropriate color
            if sentiment == "Positive":
                badge_color = "#10b981"  # green
            elif sentiment == "Negative":
                badge_color = "#ef4444"  # red
            else:
                badge_color = "#94a3b8"  # neutral/gray
                
            st.markdown(f"<span style='background-color: {badge_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;'>Sentiment: {sentiment}</span>", unsafe_allow_html=True)
            
        # Date
        st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>{date}</span>", unsafe_allow_html=True)
        
        # Separator
        st.markdown("---")