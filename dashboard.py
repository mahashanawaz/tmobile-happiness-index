import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="T-Mobile Customer Happiness Index",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with T-Mobile brand colors
st.markdown("""
    <style>
    /* T-Mobile Brand Colors: Magenta (#E20074), Black (#000000), White (#FFFFFF), Gray (#5E5E5E) */
    
    .main-header {
        font-size: 5.5rem;
        font-weight: 900;
        color: #E20074;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #5E5E5E;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #E20074 0%, #FF6B9D 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-negative {
        background-color: #ffebee;
        border-left: 4px solid #E20074;
    }
    .alert-positive {
        background-color: #f0f0f0;
        border-left: 4px solid #000000;
    }
    
    /* Override Streamlit default colors */
    .stProgress > div > div > div > div {
        background-color: #E20074 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #E20074 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #C10062 !important;
        border: none !important;
    }
    
    /* Sidebar styling - FIXED */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        color: white !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #E20074 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #5E5E5E !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_data
def load_data():
    df = pd.read_csv('tmobile_reddit_cleaned.csv')
    # Add timestamps for demo (simulating real-time data)
    df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
    return df

# Prediction function
def predict_sentiment(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    prob_dict = {
        label: prob for label, prob in zip(model.classes_, probabilities)
    }
    
    return prediction, prob_dict

# Extract keywords from text
def extract_keywords(texts, sentiment_filter=None):
    if sentiment_filter is not None and len(sentiment_filter) > 0:
        texts = texts[texts.index.isin(sentiment_filter)]
    
    all_words = ' '.join(texts.values).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_words)
    
    # Remove common words
    stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'they', 'will', 'your', 'their'}
    words = [w for w in words if w not in stop_words]
    
    return Counter(words).most_common(10)

# Main app
def main():
    # Header without logo - just title and subtitle
    st.markdown('<p class="main-header">Customer Happiness Index</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time sentiment analysis and customer feedback monitoring</p>', unsafe_allow_html=True)
    
    # Load resources
    try:
        model, vectorizer = load_model()
        df = load_data()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/T-Mobile_New_Logo.svg/320px-T-Mobile_New_Logo.svg.png", width=200)
        st.markdown("---")
        
        page = st.radio("Navigation", [
            "🏠 Dashboard Overview",
            "📊 Historical Analytics",
            "🔮 Live Sentiment Detector",
            "🚨 Issue Detection",
            "⭐ Moments of Delight"
        ])
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This system analyzes customer feedback in real-time to detect issues early and highlight positive experiences.")
    
    # Page routing
    if page == "🏠 Dashboard Overview":
        dashboard_overview(df, model, vectorizer)
    elif page == "📊 Historical Analytics":
        historical_analytics(df)
    elif page == "🔮 Live Sentiment Detector":
        live_sentiment_detector(model, vectorizer, df)
    elif page == "🚨 Issue Detection":
        issue_detection(df)
    elif page == "⭐ Moments of Delight":
        moments_of_delight(df)

# Dashboard Overview
def dashboard_overview(df, model, vectorizer):
    st.header("📊 Real-Time Customer Happiness Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_feedback = len(df)
    positive_pct = (df['SentimentLabel'] == 'Positive').sum() / total_feedback * 100
    neutral_pct = (df['SentimentLabel'] == 'Neutral').sum() / total_feedback * 100
    negative_pct = (df['SentimentLabel'] == 'Negative').sum() / total_feedback * 100
    
    # Calculate happiness index (weighted score)
    happiness_index = (positive_pct * 1.0 + neutral_pct * 0.5 + negative_pct * 0.0)
    
    with col1:
        st.metric("😊 Happiness Index", f"{happiness_index:.1f}%", 
                  delta=f"+2.3% vs last week", delta_color="normal")
    
    with col2:
        st.metric("💬 Total Feedback", total_feedback,
                  delta=f"+{int(total_feedback * 0.12)} this week")
    
    with col3:
        st.metric("😃 Positive Rate", f"{positive_pct:.1f}%",
                  delta="+1.5%", delta_color="normal")
    
    with col4:
        st.metric("😟 Issues Detected", int(negative_pct * total_feedback / 100),
                  delta="-2", delta_color="inverse")
    
    st.markdown("---")
    
    # Sentiment Distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['SentimentLabel'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=.4,
            marker=dict(colors=['#E20074', '#5E5E5E', '#000000'])  # T-Mobile colors: Positive(Magenta), Neutral(Gray), Negative(Black)
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Trend (Last 24 Hours)")
        
        # Simulate hourly trend
        df_recent = df.tail(24).copy()
        df_recent['hour'] = pd.to_datetime(df_recent['timestamp']).dt.hour
        
        trend_data = df_recent.groupby(['hour', 'SentimentLabel']).size().reset_index(name='count')
        
        fig = px.line(trend_data, x='hour', y='count', color='SentimentLabel',
                     color_discrete_map={'Positive': '#E20074', 'Neutral': '#5E5E5E', 'Negative': '#000000'})
        fig.update_layout(height=400, xaxis_title="Hour", yaxis_title="Feedback Count")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts & Highlights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Issues Requiring Attention")
        negative_feedback = df[df['SentimentLabel'] == 'Negative'].tail(3)
        
        if len(negative_feedback) > 0:
            for idx, row in negative_feedback.iterrows():
                st.markdown(f"""
                <div class="alert-box alert-negative">
                    <strong>🔴 Negative Feedback</strong><br>
                    {row['text'][:150]}...<br>
                    <small>{row['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical issues detected! 🎉")
    
    with col2:
        st.markdown("#### ⭐ Moments of Delight")
        positive_feedback = df[df['SentimentLabel'] == 'Positive'].tail(3)
        
        for idx, row in positive_feedback.iterrows():
            st.markdown(f"""
            <div class="alert-box alert-positive">
                <strong>🟢 Positive Feedback</strong><br>
                {row['text'][:150]}...<br>
                <small>{row['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)

# Historical Analytics
def historical_analytics(df):
    st.header("📊 Historical Data Analytics")
    
    st.markdown(f"""
    ### Dataset Overview
    - **Total Feedback Analyzed**: {len(df)} posts
    - **Data Source**: T-Mobile Reddit Community
    - **Time Range**: Last 30 days (simulated)
    """)
    
    # Sentiment breakdown
    col1, col2, col3 = st.columns(3)
    
    sentiment_counts = df['SentimentLabel'].value_counts()
    
    with col1:
        st.metric("😃 Positive", sentiment_counts.get('Positive', 0),
                 f"{sentiment_counts.get('Positive', 0)/len(df)*100:.1f}%")
    
    with col2:
        st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0),
                 f"{sentiment_counts.get('Neutral', 0)/len(df)*100:.1f}%")
    
    with col3:
        st.metric("😟 Negative", sentiment_counts.get('Negative', 0),
                 f"{sentiment_counts.get('Negative', 0)/len(df)*100:.1f}%")
    
    st.markdown("---")
    
    # Keyword Analysis
    st.subheader("🔍 Top Keywords by Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Positive Feedback**")
        positive_keywords = extract_keywords(
            df['FeedbackText'], 
            df[df['SentimentLabel'] == 'Positive'].index
        )
        for word, count in positive_keywords[:5]:
            st.write(f"✅ {word}: {count}")
    
    with col2:
        st.markdown("**Neutral Feedback**")
        neutral_keywords = extract_keywords(
            df['FeedbackText'], 
            df[df['SentimentLabel'] == 'Neutral'].index
        )
        for word, count in neutral_keywords[:5]:
            st.write(f"➖ {word}: {count}")
    
    with col3:
        st.markdown("**Negative Feedback**")
        negative_keywords = extract_keywords(
            df['FeedbackText'], 
            df[df['SentimentLabel'] == 'Negative'].index
        )
        for word, count in negative_keywords[:5]:
            st.write(f"❌ {word}: {count}")
    
    st.markdown("---")
    
    # Sample feedback
    st.subheader("📝 Sample Feedback by Sentiment")
    
    sentiment_filter = st.selectbox("Filter by sentiment:", ['All', 'Positive', 'Neutral', 'Negative'])
    
    if sentiment_filter == 'All':
        display_df = df
    else:
        display_df = df[df['SentimentLabel'] == sentiment_filter]
    
    st.dataframe(
        display_df[['title', 'text', 'SentimentLabel', 'timestamp']].tail(10),
        use_container_width=True
    )

# Live Sentiment Detector
def live_sentiment_detector(model, vectorizer, df):
    st.header("🔮 Live Sentiment Detector")
    
    st.markdown("""
    ### Analyze New Customer Feedback
    Enter customer feedback below to get instant sentiment analysis. This simulates real-time 
    processing that would integrate with live data streams (Kafka, Reddit API, etc.)
    """)
    
    # Pre-loaded examples
    st.subheader("📌 Quick Test Examples")
    
    example_texts = {
        "Positive Example": "T-Mobile's 5G network is incredibly fast! Customer service resolved my issue in minutes. Very impressed!",
        "Neutral Example": "Switched to T-Mobile last month. Service is okay, nothing special. Price is reasonable.",
        "Negative Example": "Terrible experience. Billing errors for 3 months straight. Customer support unhelpful. Very frustrated.",
        "Network Issue": "No signal in my area for the past week. Can't make calls or use data. This is unacceptable.",
        "Praise Example": "Best carrier I've ever had! Great coverage, amazing deals, and friendly staff. Highly recommend!"
    }
    
    selected_example = st.selectbox("Try a sample:", ["Custom Input"] + list(example_texts.keys()))
    
    if selected_example == "Custom Input":
        user_input = st.text_area("Enter customer feedback:", height=150, 
                                   placeholder="Type or paste customer feedback here...")
    else:
        user_input = st.text_area("Enter customer feedback:", value=example_texts[selected_example], height=150)
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing feedback..."):
                prediction, probabilities = predict_sentiment(user_input, model, vectorizer)
                
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                sentiment_emoji = {
                    'Positive': '😃',
                    'Neutral': '😐',
                    'Negative': '😟'
                }
                
                sentiment_color = {
                    'Positive': '#E20074',
                    'Neutral': '#5E5E5E',
                    'Negative': '#000000'
                }
                
                with col1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: {sentiment_color[prediction]}; 
                                border-radius: 10px; color: white;'>
                        <h1>{sentiment_emoji[prediction]}</h1>
                        <h3>{prediction}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### Confidence Scores")
                    for sentiment, prob in probabilities.items():
                        st.progress(prob, text=f"{sentiment}: {prob*100:.1f}%")
                
                with col3:
                    st.markdown("#### Recommended Action")
                    if prediction == 'Negative':
                        st.error("🚨 **Alert Team** - Immediate attention required")
                        st.write("• Escalate to customer support")
                        st.write("• Flag for quality review")
                    elif prediction == 'Neutral':
                        st.warning("⚠️ **Monitor** - Track for patterns")
                        st.write("• Add to watchlist")
                        st.write("• Check for follow-ups")
                    else:
                        st.success("✅ **Celebrate** - Share positive feedback")
                        st.write("• Share with team")
                        st.write("• Use in testimonials")
                
                # Additional insights
                st.markdown("---")
                st.subheader("📊 Context & Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Feedback Analysis**")
                    st.write(f"• Word count: {len(user_input.split())} words")
                    st.write(f"• Character count: {len(user_input)} characters")
                    st.write(f"• Predicted: {prediction}")
                    st.write(f"• Confidence: {max(probabilities.values())*100:.1f}%")
                
                with col2:
                    st.markdown("**Historical Comparison**")
                    similar_sentiment = df[df['SentimentLabel'] == prediction]
                    st.write(f"• Similar feedback in dataset: {len(similar_sentiment)}")
                    st.write(f"• Percentage of total: {len(similar_sentiment)/len(df)*100:.1f}%")
                    
        else:
            st.warning("⚠️ Please enter some feedback to analyze.")

# Issue Detection
def issue_detection(df):
    st.header("🚨 Issue Detection & Early Warning System")
    
    st.markdown("""
    ### Detect Issues Before They Spread
    Identify patterns in negative feedback to catch problems early and prevent escalation.
    """)
    
    # Issue metrics
    negative_df = df[df['SentimentLabel'] == 'Negative']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Active Issues", len(negative_df),
                 delta="-2 from yesterday", delta_color="inverse")
    
    with col2:
        # Simulate issue velocity (how fast issues are being reported)
        issue_velocity = len(negative_df) / (len(df) / 24)  # per hour
        st.metric("📈 Issue Velocity", f"{issue_velocity:.1f}/hr",
                 delta="+0.3", delta_color="inverse")
    
    with col3:
        # Average time to detect (simulated)
        st.metric("⏱️ Avg Detection Time", "1.2 hrs",
                 delta="-0.5 hrs", delta_color="normal")
    
    st.markdown("---")
    
    # Issue categories
    st.subheader("📋 Issue Categories")
    
    # Extract common themes from negative feedback
    if len(negative_df) > 0:
        keywords = extract_keywords(df['FeedbackText'], negative_df.index)
        
        # Categorize issues (simplified)
        categories = {
            'Network/Coverage': 0,
            'Billing': 0,
            'Customer Service': 0,
            'Device/Hardware': 0,
            'Other': 0
        }
        
        for text in negative_df['text']:
            text_lower = text.lower()
            if any(word in text_lower for word in ['network', 'signal', 'coverage', 'connection', 'speed']):
                categories['Network/Coverage'] += 1
            elif any(word in text_lower for word in ['bill', 'charge', 'payment', 'price', 'cost']):
                categories['Billing'] += 1
            elif any(word in text_lower for word in ['support', 'service', 'help', 'representative', 'agent']):
                categories['Customer Service'] += 1
            elif any(word in text_lower for word in ['phone', 'device', 'hardware', 'screen', 'battery']):
                categories['Device/Hardware'] += 1
            else:
                categories['Other'] += 1
        
        fig = go.Figure(data=[
            go.Bar(x=list(categories.keys()), y=list(categories.values()),
                   marker_color=['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5'])
        ])
        fig.update_layout(title="Issues by Category", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent issues
    st.subheader("🔍 Recent Issues Requiring Attention")
    
    if len(negative_df) > 0:
        for idx, row in negative_df.tail(5).iterrows():
            with st.expander(f"🔴 Issue from {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(f"**Title:** {row['title']}")
                st.markdown(f"**Feedback:** {row['text']}")
                st.markdown("**Recommended Actions:**")
                st.write("• Escalate to technical support team")
                st.write("• Monitor for similar patterns")
                st.write("• Follow up with customer within 24 hours")
    else:
        st.success("🎉 No critical issues detected! Keep up the great work!")

# Moments of Delight
def moments_of_delight(df):
    st.header("⭐ Moments of Delight")
    
    st.markdown("""
    ### Celebrate Customer Success Stories
    Highlight exceptional experiences and positive feedback to motivate teams and showcase wins.
    """)
    
    positive_df = df[df['SentimentLabel'] == 'Positive']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌟 Positive Mentions", len(positive_df),
                 delta="+12 this week", delta_color="normal")
    
    with col2:
        delight_rate = len(positive_df) / len(df) * 100
        st.metric("💚 Delight Rate", f"{delight_rate:.1f}%",
                 delta="+2.5%", delta_color="normal")
    
    with col3:
        st.metric("🏆 5-Star Experiences", int(len(positive_df) * 0.6),
                 delta="+8", delta_color="normal")
    
    st.markdown("---")
    
    # Top positive keywords
    st.subheader("✨ What Customers Love")
    
    positive_keywords = extract_keywords(df['FeedbackText'], positive_df.index)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Top Praise Keywords")
        for i, (word, count) in enumerate(positive_keywords[:10], 1):
            st.write(f"{i}. **{word.title()}** - mentioned {count} times")
    
    with col2:
        # Visualize as bar chart
        words, counts = zip(*positive_keywords[:10])
        fig = go.Figure(data=[
            go.Bar(x=list(counts), y=list(words), orientation='h',
                   marker_color='#4CAF50')
        ])
        fig.update_layout(height=400, xaxis_title="Mentions", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent positive feedback
    st.subheader("💬 Recent Positive Feedback")
    
    for idx, row in positive_df.tail(5).iterrows():
        st.success(f"""
        **{row['title']}**  
        {row['text'][:200]}...  
        *{row['timestamp'].strftime('%Y-%m-%d %H:%M')}*
        """)
    
    st.markdown("---")
    
    # Share worthy moments
    st.subheader("📣 Share-Worthy Moments")
    st.info("""
    **Action Items:**
    - Share top feedback with team in weekly newsletter
    - Feature customer testimonials on social media
    - Recognize team members mentioned in positive feedback
    - Use insights to replicate success across all channels
    """)

if __name__ == "__main__":
    main()