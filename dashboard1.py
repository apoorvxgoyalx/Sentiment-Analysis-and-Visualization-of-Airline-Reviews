import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Airline Reviews Analysis", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv("processed_reviews.csv")
    return df

# Main title
st.title("✈️ Airline Reviews Analysis Dashboard")
st.markdown("---")

# Load the data
try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Airline selection
    airlines = ['All'] + sorted(df['Airline'].unique().tolist())
    selected_airline = st.sidebar.selectbox('Select Airline', airlines)
    
    # Filter data based on selection
    if selected_airline != 'All':
        filtered_df = df[df['Airline'] == selected_airline]
    else:
        filtered_df = df.copy()

    # Main content area
    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(filtered_df))
    with col2:
        avg_rating = filtered_df['Overall Rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}⭐")
    with col3:
        positive_pct = (filtered_df['sentiment_category'] == 'Positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    with col4:
        recommend_pct = (filtered_df['Recommended'].fillna('No') == 'Yes').mean() * 100
        st.metric("Would Recommend", f"{recommend_pct:.1f}%")

    # Row 2: Sentiment Distribution and Ratings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment_category'].value_counts()
        fig = px.pie(values=sentiment_counts.values, 
                    names=sentiment_counts.index, 
                    title="Review Sentiment Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Rating Distribution")
        fig = px.histogram(filtered_df, x="Overall Rating", 
                         nbins=10, title="Distribution of Overall Ratings",
                         color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Service Aspects Analysis
    st.subheader("Service Aspects Analysis")
    
    aspects = ['Seat Comfort', 'Staff Service', 'Food & Beverages', 
               'Inflight Entertainment', 'Value For Money']
    
    fig = go.Figure()
    for aspect in aspects:
        fig.add_trace(go.Box(y=filtered_df[aspect], name=aspect))
    
    fig.update_layout(title="Ratings Distribution Across Service Aspects",
                     yaxis_title="Rating",
                     showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Row 4: Word Cloud and Common Topics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Review Word Cloud")
        # Generate word cloud
        text = ' '.join(filtered_df['Cleaned_Reviews'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.subheader("Average Ratings by Traveller Type")
        avg_by_traveller = filtered_df.groupby('Type of Traveller')['Overall Rating'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_by_traveller.index, y=avg_by_traveller.values,
                    title="Average Rating by Traveller Type",
                    labels={'x': 'Traveller Type', 'y': 'Average Rating'},
                    color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)

    # Row 5: Detailed Analysis
    st.markdown("---")
    st.subheader("Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Ratings Correlation", "✈️ Airline Comparison", "📝 Review Analysis"])
    
    with tab1:
        # Correlation heatmap
        corr_matrix = filtered_df[aspects + ['Overall Rating']].corr()
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if selected_airline == 'All':
            # Airline comparison
            avg_ratings = df.groupby('Airline')[aspects + ['Overall Rating']].mean()
            fig = px.bar(avg_ratings.reset_index(), x='Airline', y='Overall Rating',
                        title="Average Rating by Airline",
                        color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        # Show sample reviews
        st.write("Sample Reviews:")
        sample_reviews = filtered_df[['Reviews', 'sentiment_category', 'Overall Rating']].sample(5)
        st.dataframe(sample_reviews)

    # Footer
    st.markdown("---")
    st.markdown("### 📈 Dashboard Insights")
    st.write("""
    - This dashboard provides a comprehensive analysis of airline reviews and customer sentiment
    - Use the sidebar filters to explore specific airlines
    - The analysis includes sentiment distribution, rating patterns, and service aspect evaluations
    - Word cloud visualization helps identify common themes in reviews
    """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.write("Please ensure the 'processed_reviews.csv' file is in the same directory as this script.")