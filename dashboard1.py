import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Airline Reviews Analysis", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv("processed_reviews.csv")
    return df

# Function to analyze pain points
def analyze_pain_points(df):
    negative_reviews = df[df['sentiment_category'] == 'Negative']['Cleaned_Reviews']
    negative_text = ' '.join(negative_reviews)
    
    categories = {
        "Delays": ["delay", "delayed", "waiting", "time"],
        "Customer Service": ["staff", "service", "rude", "attitude"],
        "Comfort": ["seat", "legroom", "space", "cramped"],
        "Food": ["food", "meal", "tasteless", "quality"],
        "Baggage": ["baggage", "luggage", "lost", "damaged"]
    }
    
    pain_points = {category: sum(negative_text.count(word) for word in words)
                  for category, words in categories.items()}
    
    return pain_points

# Main title with custom CSS for better spacing
st.title("✈️ Airline Reviews Analysis Dashboard")
st.markdown("---")

try:
    # Load and filter data
    df = load_data()
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        airlines = ['All'] + sorted(df['Airline'].unique().tolist())
        selected_airline = st.selectbox('Select Airline', airlines)
    
    filtered_df = df[df['Airline'] == selected_airline] if selected_airline != 'All' else df.copy()

    # Row 1: Key Metrics with improved spacing
    st.markdown("<div class='chart-spacing'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(filtered_df))
    with col2:
        st.metric("Average Rating", f"{filtered_df['Overall Rating'].mean():.2f}⭐")
    with col3:
        positive_pct = (filtered_df['sentiment_category'] == 'Positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    with col4:
        recommend_pct = (filtered_df['Recommended'].fillna('No') == 'Yes').mean() * 100
        st.metric("Would Recommend", f"{recommend_pct:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Sentiment and Ratings (2 columns)
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_counts = filtered_df['sentiment_category'].value_counts()
        fig = px.pie(values=sentiment_counts.values, 
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = px.histogram(filtered_df, x="Overall Rating",
                         nbins=10,
                         title="Rating Distribution",
                         color_discrete_sequence=['#2ecc71'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Pain Points Analysis (2 columns)
    st.markdown("<div class='chart-spacing'>", unsafe_allow_html=True)
    st.subheader("Pain Points Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        pain_points = analyze_pain_points(filtered_df)
        fig = px.bar(x=list(pain_points.keys()), 
                    y=list(pain_points.values()),
                    title="Frequency of Pain Points in Negative Reviews",
                    color_discrete_sequence=['#e74c3c'])
        fig.update_layout(height=400,
                         xaxis_title="Category",
                         yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        pain_points_df = pd.DataFrame(list(pain_points.items()), 
                                    columns=['Category', 'Frequency'])
        total_mentions = pain_points_df['Frequency'].sum()
        pain_points_df['Percentage'] = (pain_points_df['Frequency'] / total_mentions * 100).round(1)
        pain_points_df['Percentage'] = pain_points_df['Percentage'].apply(lambda x: f"{x}%")
        
        st.markdown("### Pain Points Breakdown")
        st.dataframe(pain_points_df.sort_values('Frequency', ascending=False),
                    hide_index=True,
                    height=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # Row 4: Service Aspects Analysis
    aspects = ['Seat Comfort', 'Staff Service', 'Food & Beverages', 
               'Inflight Entertainment', 'Value For Money']
    
    fig = go.Figure()
    for aspect in aspects:
        fig.add_trace(go.Box(y=filtered_df[aspect], name=aspect))
    
    fig.update_layout(title="Service Aspects Ratings Distribution",
                     height=500,
                     yaxis_title="Rating",
                     showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Row 5: Word Cloud and Traveller Analysis (2 columns)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Review Word Cloud")
        text = ' '.join(filtered_df['Cleaned_Reviews'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    with col2:
        avg_by_traveller = filtered_df.groupby('Type of Traveller')['Overall Rating'].mean()
        fig = px.bar(x=avg_by_traveller.index, 
                    y=avg_by_traveller.values,
                    title="Average Rating by Traveller Type",
                    color_discrete_sequence=['#3498db'])
        fig.update_layout(height=400,
                         xaxis_title="Traveller Type",
                         yaxis_title="Average Rating")
        st.plotly_chart(fig, use_container_width=True)

    # Tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["📊 Correlation Analysis", "✈️ Airline Comparison", "📝 Review Samples"])
    
    with tab1:
        corr_matrix = filtered_df[aspects + ['Overall Rating']].corr()
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       color_continuous_scale="RdBu")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if selected_airline == 'All':
            avg_ratings = df.groupby('Airline')['Overall Rating'].mean()
            fig = px.bar(x=avg_ratings.index,
                        y=avg_ratings.values,
                        title="Average Rating by Airline",
                        color_discrete_sequence=['#e74c3c'])
            fig.update_layout(height=500,
                            xaxis_title="Airline",
                            yaxis_title="Average Rating")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        st.write("Sample Reviews:")
        sample_reviews = filtered_df[['Reviews', 'sentiment_category', 'Overall Rating']].sample(5)
        st.dataframe(sample_reviews, height=300)

    # Footer
    st.markdown("---")
    st.markdown("### 📈 Dashboard Insights")
    st.write("""
    - This dashboard provides a comprehensive analysis of airline reviews and customer sentiment
    - Use the sidebar filters to explore specific airlines
    - The analysis includes sentiment distribution, rating patterns, and service aspect evaluations
    - Pain points analysis helps identify key areas of customer dissatisfaction
    - Word cloud visualization helps identify common themes in reviews
    """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.write("Please ensure the 'processed_reviews.csv' file is in the same directory as this script.")