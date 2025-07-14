import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Amazon Commerce Reviews - EDA Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #ff9900;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #232f3e;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff9900;
        }
        .insight-box {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff9900;
            margin: 1rem 0;
        }
        .sentiment-positive {
            color: #28a745;
            font-weight: bold;
        }
        .sentiment-negative {
            color: #dc3545;
            font-weight: bold;
        }
        .sentiment-neutral {
            color: #6c757d;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try different possible URLs for the Amazon dataset
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/amazon_product_reviews.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/amazon_reviews.csv"
        ]
        
        for url in urls:
            try:
                df = pd.read_csv(url)
                return df
            except:
                continue
                
        # If URLs fail, create sample data structure for Amazon reviews
        st.warning("⚠️ Could not load data from UCI repository. Please upload your own Amazon reviews dataset.")
        return None
    except:
        return None

@st.cache_data
def create_sample_data():
    """Create sample Amazon reviews data for demonstration"""
    np.random.seed(42)
    
    sample_data = {
        'product_id': ['B001', 'B002', 'B003', 'B004', 'B005'] * 200,
        'product_title': ['iPhone 13', 'Samsung Galaxy', 'MacBook Pro', 'Dell Laptop', 'iPad Air'] * 200,
        'review_title': ['Great phone!', 'Love it!', 'Amazing quality', 'Good value', 'Excellent'] * 200,
        'review_text': [
            'This phone is absolutely amazing. Great camera quality and battery life.',
            'Love the design and performance. Highly recommended!',
            'Excellent build quality and fast performance. Worth the price.',
            'Good laptop for the price. Fast delivery and good packaging.',
            'Perfect tablet for work and entertainment. Very satisfied.'
        ] * 200,
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.15, 0.35, 0.35]),
        'helpful_votes': np.random.randint(0, 50, 1000),
        'total_votes': np.random.randint(0, 100, 1000),
        'verified_purchase': np.random.choice(['Y', 'N'], 1000, p=[0.8, 0.2]),
        'review_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000]
    }
    
    return pd.DataFrame(sample_data)

@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()
    
    # Basic text preprocessing
    if 'review_text' in df_processed.columns:
        df_processed['review_length'] = df_processed['review_text'].str.len()
        df_processed['word_count'] = df_processed['review_text'].str.split().str.len()
    
    # Create helpful ratio
    if 'helpful_votes' in df_processed.columns and 'total_votes' in df_processed.columns:
        df_processed['helpful_ratio'] = df_processed['helpful_votes'] / (df_processed['total_votes'] + 1)
    
    # Convert date columns
    date_cols = [col for col in df_processed.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            df_processed[col] = pd.to_datetime(df_processed[col])
        except:
            pass
    
    return df_processed

def simple_sentiment_analysis(text):
    """Simple sentiment analysis based on keywords"""
    if pd.isna(text):
        return 'neutral'
    
    text = text.lower()
    
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor', 'useless']
    
    pos_score = sum(1 for word in positive_words if word in text)
    neg_score = sum(1 for word in negative_words if word in text)
    
    if pos_score > neg_score:
        return 'positive'
    elif neg_score > pos_score:
        return 'negative'
    else:
        return 'neutral'

def extract_keywords(text_series, n_words=20):
    """Extract top keywords from text"""
    # Combine all text
    all_text = ' '.join(text_series.dropna().astype(str))
    
    # Simple keyword extraction (remove common words)
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an', 'this', 'that', 'it', 'i', 'you', 'he', 'she', 'we', 'they'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return Counter(words).most_common(n_words)

def main():
    st.markdown('<h1 class="main-header">🛒 Amazon Commerce Reviews - EDA Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.info("Using sample Amazon reviews data for demonstration.")
        uploaded_file = st.file_uploader("Upload your Amazon reviews CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = create_sample_data()
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    features = [
        "📊 Dataset Overview",
        "⭐ Rating Analysis", 
        "📝 Review Text Analysis",
        "💬 Sentiment Analysis",
        "🔤 Keyword & Topic Analysis",
        "📈 Product Performance",
        "⏰ Temporal Analysis",
        "👥 User Behavior Analysis",
        "🔍 Interactive Review Explorer",
        "📋 Review Quality Assessment"
    ]
    
    selected_feature = st.sidebar.selectbox("Select Analysis Feature:", features)
    
    # Feature 1: Dataset Overview
    if selected_feature == "📊 Dataset Overview":
        st.markdown('<h2 class="feature-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        with col2:
            st.metric("Unique Products", f"{df['product_id'].nunique() if 'product_id' in df.columns else 'N/A'}")
        with col3:
            st.metric("Average Rating", f"{df['rating'].mean():.1f}" if 'rating' in df.columns else "N/A")
        with col4:
            st.metric("Features", len(df.columns))
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sample Reviews:**")
            display_cols = ['product_title', 'rating', 'review_title', 'review_text'] if all(col in df.columns for col in ['product_title', 'rating', 'review_title', 'review_text']) else df.columns[:4]
            st.dataframe(df[display_cols].head(10))
            
        with col2:
            st.write("**Dataset Schema:**")
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(schema_df)
    
    # Feature 2: Rating Analysis
    elif selected_feature == "⭐ Rating Analysis":
        st.markdown('<h2 class="feature-header">⭐ Rating Analysis</h2>', unsafe_allow_html=True)
        
        if 'rating' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution
                rating_counts = df['rating'].value_counts().sort_index()
                fig = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    title="Rating Distribution",
                    labels={'x': 'Rating', 'y': 'Count'}
                )
                fig.update_traces(marker_color=['#ff4444' if x <= 2 else '#ffaa00' if x == 3 else '#44ff44' for x in rating_counts.index])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Rating statistics
                st.subheader("Rating Statistics")
                st.write(f"**Average Rating:** {df['rating'].mean():.2f}")
                st.write(f"**Median Rating:** {df['rating'].median():.1f}")
                st.write(f"**Standard Deviation:** {df['rating'].std():.2f}")
                st.write(f"**Most Common Rating:** {df['rating'].mode()[0]}")
                
                # Rating percentages
                st.write("**Rating Breakdown:**")
                for rating in sorted(df['rating'].unique()):
                    count = (df['rating'] == rating).sum()
                    percentage = (count / len(df)) * 100
                    st.write(f"⭐ {rating}: {count} ({percentage:.1f}%)")
            
            # Rating trends over time
            if 'review_date' in df.columns:
                st.subheader("Rating Trends Over Time")
                df['review_date'] = pd.to_datetime(df['review_date'])
                df['year_month'] = df['review_date'].dt.to_period('M')
                
                monthly_ratings = df.groupby('year_month')['rating'].mean().reset_index()
                monthly_ratings['year_month'] = monthly_ratings['year_month'].astype(str)
                
                fig = px.line(
                    monthly_ratings,
                    x='year_month',
                    y='rating',
                    title="Average Rating Trends Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Rating column not found in the dataset.")
    
    # Feature 3: Review Text Analysis
    elif selected_feature == "📝 Review Text Analysis":
        st.markdown('<h2 class="feature-header">📝 Review Text Analysis</h2>', unsafe_allow_html=True)
        
        if 'review_text' in df_processed.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Review length distribution
                fig = px.histogram(
                    df_processed,
                    x='review_length',
                    nbins=50,
                    title="Review Length Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Word count distribution
                fig = px.histogram(
                    df_processed,
                    x='word_count',
                    nbins=50,
                    title="Word Count Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Text statistics
            st.subheader("Text Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Review Length", f"{df_processed['review_length'].mean():.0f} chars")
            with col2:
                st.metric("Avg Word Count", f"{df_processed['word_count'].mean():.0f} words")
            with col3:
                st.metric("Shortest Review", f"{df_processed['review_length'].min()} chars")
            with col4:
                st.metric("Longest Review", f"{df_processed['review_length'].max()} chars")
            
            # Review length vs rating
            if 'rating' in df.columns:
                st.subheader("Review Length vs Rating")
                fig = px.box(
                    df_processed,
                    x='rating',
                    y='review_length',
                    title="Review Length by Rating"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Review text column not found in the dataset.")
    
    # Feature 4: Sentiment Analysis
    elif selected_feature == "💬 Sentiment Analysis":
        st.markdown('<h2 class="feature-header">💬 Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        if 'review_text' in df.columns:
            # Perform sentiment analysis
            with st.spinner("Analyzing sentiment..."):
                df['sentiment'] = df['review_text'].apply(simple_sentiment_analysis)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = df['sentiment'].value_counts()
                
                colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#6c757d'}
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map=colors
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment statistics
                st.subheader("Sentiment Statistics")
                total_reviews = len(df)
                for sentiment in ['positive', 'negative', 'neutral']:
                    count = (df['sentiment'] == sentiment).sum()
                    percentage = (count / total_reviews) * 100
                    
                    if sentiment == 'positive':
                        st.markdown(f'<span class="sentiment-positive">😊 Positive: {count} ({percentage:.1f}%)</span>', unsafe_allow_html=True)
                    elif sentiment == 'negative':
                        st.markdown(f'<span class="sentiment-negative">😞 Negative: {count} ({percentage:.1f}%)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="sentiment-neutral">😐 Neutral: {count} ({percentage:.1f}%)</span>', unsafe_allow_html=True)
            
            # Sentiment vs Rating
            if 'rating' in df.columns:
                st.subheader("Sentiment vs Rating Analysis")
                
                # Create cross-tabulation
                sentiment_rating = pd.crosstab(df['rating'], df['sentiment'])
                
                fig = px.bar(
                    sentiment_rating,
                    title="Sentiment Distribution by Rating",
                    color_discrete_map=colors
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sample reviews by sentiment
            st.subheader("Sample Reviews by Sentiment")
            selected_sentiment = st.selectbox("Select sentiment:", ['positive', 'negative', 'neutral'])
            
            sentiment_reviews = df[df['sentiment'] == selected_sentiment]
            if not sentiment_reviews.empty:
                sample_reviews = sentiment_reviews.sample(min(5, len(sentiment_reviews)))
                for idx, row in sample_reviews.iterrows():
                    st.write(f"**Rating:** {row['rating'] if 'rating' in row else 'N/A'}")
                    st.write(f"**Review:** {row['review_text'][:200]}...")
                    st.write("---")
        else:
            st.warning("Review text column not found in the dataset.")
    
    # Feature 5: Keyword & Topic Analysis
    elif selected_feature == "🔤 Keyword & Topic Analysis":
        st.markdown('<h2 class="feature-header">🔤 Keyword & Topic Analysis</h2>', unsafe_allow_html=True)
        
        if 'review_text' in df.columns:
            # Extract keywords
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(df['review_text'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top keywords
                st.subheader("Top Keywords")
                if keywords:
                    keywords_df = pd.DataFrame(keywords, columns=['Word', 'Frequency'])
                    
                    fig = px.bar(
                        keywords_df.head(15),
                        x='Frequency',
                        y='Word',
                        orientation='h',
                        title="Top 15 Keywords"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(keywords_df.head(20))
            
            with col2:
                # Word cloud simulation with bar chart
                st.subheader("Keyword Frequency Analysis")
                if keywords:
                    top_10_keywords = keywords[:10]
                    
                    fig = px.scatter(
                        x=[item[1] for item in top_10_keywords],
                        y=range(len(top_10_keywords)),
                        size=[item[1] for item in top_10_keywords],
                        hover_name=[item[0] for item in top_10_keywords],
                        title="Keyword Frequency Bubble Chart"
                    )
                    fig.update_yaxis(ticktext=[item[0] for item in top_10_keywords], tickvals=range(len(top_10_keywords)))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Keywords by rating
            if 'rating' in df.columns:
                st.subheader("Keywords by Rating")
                selected_rating = st.selectbox("Select rating:", sorted(df['rating'].unique()))
                
                rating_reviews = df[df['rating'] == selected_rating]
                rating_keywords = extract_keywords(rating_reviews['review_text'], n_words=10)
                
                if rating_keywords:
                    rating_keywords_df = pd.DataFrame(rating_keywords, columns=['Word', 'Frequency'])
                    
                    fig = px.bar(
                        rating_keywords_df,
                        x='Word',
                        y='Frequency',
                        title=f"Top Keywords for {selected_rating}-Star Reviews"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Review text column not found in the dataset.")
    
    # Feature 6: Product Performance
    elif selected_feature == "📈 Product Performance":
        st.markdown('<h2 class="feature-header">📈 Product Performance</h2>', unsafe_allow_html=True)
        
        if 'product_id' in df.columns or 'product_title' in df.columns:
            product_col = 'product_title' if 'product_title' in df.columns else 'product_id'
            
            # Product performance metrics
            product_stats = df.groupby(product_col).agg({
                'rating': ['mean', 'count', 'std'] if 'rating' in df.columns else ['count'],
                'helpful_votes': 'sum' if 'helpful_votes' in df.columns else 'count'
            }).round(2)
            
            product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
            product_stats = product_stats.reset_index()
            
            # Top products by rating
            if 'rating_mean' in product_stats.columns:
                st.subheader("Top Products by Average Rating")
                top_products = product_stats.nlargest(10, 'rating_mean')
                
                fig = px.bar(
                    top_products,
                    x='rating_mean',
                    y=product_col,
                    orientation='h',
                    title="Top 10 Products by Average Rating"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Product review volume
            if 'rating_count' in product_stats.columns:
                st.subheader("Most Reviewed Products")
                most_reviewed = product_stats.nlargest(10, 'rating_count')
                
                fig = px.bar(
                    most_reviewed,
                    x='rating_count',
                    y=product_col,
                    orientation='h',
                    title="Top 10 Most Reviewed Products"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Product performance table
            st.subheader("Product Performance Summary")
            st.dataframe(product_stats.head(20))
            
            # Product comparison
            st.subheader("Product Comparison")
            if len(df[product_col].unique()) > 1:
                selected_products = st.multiselect(
                    "Select products to compare:",
                    df[product_col].unique(),
                    default=list(df[product_col].unique())[:5]
                )
                
                if selected_products and 'rating' in df.columns:
                    comparison_data = df[df[product_col].isin(selected_products)]
                    
                    fig = px.box(
                        comparison_data,
                        x=product_col,
                        y='rating',
                        title="Rating Distribution Comparison"
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Product identification columns not found in the dataset.")
    
    # Feature 7: Temporal Analysis
    elif selected_feature == "⏰ Temporal Analysis":
        st.markdown('<h2 class="feature-header">⏰ Temporal Analysis</h2>', unsafe_allow_html=True)
        
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Reviews over time
            st.subheader("Reviews Over Time")
            
            # Daily reviews
            daily_reviews = df.groupby(df[date_col].dt.date).size().reset_index(name='count')
            daily_reviews.columns = ['date', 'count']
            
            fig = px.line(
                daily_reviews,
                x='date',
                y='count',
                title="Daily Review Count"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly trends
            df['year_month'] = df[date_col].dt.to_period('M')
            monthly_stats = df.groupby('year_month').agg({
                'rating': 'mean' if 'rating' in df.columns else 'count',
                date_col: 'count'
            }).reset_index()
            
            monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)
            monthly_stats.columns = ['year_month', 'avg_rating', 'review_count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'rating' in df.columns:
                    fig = px.line(
                        monthly_stats,
                        x='year_month',
                        y='avg_rating',
                        title="Average Rating Trends"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    monthly_stats,
                    x='year_month',
                    y='review_count',
                    title="Monthly Review Volume"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal patterns
            st.subheader("Seasonal Patterns")
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            
            seasonal_data = df.groupby('month').size().reset_index(name='count')
            seasonal_data['month_name'] = pd.to_datetime(seasonal_data['month'], format='%m').dt.strftime('%B')
            
            fig = px.bar(
                seasonal_data,
                x='month_name',
                y='count',
                title="Reviews by Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date columns found in the dataset.")
    
    # Feature 8: User Behavior Analysis
    elif selected_feature == "👥 User Behavior Analysis":
        st.markdown('<h2 class="feature-header">👥 User Behavior Analysis</h2>', unsafe_allow_html=True)
        
        # Helpful votes analysis
        if 'helpful_votes' in df.columns and 'total_votes' in df.columns:
            st.subheader("Review Helpfulness Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Helpful votes distribution
                fig = px.histogram(
                    df,
                    x='helpful_votes',
                    nbins=50,
                    title="Helpful Votes Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Helpful ratio distribution
                fig = px.histogram(
                    df_processed,
                    x='helpful_ratio',
                    nbins=50,
                    title="Helpful Ratio Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Helpfulness vs rating
            if 'rating' in df.columns:
                st.subheader("Helpfulness vs Rating")
                
                helpfulness_by_rating = df.groupby('rating').agg({
                    'helpful_votes': 'mean',
                    'helpful_ratio': 'mean'
                }).reset_index()
                
                fig = px.bar(
                    helpfulness_by_rating,
                    x='rating',
                    y='helpful_votes',
                    title="Average Helpful Votes by Rating"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Verified purchase analysis
        if 'verified_purchase' in df.columns:
            st.subheader("Verified Purchase Analysis")
            
            verified_stats = df['verified_purchase'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=verified_stats.values,
                    names=verified_stats.index,
                    title="Verified vs Non-Verified Purchases"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'rating' in df.columns:
                    # Rating by verification status
                    verification_rating = df.groupby('verified_purchase')['rating'].mean()
                    
                    fig = px.bar(
                        x=verification_rating.index,
                        y=verification_rating.values,
                        title="Average Rating by Verification Status"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Review length vs engagement
        if 'review_length' in df_processed.columns and 'helpful_votes' in df.columns:
            st.subheader("Review Length vs Engagement")
            
            # Scatter plot
            fig = px.scatter(
                df_processed.sample(min(1000, len(df_processed))),
                x='review_length',
                y='helpful_votes',
                title="Review Length vs Helpful Votes",
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature 9: Interactive Review Explorer
    elif selected_feature == "🔍 Interactive Review Explorer":
        st.markdown('<h2 class="feature-header">🔍 Interactive Review Explorer</h2>', unsafe_allow_html=True)
        
        st.subheader("Filter and Explore Reviews")
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Rating filter
            if 'rating' in df.columns:
                rating_filter = st.multiselect(
                    "Filter by Rating:",
                    sorted(df['rating'].unique()),
                    default=sorted(df['rating'].unique())
                )
            else:
                rating_filter = []

        with col2:
            # Product filter
            if 'product_title' in df.columns:
                product_filter = st.multiselect(
                    "Filter by Product:",
                    df['product_title'].unique(),
                    default=df['product_title'].unique()
                )
            else:
                product_filter = []

        with col3:
            # Verified purchase filter
            if 'verified_purchase' in df.columns:
                verified_filter = st.multiselect(
                    "Filter by Verified Purchase:",
                    df['verified_purchase'].unique(),
                    default=df['verified_purchase'].unique()
                )
            else:
                verified_filter = []

        # Filter data based on selected filters
        filtered_df = df[
            df['rating'].isin(rating_filter) &
            df['product_title'].isin(product_filter) &
            df['verified_purchase'].isin(verified_filter)
        ]
        
        st.write(f"Showing {len(filtered_df)} reviews after applying filters.")
        
        # Display the filtered data
        st.dataframe(filtered_df[['product_title', 'rating', 'review_title', 'review_text', 'verified_purchase', 'review_date']].head(10))

        # Review text exploration
        st.subheader("Explore Reviews")
        selected_review = st.selectbox(
            "Select a Review:",
            filtered_df['review_title'].values
        )
        
        review_text = filtered_df[filtered_df['review_title'] == selected_review]['review_text'].values[0]
        st.write(f"**Review Text:** {review_text}")
    
    # Feature 10: Review Quality Assessment
    elif selected_feature == "📋 Review Quality Assessment":
        st.markdown('<h2 class="feature-header">📋 Review Quality Assessment</h2>', unsafe_allow_html=True)
        
        if 'review_text' in df.columns:
            st.subheader("Assessing Review Quality")
            
            # Create a text length and word count metric
            st.write("**Review Length vs Quality**")
            fig = px.scatter(
                df_processed,
                x='review_length',
                y='rating',
                color='rating',
                title="Review Length vs Rating",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Review helpfulness analysis
            if 'helpful_votes' in df.columns:
                st.subheader("Review Helpfulness vs Rating")
                helpfulness_data = df[['helpful_votes', 'rating']].dropna()
                fig = px.scatter(
                    helpfulness_data,
                    x='helpful_votes',
                    y='rating',
                    title="Helpful Votes vs Rating",
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)

            # Word count and rating correlation
            st.write("**Word Count vs Rating**")
            fig = px.scatter(
                df_processed,
                x='word_count',
                y='rating',
                color='rating',
                title="Word Count vs Rating",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

            # Review quality analysis based on sentiment
            if 'sentiment' in df.columns:
                st.subheader("Review Sentiment vs Rating")
                sentiment_data = df[['sentiment', 'rating']].dropna()
                sentiment_counts = sentiment_data.groupby(['sentiment', 'rating']).size().reset_index(name='count')
                
                fig = px.bar(
                    sentiment_counts,
                    x='sentiment',
                    y='count',
                    color='rating',
                    title="Review Sentiment vs Rating",
                    labels={'sentiment': 'Sentiment', 'count': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Review text column not found in the dataset.")

                
