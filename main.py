import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title("Amazon Commerce Reviews EDA Dashboard")

# Sidebar for file upload
st.sidebar.title("Upload your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display dataset information
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Basic information about the dataset
    st.subheader("Basic Information")
    st.write(df.info())

    # Check for missing values
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0])

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Feature 1: Sentiment Distribution (Assumed 'sentiment' column)
    if 'sentiment' in df.columns:
        sentiment_count = df['sentiment'].value_counts()
        st.subheader("Sentiment Distribution")
        st.write(sentiment_count)

        # Sentiment Distribution Plot
        fig, ax = plt.subplots()
        sentiment_count.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # Feature 2: Review Rating Distribution (Assumed 'rating' column)
    if 'rating' in df.columns:
        st.subheader("Review Rating Distribution")
        fig = px.histogram(df, x='rating', title="Review Rating Distribution")
        st.plotly_chart(fig)

    # Feature 3: Word Cloud of Review Text (Assumed 'reviewText' column)
    if 'reviewText' in df.columns:
        st.subheader("Word Cloud for Review Text")
        text = " ".join(review for review in df['reviewText'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array(), use_column_width=True)

    # Feature 4: Rating vs Sentiment (Assumed 'rating' and 'sentiment' columns)
    if 'rating' in df.columns and 'sentiment' in df.columns:
        st.subheader("Rating vs Sentiment")
        fig = px.box(df, x='sentiment', y='rating', title="Box Plot: Rating vs Sentiment")
        st.plotly_chart(fig)

    # Feature 5: Review Length vs Rating (Assumed 'reviewText' and 'rating' columns)
    if 'reviewText' in df.columns and 'rating' in df.columns:
        df['review_length'] = df['reviewText'].apply(len)
        st.subheader("Review Length vs Rating")
        fig = px.scatter(df, x='review_length', y='rating', title="Scatter Plot: Review Length vs Rating")
        st.plotly_chart(fig)

    # Feature 6: Review Count by Product (Assumed 'product' column)
    if 'product' in df.columns:
        st.subheader("Review Count by Product")
        product_review_count = df['product'].value_counts().head(10)
        st.write(product_review_count)

        # Bar chart for review count by product
        fig, ax = plt.subplots()
        product_review_count.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # Feature 7: Rating Distribution by Review Sentiment (Assumed 'sentiment' and 'rating' columns)
    if 'sentiment' in df.columns and 'rating' in df.columns:
        st.subheader("Rating Distribution by Review Sentiment")
        sentiment_rating = df.groupby('sentiment')['rating'].mean()
        st.write(sentiment_rating)

        # Sentiment vs Rating Box Plot
        fig, ax = plt.subplots()
        sns.boxplot(x='sentiment', y='rating', data=df, ax=ax)
        st.pyplot(fig)

    # Feature 8: Correlation Heatmap (Numerical features only)
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to proceed.")
