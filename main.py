import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()

# Streamlit App Title
st.title("Comprehensive EDA for Bank Marketing Dataset")

# Sidebar for user inputs
st.sidebar.title("App Navigation")
sidebar_option = st.sidebar.radio(
    "Choose a section",
    ["Dataset Overview", "Statistics Summary", "Target Variable Distribution", "Correlation Analysis", "Feature Distribution", "Categorical Features", "Pairplot"]
)

# Dataset Overview
if sidebar_option == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("The dataset contains information from direct marketing campaigns (phone calls) of a Portuguese banking institution.")
    st.write("The goal is to predict if a client will subscribe to a term deposit (target variable `y`).")
    st.write("Total records:", df.shape[0])
    st.write("Features:", df.columns.tolist())
    st.write("Data Types:", df.dtypes)

# Data Summary (Statistics)
if sidebar_option == "Statistics Summary":
    st.subheader("Statistics Summary")
    st.write("This section provides the descriptive statistics of the numerical features in the dataset.")
    st.write(df.describe())

# Target Variable Distribution
if sidebar_option == "Target Variable Distribution":
    st.subheader("Target Variable Distribution")
    target_counts = df['y'].value_counts()
    fig = px.pie(target_counts, names=target_counts.index, values=target_counts.values, title="Term Deposit Subscription Distribution")
    st.plotly_chart(fig)

# Correlation Heatmap
if sidebar_option == "Correlation Analysis":
    st.subheader("Correlation Analysis")
    st.write("This heatmap shows the correlation between numerical features.")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Feature Distribution (Numerical)
if sidebar_option == "Feature Distribution":
    st.subheader("Feature Distribution")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        st.write(f"### Distribution of {col}")
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=30)
        st.plotly_chart(fig)

# Categorical Feature Distribution
if sidebar_option == "Categorical Features":
    st.subheader("Categorical Feature Distribution")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        st.write(f"### Distribution of {col}")
        fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, title=f'Distribution of {col}', labels={'index': col, col: 'Count'})
        st.plotly_chart(fig)

# Pairplot
if sidebar_option == "Pairplot":
    st.subheader("Pairplot of Numerical Features")
    st.write("A pairplot to show the relationships between the numerical features.")
    sns.pairplot(df.select_dtypes(include=['int64', 'float64']))
    st.pyplot()

# Additional features for interactivity
st.sidebar.subheader("Advanced Filtering Options")
contact_filter = st.sidebar.slider("Filter by Number of Contacts in Campaign", min_value=1, max_value=50, value=10, step=1)
filtered_data = df[df['campaign'] == contact_filter]
st.sidebar.write(f"Showing data for {contact_filter} contacts")
st.write(filtered_data)

# Customization: Add Dropdown for Selecting Multiple Features
feature_selector = st.sidebar.multiselect("Select Features for Target vs Features Analysis", df.columns.tolist())
if feature_selector:
    st.subheader("Target vs Selected Features")
    for feature in feature_selector:
        fig = px.box(df, x='y', y=feature, title=f"Target vs {feature}")
        st.plotly_chart(fig)

