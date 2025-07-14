import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Amazon Reviews EDA Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_amazon_data():
    """Load Amazon Reviews dataset directly from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/Amazon%20Commerce%20Reviews.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    """Preprocess the data for analysis"""
    df_processed = df.copy()
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown')
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def main():
    st.markdown('<div class="main-header">📦 Amazon Reviews EDA Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data loading section - Automatically loads from UCI
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["Amazon Reviews Data"])
    
    if data_source == "Amazon Reviews Data":
        df = load_amazon_data()
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Data Filters")
    
    # Categorical columns filter
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    selected_cat_cols = st.sidebar.multiselect(
        "Select Categorical Columns:",
        options=categorical_cols,
        default=categorical_cols[:5] if len(categorical_cols) > 5 else categorical_cols
    )
    
    # Numerical columns filter
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    selected_num_cols = st.sidebar.multiselect(
        "Select Numerical Columns:",
        options=numerical_cols,
        default=numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
    )
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Overview", "📊 Distributions", "🔗 Correlations", "📉 Relationships", 
        "🎯 Target Analysis", "🔍 Dimensionality", "🤖 ML Insights", "📋 Data Quality"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df_processed))
        with col2:
            st.metric("Features", len(df_processed.columns))
        with col3:
            st.metric("Missing Values", df_processed.isnull().sum().sum())
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(df_processed.head(10), use_container_width=True)
        
        # Data types
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df_processed.columns,
                'Data Type': df_processed.dtypes.astype(str),
                'Non-Null Count': df_processed.count(),
                'Null Count': df_processed.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(df_processed.describe(), use_container_width=True)
    
    # TAB 2: DISTRIBUTIONS
    with tab2:
        st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
        
        # Numerical distributions
        if numerical_cols:
            st.subheader("Numerical Feature Distributions")
            
            col1, col2 = st.columns(2)
            with col1:
                plot_type = st.selectbox("Plot Type:", ["Histogram", "Box Plot", "Violin Plot"])
            with col2:
                bins = st.slider("Histogram Bins:", 10, 100, 30)
            
            for col in selected_num_cols:
                fig = go.Figure()
                
                if plot_type == "Histogram":
                    fig.add_trace(go.Histogram(
                        x=df_processed[col],
                        nbinsx=bins,
                        name=col,
                        opacity=0.7
                    ))
                elif plot_type == "Box Plot":
                    fig.add_trace(go.Box(
                        y=df_processed[col],
                        name=col,
                        boxpoints='outliers'
                    ))
                else:  # Violin Plot
                    fig.add_trace(go.Violin(
                        y=df_processed[col],
                        name=col,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                fig.update_layout(
                    title=f"{plot_type} - {col}",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical distributions
        if categorical_cols:
            st.subheader("Categorical Feature Distributions")
            
            for col in selected_cat_cols:
                value_counts = df_processed[col].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Bar Chart - {col}",
                        labels={'x': col, 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart - {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: CORRELATIONS
    with tab3:
        st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
        
        if len(numerical_cols) > 1:
            # Correlation matrix
            corr_matrix = df_processed[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: RELATIONSHIPS
    with tab4:
        st.markdown('<div class="section-header">Feature Relationships</div>', unsafe_allow_html=True)
        
        if len(numerical_cols) >= 2:
            # Scatter plot controls
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("X Variable:", numerical_cols)
            with col2:
                y_var = st.selectbox("Y Variable:", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)
            with col3:
                color_var = st.selectbox("Color By:", ['None'] + categorical_cols)
            
            # Create scatter plot
            if color_var != 'None':
                fig = px.scatter(
                    df_processed,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    title=f"{x_var} vs {y_var}",
                    opacity=0.7,
                    hover_data=numerical_cols[:3]
                )
            else:
                fig = px.scatter(
                    df_processed,
                    x=x_var,
                    y=y_var,
                    title=f"{x_var} vs {y_var}",
                    opacity=0.7
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: TARGET ANALYSIS
    with tab5:
        st.markdown('<div class="section-header">Target Variable Analysis</div>', unsafe_allow_html=True)
        
        if 'Class' in df_processed.columns:
            # Target distribution
            target_dist = df_processed['Class'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    x=target_dist.index,
                    y=target_dist.values,
                    title="Target Distribution",
                    labels={'x': 'Class', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=target_dist.values,
                    names=target_dist.index,
                    title="Target Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: DIMENSIONALITY
    with tab6:
        st.markdown('<div class="section-header">Dimensionality Reduction</div>', unsafe_allow_html=True)
        
        if len(numerical_cols) >= 2:
            # PCA
            st.subheader("Principal Component Analysis (PCA)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_processed[numerical_cols])
            
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            if 'Class' in df_processed.columns:
                pca_df['Class'] = df_processed['Class']
            
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Class',
                title="PCA - First Two Components"
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 7: ML INSIGHTS
    with tab7:
        st.markdown('<div class="section-header">Machine Learning Insights</div>', unsafe_allow_html=True)
        
        if 'Class' in df_processed.columns and len(numerical_cols) > 0:
            # Prepare data for ML
            df_ml = df_processed.copy()
            
            # Encode categorical variables
            le = LabelEncoder()
            for col in categorical_cols:
                if col != 'Class':
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            
            # Encode target
            df_ml['Class'] = le.fit_transform(df_ml['Class'])
            
            # Feature importance using Random Forest
            st.subheader("Feature Importance")
            
            X = df_ml.drop('Class', axis=1)
            y = df_ml['Class']
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance (Random Forest)"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
