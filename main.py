import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Dataset - EDA Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-header {
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
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try to load from URL first
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
        df = pd.read_csv(url, sep=';')
        return df
    except:
        # If URL fails, create sample data structure
        st.warning("⚠️ Could not load data from UCI repository. Please upload your own bank marketing dataset.")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset"""
    # Create a copy
    df_processed = df.copy()
    
    # Convert categorical variables to numerical for certain analyses
    le = LabelEncoder()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    df_encoded = df_processed.copy()
    for col in categorical_cols:
        df_encoded[col + '_encoded'] = le.fit_transform(df_processed[col])
    
    return df_processed, df_encoded

def main():
    st.markdown('<h1 class="main-header">🏦 Bank Marketing Dataset - EDA Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Please upload a bank marketing dataset to continue.")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            return
    
    # Preprocess data
    df_processed, df_encoded = preprocess_data(df)
    
    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    features = [
        "📊 Dataset Overview",
        "📈 Statistical Summary", 
        "🎯 Target Variable Analysis",
        "📉 Distribution Analysis",
        "🔗 Correlation Analysis",
        "📊 Categorical Analysis",
        "⏰ Time Series Analysis",
        "🤖 Feature Importance",
        "🔍 Interactive Filtering",
        "📋 Data Quality Report"
    ]
    
    selected_feature = st.sidebar.selectbox("Select Analysis Feature:", features)
    
    # Feature 1: Dataset Overview
    if selected_feature == "📊 Dataset Overview":
        st.markdown('<h2 class="feature-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Numerical Features", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 10 rows:**")
            st.dataframe(df.head(10))
            
        with col2:
            st.write("**Dataset Schema:**")
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(schema_df)
    
    # Feature 2: Statistical Summary
    elif selected_feature == "📈 Statistical Summary":
        st.markdown('<h2 class="feature-header">📈 Statistical Summary</h2>', unsafe_allow_html=True)
        
        # Numerical columns summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.subheader("Numerical Features Summary")
            st.dataframe(df[numerical_cols].describe())
            
            # Box plots for numerical features
            st.subheader("Box Plots - Numerical Features")
            selected_num_cols = st.multiselect("Select numerical columns:", numerical_cols, default=list(numerical_cols)[:4])
            
            if selected_num_cols:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_num_cols[:4]
                )
                
                for i, col in enumerate(selected_num_cols[:4]):
                    row = i // 2 + 1
                    col_pos = i % 2 + 1
                    fig.add_trace(
                        go.Box(y=df[col], name=col),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=600, title_text="Box Plots for Numerical Features")
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("Categorical Features Summary")
            for col in categorical_cols:
                st.write(f"**{col}:**")
                value_counts = df[col].value_counts()
                st.write(f"Unique values: {len(value_counts)}")
                st.write(value_counts.head(10))
                st.write("---")
    
    # Feature 3: Target Variable Analysis
    elif selected_feature == "🎯 Target Variable Analysis":
        st.markdown('<h2 class="feature-header">🎯 Target Variable Analysis</h2>', unsafe_allow_html=True)
        
        # Assuming 'y' is the target variable (common in bank marketing datasets)
        target_col = st.selectbox("Select target variable:", df.columns)
        
        if target_col:
            # Target distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Target Distribution")
                target_counts = df[target_col].value_counts()
                
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title=f"Distribution of {target_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Target Statistics")
                st.write(f"**Total samples:** {len(df)}")
                for value, count in target_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"**{value}:** {count} ({percentage:.1f}%)")
            
            # Target vs other features
            st.subheader("Target vs Other Features")
            feature_col = st.selectbox("Select feature to compare with target:", 
                                     [col for col in df.columns if col != target_col])
            
            if feature_col:
                if df[feature_col].dtype == 'object':
                    # Categorical feature
                    crosstab = pd.crosstab(df[feature_col], df[target_col])
                    fig = px.bar(
                        crosstab,
                        title=f"{feature_col} vs {target_col}",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Numerical feature
                    fig = px.box(
                        df, 
                        x=target_col, 
                        y=feature_col,
                        title=f"{feature_col} distribution by {target_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Feature 4: Distribution Analysis
    elif selected_feature == "📉 Distribution Analysis":
        st.markdown('<h2 class="feature-header">📉 Distribution Analysis</h2>', unsafe_allow_html=True)
        
        analysis_type = st.radio("Select analysis type:", ["Numerical Features", "Categorical Features"])
        
        if analysis_type == "Numerical Features":
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                selected_col = st.selectbox("Select numerical column:", numerical_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df, 
                        x=selected_col, 
                        nbins=30,
                        title=f"Distribution of {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Q-Q plot
                    from scipy import stats
                    fig = go.Figure()
                    
                    # Generate Q-Q plot data
                    sorted_data = np.sort(df[selected_col].dropna())
                    n = len(sorted_data)
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
                    
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_data,
                        mode='markers',
                        name='Data'
                    ))
                    
                    # Add reference line
                    fig.add_trace(go.Scatter(
                        x=[min(theoretical_quantiles), max(theoretical_quantiles)],
                        y=[min(sorted_data), max(sorted_data)],
                        mode='lines',
                        name='Reference Line'
                    ))
                    
                    fig.update_layout(
                        title=f"Q-Q Plot for {selected_col}",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                selected_col = st.selectbox("Select categorical column:", categorical_cols)
                
                # Bar chart
                value_counts = df[selected_col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature 5: Correlation Analysis
    elif selected_feature == "🔗 Correlation Analysis":
        st.markdown('<h2 class="feature-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            # Correlation matrix
            correlation_matrix = df[numerical_cols].corr()
            
            # Heatmap
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.subheader("Strong Correlations (|r| > 0.5)")
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_corr.append({
                            'Feature 1': correlation_matrix.columns[i],
                            'Feature 2': correlation_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                st.dataframe(strong_corr_df.sort_values('Correlation', key=abs, ascending=False))
            else:
                st.write("No strong correlations found.")
        else:
            st.write("Not enough numerical features for correlation analysis.")
    
    # Feature 6: Categorical Analysis
    elif selected_feature == "📊 Categorical Analysis":
        st.markdown('<h2 class="feature-header">📊 Categorical Analysis</h2>', unsafe_allow_html=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            selected_col = st.selectbox("Select categorical column:", categorical_cols)
            
            # Value counts
            value_counts = df[selected_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Frequency of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Two-way analysis
            if len(categorical_cols) > 1:
                st.subheader("Two-way Categorical Analysis")
                second_col = st.selectbox(
                    "Select second categorical column:",
                    [col for col in categorical_cols if col != selected_col]
                )
                
                if second_col:
                    # Cross-tabulation
                    crosstab = pd.crosstab(df[selected_col], df[second_col])
                    
                    # Heatmap
                    fig = px.imshow(
                        crosstab,
                        text_auto=True,
                        aspect="auto",
                        title=f"Cross-tabulation: {selected_col} vs {second_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No categorical features available for analysis.")
    
    # Feature 7: Time Series Analysis
    elif selected_feature == "⏰ Time Series Analysis":
        st.markdown('<h2 class="feature-header">⏰ Time Series Analysis</h2>', unsafe_allow_html=True)
        
        # Look for date/time columns
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                date_cols.append(col)
        
        if date_cols:
            selected_date_col = st.selectbox("Select date/time column:", date_cols)
            
            # Try to parse as datetime
            try:
                df['parsed_date'] = pd.to_datetime(df[selected_date_col])
                
                # Time series plot
                time_series_data = df.groupby('parsed_date').size().reset_index(name='count')
                
                fig = px.line(
                    time_series_data,
                    x='parsed_date',
                    y='count',
                    title=f"Time Series of Records by {selected_date_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except:
                # If not datetime, analyze as categorical
                time_counts = df[selected_date_col].value_counts().sort_index()
                
                fig = px.bar(
                    x=time_counts.index,
                    y=time_counts.values,
                    title=f"Distribution by {selected_date_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No date/time columns found. Showing month analysis if available.")
            
            # Look for month column specifically
            month_cols = [col for col in df.columns if 'month' in col.lower()]
            if month_cols:
                month_col = month_cols[0]
                month_counts = df[month_col].value_counts()
                
                fig = px.bar(
                    x=month_counts.index,
                    y=month_counts.values,
                    title=f"Distribution by {month_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature 8: Feature Importance
    elif selected_feature == "🤖 Feature Importance":
        st.markdown('<h2 class="feature-header">🤖 Feature Importance</h2>', unsafe_allow_html=True)
        
        # Select target variable
        target_col = st.selectbox("Select target variable:", df.columns)
        
        if target_col:
            try:
                # Prepare data for machine learning
                X = df_encoded[[col for col in df_encoded.columns if col != target_col and '_encoded' in col]]
                y = df_encoded[target_col + '_encoded'] if target_col + '_encoded' in df_encoded.columns else df_encoded[target_col]
                
                # Train Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(
                    feature_importance.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Feature Importances"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top features
                st.subheader("Top 10 Most Important Features")
                st.dataframe(feature_importance.head(10))
                
            except Exception as e:
                st.error(f"Error calculating feature importance: {str(e)}")
    
    # Feature 9: Interactive Filtering
    elif selected_feature == "🔍 Interactive Filtering":
        st.markdown('<h2 class="feature-header">🔍 Interactive Filtering</h2>', unsafe_allow_html=True)
        
        st.subheader("Filter Dataset")
        
        # Create filters
        filters = {}
        
        # Categorical filters
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_vals = df[col].unique()
            selected_vals = st.multiselect(
                f"Filter by {col}:",
                unique_vals,
                default=unique_vals
            )
            filters[col] = selected_vals
        
        # Numerical filters
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected_range = st.slider(
                f"Filter by {col}:",
                min_val,
                max_val,
                (min_val, max_val)
            )
            filters[col] = selected_range
        
        # Apply filters
        filtered_df = df.copy()
        for col, filter_val in filters.items():
            if col in categorical_cols:
                filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            else:
                filtered_df = filtered_df[
                    (filtered_df[col] >= filter_val[0]) & 
                    (filtered_df[col] <= filter_val[1])
                ]
        
        # Show filtered results
        st.subheader(f"Filtered Dataset ({len(filtered_df)} records)")
        st.dataframe(filtered_df)
        
        # Quick stats on filtered data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filtered Records", len(filtered_df))
        with col2:
            st.metric("Percentage of Total", f"{len(filtered_df)/len(df)*100:.1f}%")
        with col3:
            st.metric("Removed Records", len(df) - len(filtered_df))
    
    # Feature 10: Data Quality Report
    elif selected_feature == "📋 Data Quality Report":
        st.markdown('<h2 class="feature-header">📋 Data Quality Report</h2>', unsafe_allow_html=True)
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        }).sort_values('Missing Count', ascending=False)
        
        # Plot missing values
        fig = px.bar(
            missing_df[missing_df['Missing Count'] > 0],
            x='Column',
            y='Missing Count',
            title="Missing Values by Column"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(missing_df)
        
        # Duplicate analysis
        st.subheader("Duplicate Analysis")
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Records", duplicate_count)
        
        if duplicate_count > 0:
            st.write("Sample duplicate records:")
            st.dataframe(df[df.duplicated()].head())
        
        # Data types analysis
        st.subheader("Data Types Summary")
        dtype_summary = df.dtypes.value_counts()
        
        fig = px.pie(
            values=dtype_summary.values,
            names=dtype_summary.index,
            title="Distribution of Data Types"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier detection (for numerical columns)
        st.subheader("Outlier Detection")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            outlier_summary = []
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': len(outliers),
                    'Outlier Percentage': len(outliers) / len(df) * 100
                })
            
            outlier_df = pd.DataFrame(outlier_summary)
            st.dataframe(outlier_df)
        
        # Generate insights
        st.subheader("📊 Data Quality Insights")
        
        insights = []
        
        # Missing data insights
        if missing_data.sum() > 0:
            insights.append(f"⚠️ Dataset has {missing_data.sum()} missing values across {(missing_data > 0).sum()} columns")
        else:
            insights.append("✅ No missing values detected")
        
        # Duplicate insights
        if duplicate_count > 0:
            insights.append(f"⚠️ Found {duplicate_count} duplicate records ({duplicate_count/len(df)*100:.1f}%)")
        else:
            insights.append("✅ No duplicate records found")
        
        # Outlier insights
        if len(numerical_cols) > 0:
            total_outliers = sum([len(df[(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                     (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))]) 
                                for col in numerical_cols])
            if total_outliers > 0:
                insights.append(f"⚠️ Detected potential outliers in numerical columns")
            else:
                insights.append("✅ No significant outliers detected")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 Bank Marketing Dataset EDA Dashboard** | Built with Streamlit and Plotly")

if __name__ == "__main__":
    main()
