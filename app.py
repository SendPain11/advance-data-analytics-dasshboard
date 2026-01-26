"""
Advanced Data Analytics & Machine Learning Dashboard
====================================================
A comprehensive analytics platform showcasing Python, ML, and Data Science capabilities

Features:
- Exploratory Data Analysis (EDA)
- Machine Learning Models (with pre-trained model support)
- Customer Segmentation (K-Means Clustering)
- Statistical Hypothesis Testing
- Time Series Analysis

Author: Your Name
GitHub: github.com/yourusername
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import dengan error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(f"❌ Plotly import error: {e}")
    st.info("Please ensure plotly is installed: pip install plotly>=5.17.0")
    st.stop()

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from datetime import datetime, timedelta
import pickle
import os
import json

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# TITLE AND DESCRIPTION
# ============================================================

st.title("🚀 Advanced Data Analytics & ML Dashboard")
st.markdown("**A comprehensive analytics platform demonstrating Python, ML, and Data Science capabilities**")
st.markdown("---")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_pretrained_models():
    """Load pre-trained models from disk"""
    models = {}
    
    try:
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        for model_name, perf in metadata['model_performances'].items():
            model_file = f"models/{model_name.lower()}_model.pkl"
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                models[model_name] = {
                    'model': model,
                    **perf,
                    'pretrained': True
                }
        
        return models, scaler, metadata
    except Exception as e:
        return None, None, None

@st.cache_data
def generate_sales_data(n_samples=1000):
    """Generate realistic sample sales data"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
    
    # Create realistic patterns
    trend = np.linspace(1000, 5000, n_samples)
    seasonality = 500 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
    noise = np.random.normal(0, 300, n_samples)
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': (trend + seasonality + noise).clip(min=0),
        'Marketing_Spend': np.random.uniform(100, 1000, n_samples),
        'Website_Visitors': np.random.randint(500, 5000, n_samples),
        'Email_Campaigns': np.random.randint(0, 10, n_samples),
        'Social_Media_Engagement': np.random.uniform(100, 2000, n_samples),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'Customer_Satisfaction': np.random.uniform(3.0, 5.0, n_samples)
    })
    
    return df

@st.cache_data
def generate_time_series_data(n_samples=500):
    """Generate sample time series data"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
    
    trend = np.linspace(100, 200, n_samples)
    seasonal = 20 * np.sin(np.arange(n_samples) * 2 * np.pi / 30)
    noise = np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'Date': dates,
        'Metric_A': trend + seasonal + noise,
        'Metric_B': trend * 0.8 - seasonal + noise * 1.5,
        'Metric_C': 150 + noise * 3
    })
    
    return df

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Pre-trained model option
    st.subheader("🔄 Model Settings")
    use_pretrained = st.checkbox("Use Pre-trained Models", value=False, 
                                  help="Load models from disk instead of training in real-time")
    
    if use_pretrained:
        if os.path.exists('models/metadata.json'):
            with open('models/metadata.json', 'r') as f:
                metadata = json.load(f)
            st.success("✅ Pre-trained models found!")
            with st.expander("📋 Model Info"):
                st.write(f"**Trained:** {metadata['training_date'][:10]}")
                st.write(f"**Best Model:** {metadata['best_model']}")
                st.write(f"**R² Score:** {metadata['model_performances'][metadata['best_model']]['r2']:.4f}")
        else:
            st.error("❌ No pre-trained models found")
            st.info("Run `python train_model.py` first")
            use_pretrained = False
    
    st.markdown("---")
    
    # Data source selection
    st.subheader("📊 Data Source")
    data_source = st.selectbox(
        "Select data source",
        ["Generate Sample Sales Data", "Generate Time Series Data", "Upload Your Own CSV"]
    )
    
    if data_source == "Generate Sample Sales Data":
        n_samples = st.slider("Number of samples", 100, 2000, 1000, step=100)
    elif data_source == "Generate Time Series Data":
        n_samples = st.slider("Number of samples", 100, 1000, 500, step=50)
    
    st.markdown("---")
    
    # Analysis type selection
    st.subheader("🎯 Analysis Types")
    analysis_type = st.multiselect(
        "Select analyses to perform",
        ["Exploratory Data Analysis", "Predictive Modeling", "Clustering Analysis", 
         "Time Series Forecasting", "Statistical Tests"],
        default=["Exploratory Data Analysis", "Predictive Modeling"],
        help="Choose which analyses to run on your data"
    )
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Features:**
        - Real-time or pre-trained ML models
        - Interactive visualizations
        - Statistical analysis
        - Customer segmentation
        - Time series forecasting
        
        **Technologies:**
        - Python, Pandas, NumPy
        - Scikit-learn, Plotly
        - Streamlit
        """)

# ============================================================
# DATA LOADING
# ============================================================

if data_source == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
    else:
        st.warning("⚠️ Please upload a CSV file to continue")
        st.info("💡 Or select a sample data source from the sidebar")
        st.stop()
elif data_source == "Generate Sample Sales Data":
    with st.spinner("Generating sample sales data..."):
        df = generate_sales_data(n_samples)
else:
    with st.spinner("Generating time series data..."):
        df = generate_time_series_data(n_samples)

# ============================================================
# DATA OVERVIEW
# ============================================================

st.header("📋 Data Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Records", f"{len(df):,}")
with col2:
    st.metric("📈 Features", len(df.columns))
with col3:
    memory_usage = df.memory_usage(deep=True).sum() / 1024
    st.metric("💾 Memory Usage", f"{memory_usage:.1f} KB")
with col4:
    missing_values = df.isnull().sum().sum()
    st.metric("❓ Missing Values", missing_values)

# Data preview
with st.expander("🔍 View Raw Data", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col2:
        st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("---")

# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

if "Exploratory Data Analysis" in analysis_type:
    st.header("📊 Exploratory Data Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "🎯 Outliers"])
        
        with tab1:
            st.subheader("Variable Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_col = st.selectbox("Select variable to analyze", numeric_cols, key="dist_col")
                
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[selected_col],
                    name="Histogram",
                    nbinsx=30,
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title=f"Distribution of {selected_col}",
                    xaxis_title=selected_col,
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[selected_col],
                    name=selected_col,
                    marker_color='lightcoral',
                    boxmean='sd'
                ))
                
                fig.update_layout(
                    title=f"Box Plot of {selected_col}",
                    yaxis_title=selected_col,
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Descriptive statistics
            st.subheader("📈 Descriptive Statistics")
            stats_df = df[numeric_cols].describe().T
            stats_df['skewness'] = df[numeric_cols].skew()
            stats_df['kurtosis'] = df[numeric_cols].kurtosis()
            
            st.dataframe(
                stats_df.style.format("{:.2f}").background_gradient(cmap='YlOrRd', axis=0),
                use_container_width=True
            )
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                labels=dict(color="Correlation")
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pairwise scatter plots
            if len(numeric_cols) >= 2:
                st.subheader("Pairwise Relationships")
                cols_to_plot = st.multiselect(
                    "Select variables for scatter matrix (2-4 recommended)",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))],
                    max_selections=4
                )
                
                if len(cols_to_plot) >= 2:
                    fig = px.scatter_matrix(
                        df,
                        dimensions=cols_to_plot,
                        title="Scatter Plot Matrix",
                        height=600
                    )
                    fig.update_traces(diagonal_visible=False, showupperhalf=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Outlier Detection (IQR Method)")
            
            outlier_col = st.selectbox("Select variable for outlier detection", numeric_cols, key="outlier_col")
            
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Total Outliers", len(outliers))
            with col2:
                st.metric("📉 Lower Bound", f"{lower_bound:.2f}")
            with col3:
                st.metric("📈 Upper Bound", f"{upper_bound:.2f}")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[outlier_col],
                mode='markers',
                name='Normal',
                marker=dict(color='lightblue', size=5)
            ))
            fig.add_trace(go.Scatter(
                x=outliers.index,
                y=outliers[outlier_col],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange", 
                         annotation_text="Upper Bound")
            fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange", 
                         annotation_text="Lower Bound")
            
            fig.update_layout(
                title=f"Outlier Detection for {outlier_col}",
                xaxis_title="Index",
                yaxis_title=outlier_col,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns found in the dataset")

    st.markdown("---")

# ============================================================
# PREDICTIVE MODELING
# ============================================================

if "Predictive Modeling" in analysis_type and data_source == "Generate Sample Sales Data":
    st.header("🤖 Predictive Modeling")
    
    st.markdown("""
    Building machine learning models to predict **Sales** based on marketing features.
    Comparing multiple algorithms to find the best performer.
    """)
    
    # Prepare data
    feature_cols = ['Marketing_Spend', 'Website_Visitors', 'Email_Campaigns', 'Social_Media_Engagement']
    target_col = 'Sales'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split
    col1, col2 = st.columns([2, 1])
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100
    with col2:
        st.metric("Train samples", f"{int(len(df) * (1-test_size)):,}")
        st.metric("Test samples", f"{int(len(df) * test_size):,}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    results = {}
    
    # Option 1: Use Pre-trained Models
    if use_pretrained:
        pretrained_models, pretrained_scaler, metadata = load_pretrained_models()
        
        if pretrained_models:
            st.info("🔄 Using pre-trained models from disk...")
            
            for name, model_data in pretrained_models.items():
                model = model_data['model']
                
                # Make predictions
                if 'Linear' in name:
                    X_test_scaled = pretrained_scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R²': r2_score(y_test, y_pred),
                    'pretrained': True
                }
            
            st.success(f"✅ Loaded {len(results)} pre-trained models!")
    
    # Option 2: Train Models from Scratch
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}... ({idx+1}/{len(models)})")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'pretrained': False
            }
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.text("✅ All models trained successfully!")
        progress_bar.empty()
        status_text.empty()
    
    # Display results
    st.subheader("📊 Model Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (name, metrics) in enumerate(results.items()):
        with [col1, col2, col3][idx]:
            # Model card
            with st.container():
                st.markdown(f"### {name}")
                
                if metrics.get('pretrained', False):
                    st.caption("🔄 Pre-trained Model")
                else:
                    st.caption("🆕 Newly Trained")
                
                st.metric("R² Score", f"{metrics['R²']:.4f}", 
                         help="Proportion of variance explained (0-1, higher is better)")
                st.metric("RMSE", f"{metrics['RMSE']:.2f}",
                         help="Root Mean Squared Error (lower is better)")
                st.metric("MAE", f"{metrics['MAE']:.2f}",
                         help="Mean Absolute Error (lower is better)")
    
    # Best model
    best_model_name = max(results, key=lambda x: results[x]['R²'])
    st.success(f"🏆 **Best Model:** {best_model_name} (R² = {results[best_model_name]['R²']:.4f})")
    
    # Detailed visualization
    st.subheader("📈 Detailed Model Analysis")
    
    model_choice = st.selectbox("Select model to visualize", list(results.keys()), key="model_viz_choice")
    
    tab1, tab2 = st.tabs(["Predictions vs Actual", "Residual Analysis"])
    
    with tab1:
        # Predictions vs Actual scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=results[model_choice]['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(size=6, color='blue', opacity=0.6),
            text=[f"Actual: ${a:.0f}<br>Predicted: ${p:.0f}" 
                  for a, p in zip(y_test, results[model_choice]['predictions'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), results[model_choice]['predictions'].min())
        max_val = max(y_test.max(), results[model_choice]['predictions'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f"{model_choice}: Predictions vs Actual Sales",
            xaxis_title="Actual Sales ($)",
            yaxis_title="Predicted Sales ($)",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Residual plot
        residuals = y_test - results[model_choice]['predictions']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[model_choice]['predictions'],
            y=residuals,
            mode='markers',
            marker=dict(size=6, color='green', opacity=0.6),
            name='Residuals'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
        
        fig.update_layout(
            title=f"Residual Plot - {model_choice}",
            xaxis_title="Predicted Sales ($)",
            yaxis_title="Residuals ($)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Residual", f"{residuals.mean():.2f}")
        with col2:
            st.metric("Std Residual", f"{residuals.std():.2f}")
    
    # Feature importance (for tree-based models)
    if model_choice in ['Random Forest', 'Gradient Boosting']:
        st.subheader("🎯 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': results[model_choice]['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance - {model_choice}",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Interpretation:** Higher importance means the feature has more influence on predictions.")

    st.markdown("---")

# ============================================================
# CLUSTERING ANALYSIS
# ============================================================

if "Clustering Analysis" in analysis_type and data_source == "Generate Sample Sales Data":
    st.header("🎨 Customer Segmentation (K-Means Clustering)")
    
    st.markdown("""
    Identifying distinct customer segments based on sales behavior, marketing spend, and satisfaction levels.
    """)
    
    # Select features for clustering
    cluster_features = ['Sales', 'Marketing_Spend', 'Customer_Satisfaction']
    X_cluster = df[cluster_features].dropna()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Optimal Number of Clusters")
        
        # Elbow method
        inertias = []
        silhouette_scores = []
        K_range = range(2, 11)
        
        with st.spinner("Calculating optimal clusters..."):
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(K_range),
            y=inertias,
            mode='lines+markers',
            marker=dict(size=10, color='blue'),
            line=dict(width=2)
        ))
        fig.update_layout(
            title="Elbow Method",
            xaxis_title="Number of Clusters",
            yaxis_title="Inertia (Within-cluster sum of squares)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Look for the 'elbow' point where inertia starts decreasing more slowly")
    
    with col2:
        st.subheader("⚙️ Clustering Configuration")
        
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        
        # Perform clustering
        with st.spinner(f"Clustering data into {n_clusters} segments..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        df_cluster = X_cluster.copy()
        df_cluster['Cluster'] = clusters
        
        # Cluster distribution
        cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
        
        fig = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title="Cluster Distribution",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D Visualization
    st.subheader("📍 3D Cluster Visualization")
    
    fig = px.scatter_3d(
        df_cluster,
        x='Sales',
        y='Marketing_Spend',
        z='Customer_Satisfaction',
        color='Cluster',
        title="Customer Segments in 3D Space",
        height=600,
        labels={
            'Sales': 'Sales ($)',
            'Marketing_Spend': 'Marketing Spend ($)',
            'Customer_Satisfaction': 'Satisfaction Score'
        },
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("📋 Cluster Characteristics")
    
    cluster_summary = df_cluster.groupby('Cluster')[cluster_features].agg(['mean', 'median', 'std'])
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    
    st.dataframe(
        cluster_summary.style.format("{:.2f}").background_gradient(cmap='YlOrRd', axis=0),
        use_container_width=True
    )
    
    # Cluster insights
    with st.expander("💡 Cluster Insights", expanded=True):
        for cluster_id in range(n_clusters):
            cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
            st.markdown(f"**Cluster {cluster_id}** ({len(cluster_data)} customers):")
            st.write(f"- Average Sales: ${cluster_data['Sales'].mean():,.0f}")
            st.write(f"- Average Marketing Spend: ${cluster_data['Marketing_Spend'].mean():,.0f}")
            st.write(f"- Average Satisfaction: {cluster_data['Customer_Satisfaction'].mean():.2f}/5.0")
            st.markdown("")

    st.markdown("---")

# ============================================================
# STATISTICAL TESTS
# ============================================================

if "Statistical Tests" in analysis_type and data_source == "Generate Sample Sales Data":
    st.header("📊 Statistical Hypothesis Testing")
    
    st.markdown("""
    Performing statistical tests to validate business hypotheses and identify significant differences.
    """)
    
    tab1, tab2, tab3 = st.tabs(["T-Test", "ANOVA", "Correlation Test"])
    
    with tab1:
        st.subheader("Independent T-Test: Sales Comparison Between Regions")
        st.markdown("**H₀:** There is no significant difference in sales between the two regions")
        
        col1, col2 = st.columns(2)
        with col1:
            r1 = st.selectbox("Select Region 1", df['Region'].unique(), index=0, key='t_r1')
        with col2:
            r2 = st.selectbox("Select Region 2", df['Region'].unique(), index=1, key='t_r2')
        
        if r1 != r2:
            sales_r1 = df[df['Region'] == r1]['Sales']
            sales_r2 = df[df['Region'] == r2]['Sales']
            
            t_stat, p_value = stats.ttest_ind(sales_r1, sales_r2)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("T-Statistic", f"{t_stat:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            with col3:
                st.metric("Significance Level", "α = 0.05")
            with col4:
                if p_value < 0.05:
                    st.error("❌ Reject H₀")
                else:
                    st.success("✅ Accept H₀")
            
            # Interpretation
            if p_value < 0.05:
                st.warning(f"**Conclusion:** There IS a statistically significant difference in sales between {r1} and {r2} (p < 0.05)")
            else:
                st.info(f"**Conclusion:** There is NO statistically significant difference in sales between {r1} and {r2} (p ≥ 0.05)")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Box(y=sales_r1, name=r1, marker_color='lightblue', boxmean='sd'))
            fig.add_trace(go.Box(y=sales_r2, name=r2, marker_color='lightcoral', boxmean='sd'))
            fig.update_layout(
                title=f"Sales Distribution: {r1} vs {r2}",
                yaxis_title="Sales ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please select two different regions")
    
    with tab2:
        st.subheader("One-Way ANOVA: Sales Across All Regions")
        st.markdown("**H₀:** There is no significant difference in sales across all regions")
        
        region_groups = [df[df['Region'] == region]['Sales'].values for region in df['Region'].unique()]
        f_stat, p_value_anova = stats.f_oneway(*region_groups)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("F-Statistic", f"{f_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_value_anova:.4f}")
        with col3:
            st.metric("Groups", len(df['Region'].unique()))
        with col4:
            if p_value_anova < 0.05:
                st.error("❌ Reject H₀")
            else:
                st.success("✅ Accept H₀")
        
        # Interpretation
        if p_value_anova < 0.05:
            st.warning("**Conclusion:** At least one region has significantly different sales (p < 0.05)")
        else:
            st.info("**Conclusion:** No significant differences in sales across regions (p ≥ 0.05)")
        
        # Visualization
        fig = go.Figure()
        for region in df['Region'].unique():
            fig.add_trace(go.Box(
                y=df[df['Region'] == region]['Sales'],
                name=region,
                boxmean='sd'
            ))
        fig.update_layout(
            title="Sales Distribution Across All Regions",
            yaxis_title="Sales ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Pearson Correlation Test")
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Select Variable 1", numeric_cols, index=0, key='corr_v1')
        with col2:
            var2 = st.selectbox("Select Variable 2", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key='corr_v2')
        
        if var1 != var2:
            st.markdown(f"**H₀:** There is no linear correlation between {var1} and {var2}")
            
            corr_coef, p_value_corr = stats.pearsonr(df[var1], df[var2])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Correlation (r)", f"{corr_coef:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value_corr:.4f}")
            with col3:
                # Interpret correlation strength
                if abs(corr_coef) < 0.3:
                    strength = "Weak"
                elif abs(corr_coef) < 0.7:
                    strength = "Moderate"
                else:
                    strength = "Strong"
                st.metric("Strength", strength)
            with col4:
                if p_value_corr < 0.05:
                    st.success("✅ Significant")
                else:
                    st.warning("⚠️ Not Significant")
            
            # Interpretation
            if p_value_corr < 0.05:
                direction = "positive" if corr_coef > 0 else "negative"
                st.warning(f"**Conclusion:** There IS a statistically significant {direction} correlation between {var1} and {var2} (p < 0.05)")
            else:
                st.info(f"**Conclusion:** No statistically significant correlation between {var1} and {var2} (p ≥ 0.05)")
            
            # Visualization
            fig = px.scatter(
                df,
                x=var1,
                y=var2,
                trendline='ols',
                title=f"{var1} vs {var2}",
                labels={var1: var1, var2: var2}
            )
            fig.update_traces(marker=dict(size=5, opacity=0.6))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please select two different variables")

    st.markdown("---")

# ============================================================
# TIME SERIES FORECASTING
# ============================================================

if "Time Series Forecasting" in analysis_type:
    st.header("📈 Time Series Analysis & Forecasting")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_to_forecast = st.selectbox("Select metric to analyze", numeric_cols, key='ts_metric')
        
        tab1, tab2 = st.tabs(["Decomposition", "Forecasting"])
        
        with tab1:
            st.subheader("Time Series Decomposition")
            
            window = st.slider("Moving average window (days)", 7, 90, 30, step=7)
            
            # Calculate components
            df['Trend'] = df[metric_to_forecast].rolling(window=window, center=True).mean()
            df['Detrended'] = df[metric_to_forecast] - df['Trend']
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Original Series", "Trend Component", "Detrended (Seasonality + Noise)"),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df[metric_to_forecast], name='Original', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Trend'], name='Trend', line=dict(color='red', width=3)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Detrended'], name='Detrended', line=dict(color='green')),
                row=3, col=1
            )
            
            fig.update_layout(height=800, showlegend=False)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Simple Moving Average Forecast")
            
            forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, step=7)
            ma_window = st.slider("Moving average window for forecast", 7, 60, 30, step=7, key='forecast_ma')
            
            # Calculate forecast
            last_date = df['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            forecast_value = df[metric_to_forecast].tail(ma_window).mean()
            forecast_values = [forecast_value] * forecast_days
            
            # Calculate prediction interval (simple approach)
            std_dev = df[metric_to_forecast].tail(ma_window).std()
            upper_bound = [forecast_value + 1.96 * std_dev] * forecast_days
            lower_bound = [forecast_value - 1.96 * std_dev] * forecast_days
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[metric_to_forecast],
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_values,
                name='Forecast',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"{metric_to_forecast} - Historical & Forecast",
                xaxis_title="Date",
                yaxis_title=metric_to_forecast,
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast Value", f"{forecast_value:.2f}")
            with col2:
                st.metric("Std Deviation", f"{std_dev:.2f}")
            with col3:
                st.metric("Forecast Period", f"{forecast_days} days")
            
            st.info("💡 **Note:** This is a simple moving average forecast. For production use, consider advanced methods like ARIMA, Prophet, or LSTM.")
    else:
        st.warning("⚠️ No 'Date' column found in the dataset. Time series analysis requires temporal data.")

    st.markdown("---")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🚀 Built with Modern Data Science Stack</h3>
        <p><strong>Technologies:</strong> Python 🐍 | Streamlit | Plotly | Scikit-learn | Pandas | NumPy</p>
        <p><strong>Features:</strong> EDA • Machine Learning • Statistical Testing • Clustering • Time Series</p>
        <p style='margin-top: 20px;'>
            <em>Toggle "Use Pre-trained Models" in sidebar to switch between real-time and pre-trained ML models</em>
        </p>
        <p style='margin-top: 10px; font-size: 0.9em;'>
            Made with ❤️ for Data Science Portfolio | 
            <a href='https://github.com/yourusername' target='_blank'>GitHub</a> | 
            <a href='https://linkedin.com/in/yourprofile' target='_blank'>LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)