# 📊 Advanced Data Analytics & Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

A comprehensive, production-ready data analytics dashboard built with Python, showcasing advanced data science, machine learning, and statistical analysis capabilities. Perfect for data-driven business intelligence and predictive analytics.

## 🎯 Project Overview

This dashboard is a full-featured analytics platform that demonstrates enterprise-level data science workflows, from exploratory data analysis to machine learning model deployment. It's designed to showcase professional Python programming skills and real-world data science applications.

### ✨ Key Features

- **📈 Exploratory Data Analysis (EDA)**
  - Interactive data visualization with Plotly
  - Statistical distribution analysis
  - Correlation matrix and heatmaps
  - Automated outlier detection using IQR method
  - Descriptive statistics with skewness & kurtosis

- **🤖 Machine Learning Models**
  - Multiple algorithm comparison (Linear Regression, Random Forest, Gradient Boosting)
  - Real-time model training OR pre-trained model loading
  - Automated model evaluation with comprehensive metrics
  - Feature importance analysis
  - Interactive predictions vs actual visualization

- **🎨 Customer Segmentation**
  - K-Means clustering implementation
  - Elbow method for optimal cluster selection
  - 3D interactive cluster visualization
  - Cluster characteristics profiling

- **📊 Statistical Hypothesis Testing**
  - Independent T-Tests for group comparison
  - ANOVA for multi-group analysis
  - Pearson correlation testing
  - Automated p-value interpretation

- **⏰ Time Series Analysis**
  - Trend decomposition
  - Seasonality detection
  - Moving average forecasting
  - Interactive time range selection

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SendPain11/advanced-analytics-dashboard.git
cd advanced-analytics-dashboard
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎮 Running the Application

### Option 1: Quick Demo (Recommended for Portfolio)

**Real-time model training - No setup required!**

```bash
streamlit run app.py
```

The dashboard will:
- ✅ Automatically generate sample data
- ✅ Train models in real-time (< 5 seconds)
- ✅ Display interactive visualizations
- ✅ Show predictions immediately

**Perfect for:** Demos, interviews, portfolio presentations

---

### Option 2: Production Mode (With Pre-trained Models)

**For production-style deployment with saved models**

#### Step 1: Train and Save Models

```bash
python train_model.py
```

This will:
- Train all ML models with your data
- Save models to `models/` directory
- Create metadata for model versioning
- Generate performance reports

Output:
```
🚀 Starting model training...
📊 Training Linear_Regression...
   ✅ R² Score: 0.8542
📊 Training Random_Forest...
   ✅ R² Score: 0.9123
📊 Training Gradient_Boosting...
   ✅ R² Score: 0.9087
🏆 Best Model: Random_Forest
💾 Saving models to models/...
   ✅ Saved models/linear_regression_model.pkl
   ✅ Saved models/random_forest_model.pkl
   ✅ Saved models/gradient_boosting_model.pkl
✨ All models saved successfully!
```

#### Step 2: Run Dashboard with Pre-trained Models

```bash
streamlit run app.py
```

Then in the sidebar:
- ☑️ Check "Use Pre-trained Models"
- Dashboard will load models from disk
- Faster predictions (no training needed)

**Perfect for:** Production deployment, scheduled retraining, large datasets

---

## 📦 Dependencies

**requirements.txt:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
scipy>=1.11.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
advanced-analytics-dashboard/
│
├── app.py                      # Main Streamlit dashboard (REQUIRED)
├── train_model.py              # Optional: Separate training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Created by train_model.py
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   └── metadata.json           # Model versioning info
│
├── data/                       # Optional: Your datasets
│   └── custom_data.csv
│
└── media/                # Optional: For README
    ├── screnshoot.png
    └── video.mkv
```

## 💻 Usage Guide

### 1. Dashboard Interface

When you run `streamlit run app.py`, you'll see:

**Sidebar:**
- 🔄 **Use Pre-trained Models**: Toggle to load saved models
- 📊 **Data Source**: Choose sample data or upload CSV
- ⚙️ **Analysis Types**: Select which analyses to run

**Main Panel:**
- 📋 Data Overview with key metrics
- 📊 Interactive visualizations
- 🤖 ML model comparisons
- 📈 Statistical test results

### 2. Sample Workflows

#### Workflow A: Quick Data Analysis
```
1. Run: streamlit run app.py
2. Select: "Generate Sample Sales Data"
3. Choose: "Exploratory Data Analysis"
4. Explore: Distributions, correlations, outliers
```

#### Workflow B: Build Predictive Models
```
1. Run: streamlit run app.py
2. Select: "Generate Sample Sales Data"
3. Choose: "Predictive Modeling"
4. Adjust: Test set size slider
5. Compare: 3 ML algorithms automatically
6. Review: R², RMSE, MAE metrics
```

#### Workflow C: Customer Segmentation
```
1. Run: streamlit run app.py
2. Select: "Generate Sample Sales Data"
3. Choose: "Clustering Analysis"
4. Adjust: Number of clusters
5. Visualize: 3D cluster plot
6. Analyze: Cluster characteristics
```

### 3. Upload Your Own Data

```
1. In sidebar, select: "Upload Your Own CSV"
2. Click: Upload CSV file
3. Dashboard will automatically:
   - Detect numeric columns
   - Run selected analyses
   - Generate visualizations
```

## 🎓 Skills Demonstrated

This project showcases proficiency in:

- **Python Programming**: Advanced techniques, OOP, functional programming
- **Data Science**: Statistical analysis, hypothesis testing, data visualization
- **Machine Learning**: Supervised learning, clustering, model evaluation, hyperparameter tuning
- **Data Engineering**: ETL pipelines, feature engineering, data preprocessing
- **Web Development**: Streamlit framework, interactive dashboards, UX design
- **Software Engineering**: Code organization, documentation, version control, testing
- **Production Deployment**: Model persistence, caching, performance optimization

## 🔧 Technical Deep Dive

### Machine Learning Pipeline

```python
# Automatic workflow in app.py:
1. Data Ingestion → Generate or upload
2. Data Validation → Type checking, null handling
3. Train-Test Split → Configurable ratio
4. Feature Scaling → StandardScaler normalization
5. Model Training → 3 algorithms in parallel
6. Evaluation → Multiple metrics calculation
7. Visualization → Interactive Plotly charts
```

### Model Training Options

**Option A: Real-time Training (app.py)**
- Trains models on-demand when user selects "Predictive Modeling"
- Uses `@st.cache_data` for performance
- Best for: Demos, small datasets, rapid iteration

**Option B: Pre-trained Models (train_model.py)**
- Trains models once, saves to disk
- Loads pre-trained models in dashboard
- Supports hyperparameter tuning
- Best for: Production, large datasets, scheduled retraining

### Performance Optimization

- **Caching**: `@st.cache_data` and `@st.cache_resource` for expensive operations
- **Lazy Loading**: Data loaded only when needed
- **Vectorization**: NumPy/Pandas for fast computations
- **Efficient Algorithms**: Scikit-learn's optimized implementations

## 📊 Model Information

### Algorithms Implemented

1. **Linear Regression**
   - Fast baseline model
   - Interpretable coefficients
   - Assumes linear relationships

2. **Random Forest**
   - Ensemble method with 100 trees
   - Handles non-linear relationships
   - Provides feature importance
   - Usually best performer

3. **Gradient Boosting**
   - Sequential ensemble method
   - Strong predictive power
   - Captures complex patterns

### Evaluation Metrics

- **R² Score**: Variance explained by model (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## 🔒 Data Privacy & Security

- ✅ No data stored permanently
- ✅ All processing in-memory
- ✅ Sample data uses seeded randomization
- ✅ Uploaded data not logged
- ✅ Models saved only if explicitly trained via `train_model.py`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

## 📝 Future Enhancements

**Planned Features:**
- [ ] Deep Learning models (LSTM, Neural Networks)
- [ ] Real-time data streaming (Apache Kafka)
- [ ] REST API for predictions (FastAPI)
- [ ] Advanced time series (ARIMA, Prophet, Transformer models)
- [ ] NLP module for text analytics
- [ ] Automated report generation (PDF/Excel)
- [ ] User authentication (OAuth)
- [ ] Database integration (PostgreSQL, MongoDB)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] A/B testing framework
- [ ] Model monitoring & drift detection

## 🐛 Troubleshooting

### Common Issues

**Issue: Port already in use**
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

**Issue: Models not loading**
```bash
# Solution: Ensure models/ directory exists and run:
python train_model.py
```

**Issue: Import errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sendy Prismana Nurferian**
- GitHub: [@SendPain11](https://github.com/SendPain11)
- LinkedIn: [Sendy Prismana Nurferian](https://www.linkedin.com/in/sendy-prismana-nurferian-95a27b213/)
- Email: sendyprisma02@gmail.com
- Documentation Project: [advance-data-analytics-dashboard](https://advance-analytics-data-dashboard.streamlit.app/)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Amazing framework for data apps
- [Scikit-learn](https://scikit-learn.org/) - Robust ML library
- [Plotly](https://plotly.com/) - Beautiful interactive visualizations
- Open source community for inspiration

## 📞 Support

Need help?
- 📧 Email: sendyprism02@gmail.com
- 💬 Open an [issue](https://github.com/SendPain11/advanced-analytics-dashboard/issues)
- 🔗 Connect on [LinkedIn](https://www.linkedin.com/in/sendy-prismana-nurferian-95a27b213/)

---

## 🎯 Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (recommended)
streamlit run app.py

# Production mode
python train_model.py          # Train & save models
streamlit run app.py           # Run dashboard
# → Check "Use Pre-trained Models" in sidebar

# Run on different port
streamlit run app.py --server.port 8502

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

---

**⭐ If you find this project useful, please give it a star on GitHub!**

**Made with ❤️ and Python** 🐍

**See You Next Time all!**
---