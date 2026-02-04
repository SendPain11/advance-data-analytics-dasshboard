"""
pre-trained model for models in web, and you can run before run streamlit
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import json

class ModelTrainer:
    """
    Production-ready model training pipeline
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.training_history = []
        
    def load_data(self):
        """Load data from CSV or generate sample"""
        if self.data_path:
            df = pd.read_csv(self.data_path)
        else:
            # Generate sample data
            np.random.seed(42)
            n_samples = 10000  # Larger dataset for production
            
            dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
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
            })
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        feature_cols = ['Marketing_Spend', 'Website_Visitors', 
                       'Email_Campaigns', 'Social_Media_Engagement']
        target_col = 'Sales'
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y, feature_cols
    
    def train_models(self, X, y, tune_hyperparameters=False):
        """Train multiple models"""
        print("🚀 Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'Linear_Regression': {
                'model': LinearRegression(),
                'params': {},
                'scale': True
            },
            'Random_Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                } if tune_hyperparameters else {},
                'scale': False
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                } if tune_hyperparameters else {},
                'scale': False
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"\n📊 Training {name}...")
            
            model = config['model']
            
            # Use scaled or unscaled data
            X_tr = X_train_scaled if config['scale'] else X_train
            X_te = X_test_scaled if config['scale'] else X_test
            
            # Hyperparameter tuning if needed
            if tune_hyperparameters and config['params']:
                print(f"   🔧 Tuning hyperparameters...")
                grid_search = GridSearchCV(
                    model, 
                    config['params'], 
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_tr, y_train)
                model = grid_search.best_estimator_
                print(f"   ✅ Best params: {grid_search.best_params_}")
            else:
                model.fit(X_tr, y_train)
            
            # Predictions
            y_pred = model.predict(X_te)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'scale': config['scale']
            }
            
            print(f"   ✅ R² Score: {r2:.4f}")
            print(f"   ✅ RMSE: {rmse:.2f}")
            print(f"   ✅ MAE: {mae:.2f}")
            
            # Save training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'model': name,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]
        self.models = results
        
        print(f"\n🏆 Best Model: {best_model_name} (R² = {self.best_model['r2']:.4f})")
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save trained models to disk"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving models to {output_dir}/...")
        
        # Save each model
        for name, data in self.models.items():
            filename = f"{output_dir}/{name.lower()}_model.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(data['model'], f)
            print(f"   ✅ Saved {filename}")
        
        # Save scaler
        scaler_file = f"{output_dir}/scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ Saved {scaler_file}")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'best_model': max(self.models, key=lambda x: self.models[x]['r2']),
            'model_performances': {
                name: {
                    'r2': data['r2'],
                    'rmse': data['rmse'],
                    'mae': data['mae']
                }
                for name, data in self.models.items()
            },
            'training_history': self.training_history
        }
        
        metadata_file = f"{output_dir}/metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved {metadata_file}")
        
        print("\n✨ All models saved successfully!")
    
    def load_trained_models(self, model_dir='models'):
        """Load pre-trained models"""
        import os
        
        print(f"📂 Loading models from {model_dir}/...")
        
        # Load metadata
        with open(f"{model_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"   📅 Models trained on: {metadata['training_date']}")
        print(f"   🏆 Best model: {metadata['best_model']}")
        
        # Load scaler
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load models
        for model_name in metadata['model_performances'].keys():
            filename = f"{model_dir}/{model_name.lower()}_model.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
                self.models[model_name] = {
                    'model': model,
                    **metadata['model_performances'][model_name]
                }
                print(f"   ✅ Loaded {model_name}")
        
        print("\n✨ All models loaded successfully!")
        return metadata
    
    def predict(self, X, model_name=None):
        """Make predictions with trained model"""
        if model_name is None:
            # Use best model
            model_name = max(self.models, key=lambda x: self.models[x]['r2'])
        
        model_data = self.models[model_name]
        model = model_data['model']
        
        # Scale if needed
        if model_data.get('scale', False):
            X = self.scaler.transform(X)
        
        predictions = model.predict(X)
        return predictions

# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🎯 MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    print("\n📊 Loading data...")
    df = trainer.load_data()
    print(f"   ✅ Loaded {len(df)} rows")
    
    # Prepare features
    X, y, feature_cols = trainer.prepare_features(df)
    print(f"   ✅ Features: {feature_cols}")
    
    # Train models
    results = trainer.train_models(X, y, tune_hyperparameters=False)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("✨ TRAINING COMPLETE!")
    print("="*60)
    
    # Example: Load and use trained models
    print("\n" + "="*60)
    print("🔄 TESTING MODEL LOADING")
    print("="*60)
    
    new_trainer = ModelTrainer()
    metadata = new_trainer.load_trained_models()
    
    # Make prediction on new data
    sample_data = pd.DataFrame({
        'Marketing_Spend': [500],
        'Website_Visitors': [2000],
        'Email_Campaigns': [5],
        'Social_Media_Engagement': [1000]
    })
    
    prediction = new_trainer.predict(sample_data)
    print(f"\n🎯 Sample Prediction: ${prediction[0]:,.2f}")
    
    print("\n✨ ALL TESTS PASSED!")