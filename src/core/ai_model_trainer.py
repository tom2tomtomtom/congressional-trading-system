"""
AI Model Training Framework for Congressional Trading Intelligence
Prepares and trains machine learning models for trading signal generation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using scikit-learn models only.")

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI model training"""
    model_type: str = "random_forest"  # random_forest, gradient_boost, neural_network
    target_variable: str = "future_return"  # future_return, signal_strength, success_probability
    lookback_days: int = 30
    prediction_horizon: int = 7  # days
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    model_save_path: str = "models/"

class CongressionalTradingDataProcessor:
    """Processes congressional trading data for ML training"""
    
    def __init__(self, db_path: str = "congressional_data.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load congressional trades with member info
            query = """
            SELECT 
                ct.*,
                cm.committee_assignments,
                cm.leadership_position,
                cm.party,
                cm.state
            FROM congressional_trades ct
            LEFT JOIN congress_members cm ON ct.member_name = cm.name
            WHERE ct.trade_date >= date('now', '-2 years')
            ORDER BY ct.trade_date
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} trading records for training")
            return df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature engineering for ML models"""
        if df.empty:
            return df
        
        # Convert dates
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        
        # Basic features
        df['filing_delay'] = (df['filing_date'] - df['trade_date']).dt.days
        df['trade_amount_mid'] = (df['amount_min'] + df['amount_max']) / 2
        df['is_purchase'] = (df['trade_type'] == 'Purchase').astype(int)
        df['is_sale'] = (df['trade_type'] == 'Sale').astype(int)
        
        # Member features
        df['has_leadership'] = df['leadership_position'].notna().astype(int)
        df['committee_count'] = df['committee_assignments'].str.count(',') + 1
        
        # Timing features
        df['day_of_week'] = df['trade_date'].dt.dayofweek
        df['month'] = df['trade_date'].dt.month
        df['quarter'] = df['trade_date'].dt.quarter
        
        # Market features (if available)
        if 'close_price' in df.columns:
            df['price_volume_ratio'] = df['close_price'] / (df['volume'] + 1)
        
        # Encode categorical variables
        categorical_columns = ['party', 'sector', 'owner_type']
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].fillna('Unknown'))
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        if df.empty:
            return df
        
        # Sort by symbol and date for proper calculation
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Calculate future returns (if we have price data)
        if 'close_price' in df.columns:
            df['future_return_7d'] = df.groupby('symbol')['close_price'].pct_change(periods=7).shift(-7)
            df['future_return_30d'] = df.groupby('symbol')['close_price'].pct_change(periods=30).shift(-30)
        
        # Create success probability based on historical performance
        # (This would be calculated from actual market performance data)
        df['success_probability'] = np.random.uniform(0.3, 0.8, len(df))  # Placeholder
        
        # Create signal strength based on multiple factors
        df['signal_strength'] = (
            df['filing_delay'].clip(0, 180) / 180 * 0.3 +  # Filing delay factor
            df['trade_amount_mid'].rank(pct=True) * 0.4 +   # Trade size factor
            df['has_leadership'] * 0.3                       # Leadership factor
        )
        
        return df
    
    def prepare_ml_dataset(self, config: ModelConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare final dataset for ML training"""
        df = self.load_training_data()
        if df.empty:
            return np.array([]), np.array([]), []
        
        df = self.create_features(df)
        df = self.create_target_variables(df)
        
        # Select features for training
        feature_columns = [
            'filing_delay', 'trade_amount_mid', 'is_purchase', 'is_sale',
            'has_leadership', 'committee_count', 'day_of_week', 'month', 'quarter'
        ]
        
        # Add encoded categorical features
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        feature_columns.extend(encoded_cols)
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Remove rows with missing target variable
        target_col = f"{config.target_variable}"
        if target_col not in df.columns:
            target_col = 'signal_strength'  # Fallback
        
        df_clean = df[available_features + [target_col]].dropna()
        
        if df_clean.empty:
            logger.warning("No clean data available for training")
            return np.array([]), np.array([]), []
        
        X = df_clean[available_features].values
        y = df_clean[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y, available_features

class AIModelTrainer:
    """Trains and manages AI models for congressional trading prediction"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.data_processor = CongressionalTradingDataProcessor()
    
    def train_random_forest_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-self.config.train_test_split, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=self.config.cross_validation_folds)
        
        results = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': model.feature_importances_
        }
        
        return results
    
    def train_gradient_boost_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-self.config.train_test_split, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        results = {
            'model': model,
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': model.feature_importances_
        }
        
        return results
    
    def train_neural_network_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Neural Network model (if TensorFlow available)"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network training")
            return {}
        
        # Build model
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-self.config.train_test_split, random_state=42
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'model': model,
            'history': history.history,
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        
        return results
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all available models and return results"""
        logger.info("Starting AI model training pipeline...")
        
        # Prepare data
        X, y, feature_names = self.data_processor.prepare_ml_dataset(self.config)
        
        if X.size == 0:
            logger.error("No training data available")
            return {}
        
        results = {
            'feature_names': feature_names,
            'data_shape': X.shape,
            'models': {}
        }
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        rf_results = self.train_random_forest_model(X, y)
        if rf_results:
            results['models']['random_forest'] = rf_results
            self.models['random_forest'] = rf_results['model']
        
        # Train Gradient Boosting
        logger.info("Training Gradient Boosting model...")
        gb_results = self.train_gradient_boost_model(X, y)
        if gb_results:
            results['models']['gradient_boost'] = gb_results
            self.models['gradient_boost'] = gb_results['model']
        
        # Train Neural Network
        if TENSORFLOW_AVAILABLE:
            logger.info("Training Neural Network model...")
            nn_results = self.train_neural_network_model(X, y)
            if nn_results:
                results['models']['neural_network'] = nn_results
                self.models['neural_network'] = nn_results['model']
        
        logger.info(f"Training complete. Trained {len(self.models)} models.")
        return results
    
    def save_models(self, save_path: str = None):
        """Save trained models to disk"""
        import os
        save_path = save_path or self.config.model_save_path
        os.makedirs(save_path, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                model.save(f"{save_path}/{name}_model.h5")
            else:
                joblib.dump(model, f"{save_path}/{name}_model.pkl")
        
        # Save data processor
        joblib.dump(self.data_processor, f"{save_path}/data_processor.pkl")
        
        logger.info(f"Models saved to {save_path}")
    
    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Scale features
        features_scaled = self.data_processor.scaler.transform(features)
        
        return model.predict(features_scaled)

def create_sample_training_data():
    """Create sample training data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample congressional trading data
    data = {
        'member_name': [f'Member_{i%50}' for i in range(n_samples)],
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'], n_samples),
        'trade_date': pd.date_range('2022-01-01', periods=n_samples, freq='D'),
        'filing_date': pd.date_range('2022-01-01', periods=n_samples, freq='D') + pd.Timedelta(days=30),
        'trade_type': np.random.choice(['Purchase', 'Sale'], n_samples),
        'amount_min': np.random.uniform(1000, 50000, n_samples),
        'amount_max': np.random.uniform(50000, 500000, n_samples),
        'party': np.random.choice(['Democrat', 'Republican', 'Independent'], n_samples),
        'sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Energy'], n_samples),
        'owner_type': np.random.choice(['Self', 'Spouse', 'Child'], n_samples),
        'committee_assignments': 'Technology,Finance',
        'leadership_position': np.random.choice([None, 'Chair', 'Ranking Member'], n_samples),
        'close_price': np.random.uniform(50, 300, n_samples),
        'volume': np.random.uniform(1000000, 10000000, n_samples)
    }
    
    return pd.DataFrame(data)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = create_sample_training_data()
    print(f"Created sample dataset with {len(sample_data)} records")
    
    # Initialize trainer
    config = ModelConfig(
        model_type="random_forest",
        target_variable="signal_strength"
    )
    
    trainer = AIModelTrainer(config)
    
    # Train models
    results = trainer.train_all_models()
    
    if results:
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Data shape: {results['data_shape']}")
        print(f"Features: {len(results['feature_names'])}")
        
        for model_name, model_results in results['models'].items():
            print(f"\n{model_name.upper()}:")
            if 'test_score' in model_results:
                print(f"  Test Score: {model_results['test_score']:.3f}")
            if 'cv_mean' in model_results:
                print(f"  CV Score: {model_results['cv_mean']:.3f} ± {model_results['cv_std']:.3f}")
        
        # Save models
        trainer.save_models()
        print("\n✅ Models saved successfully")
    else:
        print("❌ No models were trained")

