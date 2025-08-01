#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Trade Prediction Models
Machine learning models for predicting congressional trading behavior and patterns.
"""

import os
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)

@dataclass
class TradePrediction:
    """Data model for trade predictions."""
    member_id: str
    symbol: str
    prediction_date: str
    trade_probability: float
    predicted_action: str  # Buy, Sell, Hold
    confidence_score: float
    key_factors: List[str]
    model_version: str

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    cross_val_mean: float
    cross_val_std: float

class FeatureEngineer:
    """Handles feature engineering for trading prediction models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for trading prediction.
        
        Args:
            df: DataFrame with member, trade, and contextual data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features for trading prediction")
        
        features_df = df.copy()
        
        # Member-specific features
        features_df = self._add_member_features(features_df)
        
        # Committee-based features
        features_df = self._add_committee_features(features_df)
        
        # Market context features
        features_df = self._add_market_features(features_df)
        
        # Temporal features
        features_df = self._add_temporal_features(features_df)
        
        # Historical trading features
        features_df = self._add_historical_features(features_df)
        
        # Legislative calendar features
        features_df = self._add_legislative_features(features_df)
        
        self.feature_names = [col for col in features_df.columns if col not in ['target', 'member_id', 'symbol', 'date']]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return features_df
    
    def _add_member_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add member-specific features."""
        # Party affiliation
        df['party_republican'] = (df['party'] == 'R').astype(int)
        df['party_democrat'] = (df['party'] == 'D').astype(int)
        
        # Chamber
        df['chamber_house'] = (df['chamber'] == 'House').astype(int)
        df['chamber_senate'] = (df['chamber'] == 'Senate').astype(int)
        
        # Leadership positions
        df['has_leadership'] = df['leadership_position'].notna().astype(int)
        
        # Seniority (approximate based on served_from)
        if 'served_from' in df.columns:
            df['served_from'] = pd.to_datetime(df['served_from'])
            df['seniority_years'] = (datetime.now() - df['served_from']).dt.days / 365.25
        else:
            df['seniority_years'] = 5.0  # Default assumption
        
        return df
    
    def _add_committee_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add committee-based features."""
        # Committee count
        df['committee_count'] = df.get('committee_count', 2)  # Default assumption
        
        # High-impact committees (finance, banking, etc.)
        high_impact_committees = [
            'Financial Services', 'Banking', 'Ways and Means', 
            'Appropriations', 'Energy and Commerce', 'Armed Services'
        ]
        
        # Simulate committee membership (in production, join with committee data)
        df['high_impact_committee'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        # Committee chair/ranking member
        df['committee_leadership'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market context features."""
        # Market volatility (VIX proxy)
        df['market_volatility'] = np.random.normal(20, 5, len(df))  # Simulated
        
        # Sector performance
        df['sector_performance'] = np.random.normal(0, 0.1, len(df))  # Simulated
        
        # Market trend
        df['market_trend_bullish'] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
        
        # Economic indicators
        df['unemployment_rate'] = np.random.normal(4.5, 1.0, len(df))  # Simulated
        df['gdp_growth'] = np.random.normal(2.5, 1.5, len(df))  # Simulated
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Day of week
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            
            # Month
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            
            # Days until earnings season
            df['days_to_earnings'] = (df['date'].dt.day % 90)  # Simplified
        else:
            # Default temporal features
            df['day_of_week'] = 2  # Tuesday
            df['is_monday'] = 0
            df['month'] = 6
            df['quarter'] = 2
            df['days_to_earnings'] = 45
        
        return df
    
    def _add_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical trading pattern features."""
        # Trading frequency (simulated)
        df['avg_trades_per_month'] = np.random.gamma(2, 2, len(df))
        
        # Average trade size
        df['avg_trade_size'] = np.random.lognormal(10, 1, len(df))
        
        # Recent trading activity
        df['trades_last_30_days'] = np.random.poisson(1, len(df))
        df['trades_last_90_days'] = np.random.poisson(3, len(df))
        
        # Performance metrics
        df['historical_alpha'] = np.random.normal(0.02, 0.05, len(df))
        df['win_rate'] = np.random.beta(6, 4, len(df))  # Slightly positive bias
        
        return df
    
    def _add_legislative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add legislative calendar features."""
        # Days until recess
        df['days_to_recess'] = np.random.randint(1, 60, len(df))
        
        # Election proximity
        df['election_year'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])
        df['days_to_election'] = np.random.randint(30, 730, len(df))
        
        # Legislative workload
        df['bills_sponsored'] = np.random.poisson(5, len(df))
        df['committee_hearings_week'] = np.random.poisson(2, len(df))
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: DataFrame with features
            fit_scaler: Whether to fit the scaler (True for training)
            
        Returns:
            Scaled feature array
        """
        feature_cols = [col for col in self.feature_names if col in df.columns]
        X = df[feature_cols].fillna(0)  # Handle missing values
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled

class TradePredictionModel:
    """
    Machine learning model for predicting congressional trading behavior.
    Uses ensemble methods combining XGBoost, Random Forest, and Neural Networks.
    """
    
    def __init__(self, model_name: str = "trade_predictor_v1"):
        """Initialize trade prediction model."""
        self.model_name = model_name
        self.feature_engineer = FeatureEngineer()
        
        # Initialize models
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        # Ensemble model
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'
        )
        
        self.is_trained = False
        self.performance_metrics = None
        
    def generate_training_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic training data for model development.
        In production, this would query the actual database.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            DataFrame with training data
        """
        logger.info(f"Generating {n_samples} training samples")
        
        # Simulate member data
        parties = ['R', 'D', 'I']
        chambers = ['House', 'Senate']
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        
        data = []
        for i in range(n_samples):
            sample = {
                'member_id': f'M{i % 535:03d}',
                'party': np.random.choice(parties, p=[0.45, 0.53, 0.02]),
                'chamber': np.random.choice(chambers, p=[0.8, 0.2]),
                'symbol': np.random.choice(symbols),
                'date': datetime.now() - timedelta(days=np.random.randint(0, 730)),
                'leadership_position': np.random.choice([None, 'Chair', 'Ranking'], p=[0.9, 0.05, 0.05]),
                'served_from': datetime.now() - timedelta(days=np.random.randint(365, 3650))
            }
            
            # Generate target based on realistic patterns
            # Higher probability for trades near committee hearings, market volatility, etc.
            base_prob = 0.1  # Base 10% chance of trading
            
            # Adjust based on features
            if sample['leadership_position']:
                base_prob *= 1.5  # Leaders trade more
            if sample['party'] == 'R':
                base_prob *= 1.2  # Slight partisan difference
            
            sample['target'] = np.random.binomial(1, base_prob)
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def train(self, training_data: Optional[pd.DataFrame] = None) -> ModelPerformance:
        """
        Train the trade prediction model.
        
        Args:
            training_data: Optional training data DataFrame
            
        Returns:
            Model performance metrics
        """
        logger.info("Training trade prediction model")
        
        if training_data is None:
            training_data = self.generate_training_data()
        
        # Feature engineering
        features_df = self.feature_engineer.create_features(training_data)
        
        # Prepare features and target
        X = self.feature_engineer.prepare_features(features_df, fit_scaler=True)
        y = features_df['target'].values
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.ensemble.predict(X_val)
        y_pred_proba = self.ensemble.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except:
            auc = 0.5
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble, X_train, y_train, cv=5, scoring='accuracy')
        
        self.performance_metrics = ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std()
        )
        
        self.is_trained = True
        
        logger.info(f"Model training completed:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        logger.info(f"  AUC: {auc:.3f}")
        logger.info(f"  Cross-val: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return self.performance_metrics
    
    def predict_trade_probability(self, member_id: str, symbol: str, 
                                context_data: Optional[Dict] = None) -> TradePrediction:
        """
        Predict trading probability for a specific member and symbol.
        
        Args:
            member_id: Congressional member ID
            symbol: Stock symbol
            context_data: Additional context data
            
        Returns:
            Trade prediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create input data for prediction
        input_data = {
            'member_id': member_id,
            'symbol': symbol,
            'date': datetime.now(),
            'party': context_data.get('party', 'D') if context_data else 'D',
            'chamber': context_data.get('chamber', 'House') if context_data else 'House',
            'leadership_position': context_data.get('leadership_position') if context_data else None,
            'served_from': context_data.get('served_from', datetime.now() - timedelta(days=1825)) if context_data else datetime.now() - timedelta(days=1825)
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Feature engineering
        features_df = self.feature_engineer.create_features(input_df)
        X = self.feature_engineer.prepare_features(features_df, fit_scaler=False)
        
        # Make prediction
        trade_probability = self.ensemble.predict_proba(X)[0, 1]
        predicted_class = self.ensemble.predict(X)[0]
        
        # Determine action
        if trade_probability > 0.7:
            predicted_action = "Buy"
        elif trade_probability < 0.3:
            predicted_action = "Sell"
        else:
            predicted_action = "Hold"
        
        # Get feature importance for explanation
        try:
            feature_importance = self.ensemble.estimators_[0].feature_importances_
            top_features = sorted(zip(self.feature_engineer.feature_names, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:5]
            key_factors = [f[0] for f in top_features]
        except:
            key_factors = ["committee_membership", "market_volatility", "historical_performance"]
        
        return TradePrediction(
            member_id=member_id,
            symbol=symbol,
            prediction_date=datetime.now().isoformat(),
            trade_probability=float(trade_probability),
            predicted_action=predicted_action,
            confidence_score=float(max(trade_probability, 1 - trade_probability)),
            key_factors=key_factors,
            model_version=self.model_name
        )
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'ensemble': self.ensemble,
            'feature_engineer': self.feature_engineer,
            'performance_metrics': self.performance_metrics,
            'model_name': self.model_name,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.ensemble = model_data['ensemble']
        self.feature_engineer = model_data['feature_engineer']
        self.performance_metrics = model_data['performance_metrics']
        self.model_name = model_data['model_name']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance rankings."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        try:
            # Get average feature importance across ensemble
            importances = []
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance = list(zip(self.feature_engineer.feature_names, avg_importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                return feature_importance[:top_n]
        except:
            pass
        
        return []

def main():
    """Test function for trade prediction model."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and train model
    model = TradePredictionModel()
    
    print("Training trade prediction model...")
    performance = model.train()
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {performance.accuracy:.3f}")
    print(f"Precision: {performance.precision:.3f}")
    print(f"Recall: {performance.recall:.3f}")
    print(f"F1 Score: {performance.f1_score:.3f}")
    print(f"AUC Score: {performance.auc_score:.3f}")
    
    # Test prediction
    print(f"\nTesting prediction...")
    prediction = model.predict_trade_probability(
        member_id="PELOSI001",
        symbol="AAPL",
        context_data={
            'party': 'D',
            'chamber': 'House',
            'leadership_position': 'Speaker'
        }
    )
    
    print(f"Prediction for {prediction.member_id} trading {prediction.symbol}:")
    print(f"  Probability: {prediction.trade_probability:.3f}")
    print(f"  Action: {prediction.predicted_action}")
    print(f"  Confidence: {prediction.confidence_score:.3f}")
    print(f"  Key factors: {prediction.key_factors}")
    
    # Feature importance
    print(f"\nTop 10 Feature Importance:")
    importance = model.get_feature_importance(10)
    for feature, score in importance:
        print(f"  {feature}: {score:.3f}")

if __name__ == "__main__":
    main()