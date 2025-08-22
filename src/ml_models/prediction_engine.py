#!/usr/bin/env python3
"""
Congressional Trading ML Prediction Engine
Advanced machine learning models for predicting trading patterns and risk assessment
"""

import os
import sys
import json
import logging
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN, KMeans
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingPredictionEngine:
    """
    Comprehensive ML engine for congressional trading predictions and analysis
    """
    
    def __init__(self, data_path: str = 'src/data'):
        self.data_path = data_path
        self.members_df = None
        self.trades_df = None
        self.features_df = None
        
        # Models storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Model parameters
        self.model_configs = {
            'risk_classifier': {
                'type': 'classification',
                'target': 'high_risk',
                'models': ['random_forest', 'xgboost', 'logistic_regression']
            },
            'amount_predictor': {
                'type': 'regression',
                'target': 'amount_avg',
                'models': ['random_forest_reg', 'xgboost_reg']
            },
            'anomaly_detector': {
                'type': 'unsupervised',
                'models': ['isolation_forest', 'one_class_svm']
            },
            'trade_classifier': {
                'type': 'classification',
                'target': 'transaction_type',
                'models': ['random_forest', 'gradient_boosting']
            }
        }
        
        # Feature engineering parameters
        self.risk_thresholds = {
            'high_amount': 100000,
            'extreme_amount': 1000000,
            'late_filing': 45,
            'extreme_delay': 90
        }
    
    def load_data(self) -> bool:
        """Load and preprocess data for ML models"""
        try:
            # Load members data
            members_file = os.path.join(self.data_path, 'congressional_members_full.json')
            if os.path.exists(members_file):
                with open(members_file, 'r') as f:
                    members_data = json.load(f)
                self.members_df = pd.DataFrame(members_data)
                logger.info(f"✅ Loaded {len(self.members_df)} congressional members")
            
            # Load trades data
            trades_file = os.path.join(self.data_path, 'congressional_trades_full.json')
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
                self.trades_df = pd.DataFrame(trades_data)
                
                # Process dates
                self.trades_df['transaction_date'] = pd.to_datetime(
                    self.trades_df['transaction_date'], errors='coerce'
                )
                self.trades_df['filing_date'] = pd.to_datetime(
                    self.trades_df['filing_date'], errors='coerce'
                )
                
                logger.info(f"✅ Loaded {len(self.trades_df)} congressional trades")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def engineer_features(self) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        if self.trades_df is None:
            return pd.DataFrame()
        
        df = self.trades_df.copy()
        
        # Basic financial features
        df['amount_avg'] = (df['amount_from'] + df['amount_to']) / 2
        df['amount_range'] = df['amount_to'] - df['amount_from']
        df['amount_log'] = np.log1p(df['amount_avg'])
        
        # Temporal features
        df['filing_delay_days'] = (df['filing_date'] - df['transaction_date']).dt.days
        df['transaction_year'] = df['transaction_date'].dt.year
        df['transaction_month'] = df['transaction_date'].dt.month
        df['transaction_quarter'] = df['transaction_date'].dt.quarter
        df['transaction_weekday'] = df['transaction_date'].dt.weekday
        df['is_year_end'] = (df['transaction_month'] >= 11).astype(int)
        
        # Member-specific features
        member_stats = df.groupby('member_name').agg({
            'amount_avg': ['count', 'mean', 'std', 'sum'],
            'filing_delay_days': ['mean', 'std'],
            'symbol': 'nunique'
        }).fillna(0)
        
        # Flatten multi-level columns
        member_stats.columns = ['_'.join(col).strip() for col in member_stats.columns.values]
        member_stats = member_stats.add_prefix('member_')
        
        # Merge member stats back to main dataframe
        df = df.merge(member_stats, left_on='member_name', right_index=True, how='left')
        
        # Party and chamber encoding
        le_party = LabelEncoder()
        le_chamber = LabelEncoder()
        le_transaction = LabelEncoder()
        
        df['party_encoded'] = le_party.fit_transform(df['party'].fillna('Unknown'))
        df['chamber_encoded'] = le_chamber.fit_transform(df['chamber'].fillna('Unknown'))
        df['transaction_type_encoded'] = le_transaction.fit_transform(df['transaction_type'].fillna('Unknown'))
        
        # Store encoders
        self.encoders['party'] = le_party
        self.encoders['chamber'] = le_chamber
        self.encoders['transaction_type'] = le_transaction
        
        # Stock-specific features
        symbol_stats = df.groupby('symbol').agg({
            'amount_avg': ['count', 'mean'],
            'member_name': 'nunique'
        }).fillna(0)
        symbol_stats.columns = ['_'.join(col).strip() for col in symbol_stats.columns.values]
        symbol_stats = symbol_stats.add_prefix('symbol_')
        
        df = df.merge(symbol_stats, left_on='symbol', right_index=True, how='left')
        
        # Risk indicators (target variables)
        df['high_risk'] = (
            (df['amount_avg'] > self.risk_thresholds['high_amount']) |
            (df['filing_delay_days'] > self.risk_thresholds['late_filing'])
        ).astype(int)
        
        df['extreme_risk'] = (
            (df['amount_avg'] > self.risk_thresholds['extreme_amount']) |
            (df['filing_delay_days'] > self.risk_thresholds['extreme_delay'])
        ).astype(int)
        
        # Compliance features
        df['is_compliant'] = (df['filing_delay_days'] <= 45).astype(int)
        df['compliance_score'] = np.clip((45 - df['filing_delay_days']) / 45, 0, 1)
        
        # Portfolio features (per member)
        portfolio_stats = df.groupby('member_name').agg({
            'symbol': 'nunique',
            'amount_avg': lambda x: (x > self.risk_thresholds['high_amount']).sum()
        }).rename(columns={'symbol': 'portfolio_diversity', 'amount_avg': 'large_trades_count'})
        
        df = df.merge(portfolio_stats, left_on='member_name', right_index=True, how='left')
        
        # Network wealth estimate (if available)
        if self.members_df is not None and 'net_worth' in self.members_df.columns:
            wealth_data = self.members_df.set_index('name')['net_worth']
            df['member_net_worth'] = df['member_name'].map(wealth_data).fillna(0)
            df['trade_to_wealth_ratio'] = df['amount_avg'] / (df['member_net_worth'] + 1)
        else:
            df['member_net_worth'] = 0
            df['trade_to_wealth_ratio'] = 0
        
        # Clean and prepare final feature set
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        self.features_df = df
        logger.info(f"✅ Engineered {len(df.columns)} features for {len(df)} trades")
        
        return df
    
    def train_risk_classifier(self) -> Dict[str, Any]:
        """Train models to classify high-risk trades"""
        if self.features_df is None:
            self.engineer_features()
        
        # Select features for risk classification
        feature_cols = [
            'amount_avg', 'amount_range', 'amount_log',
            'filing_delay_days', 'transaction_month', 'transaction_quarter',
            'party_encoded', 'chamber_encoded', 'transaction_type_encoded',
            'member_amount_avg_mean', 'member_amount_avg_std', 'member_amount_avg_count',
            'member_filing_delay_days_mean', 'member_symbol_nunique',
            'symbol_amount_avg_count', 'symbol_member_name_nunique',
            'portfolio_diversity', 'large_trades_count',
            'trade_to_wealth_ratio'
        ]
        
        # Prepare data
        X = self.features_df[feature_cols].fillna(0)
        y = self.features_df['high_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['risk_classifier'] = scaler
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"🤖 Training {name} for risk classification...")
            
            # Use scaled features for linear models
            X_train_model = X_train_scaled if name in ['logistic_regression'] else X_train
            X_test_model = X_test_scaled if name in ['logistic_regression'] else X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_model)
            y_prob = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = model.score(X_test_model, y_test)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'auc_score': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            trained_models[name] = model
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_))
                results[name]['feature_importance'] = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:10]
        
        # Create ensemble model
        ensemble = VotingClassifier([
            ('rf', models['random_forest']),
            ('xgb', models['xgboost']),
            ('gb', models['gradient_boosting'])
        ], voting='soft')
        
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        y_prob_ensemble = ensemble.predict_proba(X_test)[:, 1]
        
        results['ensemble'] = {
            'accuracy': ensemble.score(X_test, y_test),
            'auc_score': roc_auc_score(y_test, y_prob_ensemble),
            'classification_report': classification_report(y_test, y_pred_ensemble, output_dict=True)
        }
        
        trained_models['ensemble'] = ensemble
        
        # Store best model
        best_model_name = max(results.keys(), key=lambda x: results[x].get('auc_score', 0) or 0)
        self.models['risk_classifier'] = trained_models[best_model_name]
        
        logger.info(f"✅ Risk classifier training completed. Best model: {best_model_name}")
        return results
    
    def train_anomaly_detector(self) -> Dict[str, Any]:
        """Train unsupervised anomaly detection models"""
        if self.features_df is None:
            self.engineer_features()
        
        # Select features for anomaly detection
        feature_cols = [
            'amount_avg', 'amount_range', 'filing_delay_days',
            'member_amount_avg_mean', 'member_amount_avg_count',
            'portfolio_diversity', 'trade_to_wealth_ratio'
        ]
        
        X = self.features_df[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['anomaly_detector'] = scaler
        
        # Train anomaly detection models
        models = {
            'isolation_forest': IsolationForest(
                contamination=0.1, random_state=42, n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=0.1, kernel='rbf', gamma='scale'
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"🤖 Training {name} for anomaly detection...")
            
            # Fit model
            model.fit(X_scaled)
            
            # Predict anomalies
            anomaly_labels = model.predict(X_scaled)
            anomaly_scores = model.score_samples(X_scaled) if hasattr(model, 'score_samples') else model.decision_function(X_scaled)
            
            # Calculate metrics
            n_anomalies = (anomaly_labels == -1).sum()
            anomaly_rate = n_anomalies / len(X_scaled) * 100
            
            results[name] = {
                'n_anomalies': int(n_anomalies),
                'anomaly_rate': anomaly_rate,
                'score_mean': float(np.mean(anomaly_scores)),
                'score_std': float(np.std(anomaly_scores))
            }
            
            # Store top anomalies
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            top_anomalies = []
            
            for idx in anomaly_indices[:10]:  # Top 10 anomalies
                trade_data = self.features_df.iloc[idx]
                top_anomalies.append({
                    'member_name': trade_data['member_name'],
                    'symbol': trade_data['symbol'],
                    'amount_avg': trade_data['amount_avg'],
                    'filing_delay_days': trade_data['filing_delay_days'],
                    'anomaly_score': float(anomaly_scores[idx])
                })
            
            results[name]['top_anomalies'] = top_anomalies
            
            # Store model
            self.models[f'anomaly_{name}'] = model
        
        logger.info("✅ Anomaly detection models trained successfully")
        return results
    
    def train_amount_predictor(self) -> Dict[str, Any]:
        """Train regression models to predict trade amounts"""
        if self.features_df is None:
            self.engineer_features()
        
        # Select features for amount prediction
        feature_cols = [
            'filing_delay_days', 'transaction_month', 'transaction_quarter',
            'party_encoded', 'chamber_encoded', 'transaction_type_encoded',
            'member_amount_avg_mean', 'member_amount_avg_count',
            'member_filing_delay_days_mean', 'member_symbol_nunique',
            'portfolio_diversity', 'member_net_worth'
        ]
        
        # Prepare data
        X = self.features_df[feature_cols].fillna(0)
        y = np.log1p(self.features_df['amount_avg'])  # Log transform for better distribution
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {
            'random_forest_reg': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'xgboost_reg': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"🤖 Training {name} for amount prediction...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            
            results[name] = {
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_))
                results[name]['feature_importance'] = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:10]
            
            # Store model
            self.models[f'amount_{name}'] = model
        
        logger.info("✅ Amount prediction models trained successfully")
        return results
    
    def predict_member_risk(self, member_name: str, trade_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict risk for a specific member or trade scenario"""
        if 'risk_classifier' not in self.models:
            logger.error("❌ Risk classifier not trained. Train models first.")
            return {}
        
        # Get member's historical data
        if self.features_df is not None:
            member_trades = self.features_df[
                self.features_df['member_name'] == member_name
            ]
            
            if len(member_trades) == 0:
                logger.warning(f"⚠️ No historical data found for {member_name}")
                return {'error': 'No historical data available'}
            
            # Use latest trade as base or provided trade_data
            if trade_data is None:
                base_trade = member_trades.iloc[-1]
            else:
                # Create feature vector from trade_data
                base_trade = pd.Series(trade_data)
            
            # Prepare feature vector
            feature_cols = [
                'amount_avg', 'amount_range', 'amount_log',
                'filing_delay_days', 'transaction_month', 'transaction_quarter',
                'party_encoded', 'chamber_encoded', 'transaction_type_encoded',
                'member_amount_avg_mean', 'member_amount_avg_std', 'member_amount_avg_count',
                'member_filing_delay_days_mean', 'member_symbol_nunique',
                'symbol_amount_avg_count', 'symbol_member_name_nunique',
                'portfolio_diversity', 'large_trades_count',
                'trade_to_wealth_ratio'
            ]
            
            # Extract features
            features = []
            for col in feature_cols:
                if col in base_trade.index:
                    features.append(base_trade[col])
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            
            # Make prediction
            model = self.models['risk_classifier']
            
            # Scale if needed
            if 'risk_classifier' in self.scalers and hasattr(model, 'predict_proba'):
                if type(model).__name__ == 'LogisticRegression':
                    X = self.scalers['risk_classifier'].transform(X)
            
            risk_prediction = model.predict(X)[0]
            risk_probability = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
            
            # Risk level interpretation
            risk_level = 'LOW'
            if risk_probability:
                if risk_probability > 0.8:
                    risk_level = 'EXTREME'
                elif risk_probability > 0.6:
                    risk_level = 'HIGH'
                elif risk_probability > 0.4:
                    risk_level = 'MEDIUM'
            
            return {
                'member_name': member_name,
                'risk_prediction': int(risk_prediction),
                'risk_probability': float(risk_probability) if risk_probability else None,
                'risk_level': risk_level,
                'historical_trades': len(member_trades),
                'prediction_confidence': float(max(risk_probability, 1-risk_probability)) if risk_probability else None
            }
        
        return {'error': 'Feature data not available'}
    
    def cluster_members(self) -> Dict[str, Any]:
        """Cluster members based on trading behavior"""
        if self.features_df is None:
            self.engineer_features()
        
        # Aggregate member-level features
        member_features = self.features_df.groupby('member_name').agg({
            'amount_avg': ['mean', 'std', 'count'],
            'filing_delay_days': ['mean', 'std'],
            'portfolio_diversity': 'first',
            'party_encoded': 'first',
            'chamber_encoded': 'first',
            'member_net_worth': 'first'
        }).fillna(0)
        
        # Flatten column names
        member_features.columns = ['_'.join(col).strip() for col in member_features.columns.values]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(member_features)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Apply DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Organize results
        results = {
            'kmeans_clusters': {},
            'dbscan_clusters': {},
            'member_assignments': {}
        }
        
        # Process K-Means results
        for cluster_id in np.unique(cluster_labels):
            cluster_members = member_features.index[cluster_labels == cluster_id].tolist()
            cluster_stats = member_features.iloc[cluster_labels == cluster_id].mean()
            
            results['kmeans_clusters'][int(cluster_id)] = {
                'members': cluster_members,
                'member_count': len(cluster_members),
                'avg_trade_amount': float(cluster_stats['amount_avg_mean']),
                'avg_trade_count': float(cluster_stats['amount_avg_count']),
                'avg_filing_delay': float(cluster_stats['filing_delay_days_mean'])
            }
        
        # Process DBSCAN results
        for cluster_id in np.unique(dbscan_labels):
            if cluster_id == -1:  # Outliers
                continue
            cluster_members = member_features.index[dbscan_labels == cluster_id].tolist()
            results['dbscan_clusters'][int(cluster_id)] = {
                'members': cluster_members,
                'member_count': len(cluster_members)
            }
        
        # Outliers from DBSCAN
        outliers = member_features.index[dbscan_labels == -1].tolist()
        results['dbscan_outliers'] = outliers
        
        # Individual member assignments
        for i, member in enumerate(member_features.index):
            results['member_assignments'][member] = {
                'kmeans_cluster': int(cluster_labels[i]),
                'dbscan_cluster': int(dbscan_labels[i]) if dbscan_labels[i] != -1 else 'outlier'
            }
        
        logger.info(f"✅ Member clustering completed: {len(np.unique(cluster_labels))} K-means clusters, "
                   f"{len(np.unique(dbscan_labels[dbscan_labels != -1]))} DBSCAN clusters")
        
        return results
    
    def save_models(self, model_dir: str = 'models/ml_models') -> Dict[str, str]:
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        saved_files = {}
        
        try:
            # Save models
            for model_name, model in self.models.items():
                model_path = os.path.join(model_dir, f'{model_name}.pkl')
                joblib.dump(model, model_path)
                saved_files[model_name] = model_path
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = os.path.join(model_dir, f'{scaler_name}_scaler.pkl')
                joblib.dump(scaler, scaler_path)
                saved_files[f'{scaler_name}_scaler'] = scaler_path
            
            # Save encoders
            for encoder_name, encoder in self.encoders.items():
                encoder_path = os.path.join(model_dir, f'{encoder_name}_encoder.pkl')
                joblib.dump(encoder, encoder_path)
                saved_files[f'{encoder_name}_encoder'] = encoder_path
            
            # Save feature columns
            if self.features_df is not None:
                feature_info = {
                    'columns': list(self.features_df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in self.features_df.dtypes.items()}
                }
                
                feature_path = os.path.join(model_dir, 'feature_info.json')
                with open(feature_path, 'w') as f:
                    json.dump(feature_info, f, indent=2)
                saved_files['feature_info'] = feature_path
            
            logger.info(f"✅ Models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
        
        return saved_files
    
    def load_models(self, model_dir: str = 'models/ml_models') -> bool:
        """Load trained models from disk"""
        try:
            # Load models
            for model_file in os.listdir(model_dir):
                if model_file.endswith('.pkl') and not ('scaler' in model_file or 'encoder' in model_file):
                    model_name = model_file.replace('.pkl', '')
                    model_path = os.path.join(model_dir, model_file)
                    self.models[model_name] = joblib.load(model_path)
            
            # Load scalers
            for scaler_file in os.listdir(model_dir):
                if 'scaler' in scaler_file and scaler_file.endswith('.pkl'):
                    scaler_name = scaler_file.replace('_scaler.pkl', '')
                    scaler_path = os.path.join(model_dir, scaler_file)
                    self.scalers[scaler_name] = joblib.load(scaler_path)
            
            # Load encoders
            for encoder_file in os.listdir(model_dir):
                if 'encoder' in encoder_file and encoder_file.endswith('.pkl'):
                    encoder_name = encoder_file.replace('_encoder.pkl', '')
                    encoder_path = os.path.join(model_dir, encoder_file)
                    self.encoders[encoder_name] = joblib.load(encoder_path)
            
            logger.info(f"✅ Models loaded from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run complete ML training pipeline"""
        logger.info("🚀 Starting comprehensive ML training pipeline...")
        
        # Load and prepare data
        if not self.load_data():
            return {'error': 'Failed to load data'}
        
        self.engineer_features()
        
        # Train all models
        results = {
            'training_completed': datetime.now().isoformat(),
            'data_stats': {
                'total_trades': len(self.trades_df),
                'total_members': self.trades_df['member_name'].nunique(),
                'feature_count': len(self.features_df.columns)
            }
        }
        
        # Risk classifier
        logger.info("🎯 Training risk classification models...")
        results['risk_classifier'] = self.train_risk_classifier()
        
        # Anomaly detector
        logger.info("🔍 Training anomaly detection models...")
        results['anomaly_detector'] = self.train_anomaly_detector()
        
        # Amount predictor
        logger.info("💰 Training amount prediction models...")
        results['amount_predictor'] = self.train_amount_predictor()
        
        # Member clustering
        logger.info("👥 Performing member clustering analysis...")
        results['clustering'] = self.cluster_members()
        
        # Save models
        logger.info("💾 Saving trained models...")
        saved_files = self.save_models()
        results['saved_models'] = saved_files
        
        logger.info("✅ Complete ML training pipeline finished successfully")
        return results

def main():
    """Command-line interface for ML training"""
    engine = TradingPredictionEngine()
    
    # Run full training pipeline
    results = engine.run_full_training_pipeline()
    
    if 'error' not in results:
        print("\n" + "="*60)
        print("🤖 ML PREDICTION ENGINE TRAINING COMPLETED")
        print("="*60)
        
        data_stats = results.get('data_stats', {})
        print(f"📊 Training Data: {data_stats.get('total_trades', 0)} trades from {data_stats.get('total_members', 0)} members")
        print(f"🧬 Features Engineered: {data_stats.get('feature_count', 0)}")
        
        # Risk classifier results
        risk_results = results.get('risk_classifier', {})
        if risk_results:
            print(f"\n🎯 Risk Classifier Performance:")
            for model_name, metrics in risk_results.items():
                if 'accuracy' in metrics:
                    print(f"  • {model_name}: {metrics['accuracy']:.3f} accuracy, {metrics.get('auc_score', 0):.3f} AUC")
        
        # Anomaly detection results
        anomaly_results = results.get('anomaly_detector', {})
        if anomaly_results:
            print(f"\n🔍 Anomaly Detection:")
            for model_name, metrics in anomaly_results.items():
                print(f"  • {model_name}: {metrics['n_anomalies']} anomalies detected ({metrics['anomaly_rate']:.1f}%)")
        
        # Clustering results
        clustering_results = results.get('clustering', {})
        if 'kmeans_clusters' in clustering_results:
            print(f"\n👥 Member Clustering:")
            print(f"  • K-Means: {len(clustering_results['kmeans_clusters'])} clusters identified")
            print(f"  • DBSCAN: {len(clustering_results.get('dbscan_clusters', {}))} clusters, {len(clustering_results.get('dbscan_outliers', []))} outliers")
        
        # Saved models
        saved_models = results.get('saved_models', {})
        if saved_models:
            print(f"\n💾 Models Saved: {len(saved_models)} files")
        
        print("="*60)
        print("✅ ML models ready for deployment!")
    
    else:
        print(f"❌ Training failed: {results.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()