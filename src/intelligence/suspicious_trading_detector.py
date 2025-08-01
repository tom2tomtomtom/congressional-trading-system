#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Phase 2
ML-Based Suspicious Trading Pattern Detection

This module implements machine learning algorithms to detect potentially
suspicious congressional trading patterns based on multiple risk factors.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib
from pathlib import Path

# Machine Learning Imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import DBSCAN, KMeans
# ML libraries will be imported conditionally in functions

# Database and Data Processing
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml

# Statistical Analysis
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuspiciousTradingDetector:
    """
    Advanced ML-based system for detecting suspicious congressional trading patterns.
    Uses multiple algorithms and risk factors to generate suspicion scores.
    """
    
    def __init__(self, config_path: str = "config/database.yml"):
        """Initialize the suspicious trading detector."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.db_config = self.config.get('development', {}).get('database', {})
        
        # ML Models
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Risk factor weights
        self.risk_weights = {
            'timing_score': 0.25,      # Committee hearing/legislation timing
            'amount_score': 0.20,      # Trade amount suspiciousness  
            'frequency_score': 0.15,   # Trading frequency patterns
            'filing_delay_score': 0.15, # Filing delay patterns
            'committee_correlation': 0.10, # Committee jurisdiction correlation
            'market_timing_score': 0.10,   # Market timing accuracy
            'network_centrality': 0.05     # Network analysis position
        }
        
        # Model persistence
        self.model_dir = Path("models/phase2")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load database configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            database=self.db_config.get('name', 'congressional_trading_dev'),
            user=self.db_config.get('user', 'postgres'),
            password=self.db_config.get('password', 'password')
        )
    
    def extract_trading_features(self) -> pd.DataFrame:
        """
        Extract comprehensive features for ML analysis.
        
        Returns:
            DataFrame with trading features for each member-trade combination
        """
        logger.info("Extracting trading features for ML analysis...")
        
        try:
            conn = self._get_db_connection()
            
            # Complex feature extraction query
            query = """
            SELECT 
                t.id as trade_id,
                t.bioguide_id,
                m.full_name,
                m.party,
                m.chamber,
                m.state,
                t.symbol,
                t.transaction_date,
                t.filing_date,
                t.transaction_type,
                t.amount_min,
                t.amount_max,
                t.amount_mid,
                t.filing_delay_days,
                
                -- Member trading frequency (last 365 days)
                (SELECT COUNT(*) FROM trades t2 
                 WHERE t2.bioguide_id = t.bioguide_id 
                 AND t2.transaction_date >= t.transaction_date - INTERVAL '365 days'
                 AND t2.transaction_date <= t.transaction_date) as member_trade_frequency,
                
                -- Symbol trading frequency by member
                (SELECT COUNT(*) FROM trades t3 
                 WHERE t3.bioguide_id = t.bioguide_id 
                 AND t3.symbol = t.symbol
                 AND t3.transaction_date <= t.transaction_date) as member_symbol_frequency,
                
                -- Average trade amount for member
                (SELECT AVG(amount_mid) FROM trades t4 
                 WHERE t4.bioguide_id = t.bioguide_id
                 AND t4.transaction_date <= t.transaction_date) as member_avg_amount,
                
                -- Stock price data for timing analysis
                sp.close_price as stock_price_at_trade,
                
                -- Price change analysis (7 days after trade)
                (SELECT close_price FROM stock_prices sp2 
                 WHERE sp2.symbol = t.symbol 
                 AND sp2.date >= t.transaction_date + INTERVAL '7 days'
                 ORDER BY sp2.date ASC LIMIT 1) as price_7_days_after,
                
                -- Price change analysis (30 days after trade)
                (SELECT close_price FROM stock_prices sp3 
                 WHERE sp3.symbol = t.symbol 
                 AND sp3.date >= t.transaction_date + INTERVAL '30 days'
                 ORDER BY sp3.date ASC LIMIT 1) as price_30_days_after,
                
                -- Committee assignments (simplified)
                (SELECT COUNT(*) FROM committee_memberships cm 
                 JOIN committees c ON cm.committee_id = c.id
                 WHERE cm.bioguide_id = t.bioguide_id
                 AND c.name ILIKE '%financial%') as financial_committee_member,
                
                (SELECT COUNT(*) FROM committee_memberships cm 
                 JOIN committees c ON cm.committee_id = c.id
                 WHERE cm.bioguide_id = t.bioguide_id
                 AND c.name ILIKE '%energy%') as energy_committee_member,
                
                (SELECT COUNT(*) FROM committee_memberships cm 
                 JOIN committees c ON cm.committee_id = c.id
                 WHERE cm.bioguide_id = t.bioguide_id
                 AND c.name ILIKE '%technology%') as tech_committee_member
                
            FROM trades t
            JOIN members m ON t.bioguide_id = m.bioguide_id
            LEFT JOIN stock_prices sp ON t.symbol = sp.symbol 
                AND sp.date = (
                    SELECT MAX(date) FROM stock_prices sp_inner 
                    WHERE sp_inner.symbol = t.symbol 
                    AND sp_inner.date <= t.transaction_date
                )
            WHERE t.amount_mid IS NOT NULL
            ORDER BY t.transaction_date DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Extracted {len(df)} trading records with features")
            return df
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return pd.DataFrame()
    
    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate individual risk scores for each factor.
        
        Args:
            df: DataFrame with trading features
            
        Returns:
            DataFrame with calculated risk scores
        """
        logger.info("Calculating individual risk scores...")
        
        # Initialize risk scores
        df = df.copy()
        
        # 1. Timing Score (based on filing delays)
        df['timing_score'] = np.where(
            df['filing_delay_days'] > 45, 10.0,  # Late filing = max risk
            np.where(
                df['filing_delay_days'] > 30, 8.0,
                np.where(
                    df['filing_delay_days'] > 14, 6.0,
                    np.where(df['filing_delay_days'] > 7, 4.0, 2.0)
                )
            )
        )
        
        # 2. Amount Score (unusually large trades)
        amount_percentiles = df['amount_mid'].quantile([0.75, 0.90, 0.95, 0.99])
        df['amount_score'] = np.where(
            df['amount_mid'] >= amount_percentiles[0.99], 10.0,
            np.where(
                df['amount_mid'] >= amount_percentiles[0.95], 8.0,
                np.where(
                    df['amount_mid'] >= amount_percentiles[0.90], 6.0,
                    np.where(df['amount_mid'] >= amount_percentiles[0.75], 4.0, 2.0)
                )
            )
        )
        
        # 3. Frequency Score (unusual trading patterns)
        freq_percentiles = df['member_trade_frequency'].quantile([0.75, 0.90, 0.95, 0.99])
        df['frequency_score'] = np.where(
            df['member_trade_frequency'] >= freq_percentiles[0.99], 10.0,
            np.where(
                df['member_trade_frequency'] >= freq_percentiles[0.95], 8.0,
                np.where(
                    df['member_trade_frequency'] >= freq_percentiles[0.90], 6.0,
                    np.where(df['member_trade_frequency'] >= freq_percentiles[0.75], 4.0, 2.0)
                )
            )
        )
        
        # 4. Filing Delay Score (normalized)
        max_delay = df['filing_delay_days'].max()
        df['filing_delay_score'] = (df['filing_delay_days'] / max_delay) * 10.0
        
        # 5. Market Timing Score (price performance after trade)
        df['price_change_7d'] = (df['price_7_days_after'] - df['stock_price_at_trade']) / df['stock_price_at_trade'] * 100
        df['price_change_30d'] = (df['price_30_days_after'] - df['stock_price_at_trade']) / df['stock_price_at_trade'] * 100
        
        # Market timing score based on accuracy of trades
        df['market_timing_score'] = np.where(
            df['transaction_type'] == 'Purchase',
            np.where(df['price_change_30d'] > 10, 8.0,
                    np.where(df['price_change_30d'] > 5, 6.0, 
                            np.where(df['price_change_30d'] > 0, 4.0, 2.0))),
            np.where(df['transaction_type'] == 'Sale',
                    np.where(df['price_change_30d'] < -10, 8.0,
                            np.where(df['price_change_30d'] < -5, 6.0,
                                    np.where(df['price_change_30d'] < 0, 4.0, 2.0))), 3.0)
        )
        
        # 6. Committee Correlation Score
        df['committee_correlation'] = (
            df['financial_committee_member'] * 3.0 +
            df['energy_committee_member'] * 2.0 +
            df['tech_committee_member'] * 2.0
        ).clip(0, 10)
        
        # 7. Network Centrality Score (simplified - based on trading frequency)
        df['network_centrality'] = (df['member_trade_frequency'] / df['member_trade_frequency'].max()) * 10.0
        
        # Handle NaN values
        score_columns = ['timing_score', 'amount_score', 'frequency_score', 'filing_delay_score', 
                        'market_timing_score', 'committee_correlation', 'network_centrality']
        for col in score_columns:
            df[col] = df[col].fillna(df[col].median())
        
        logger.info("Risk scores calculated successfully")
        return df
    
    def calculate_composite_suspicion_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite suspicion score using weighted factors.
        
        Args:
            df: DataFrame with individual risk scores
            
        Returns:
            DataFrame with composite suspicion scores
        """
        logger.info("Calculating composite suspicion scores...")
        
        # Calculate weighted composite score
        df['suspicion_score'] = (
            df['timing_score'] * self.risk_weights['timing_score'] +
            df['amount_score'] * self.risk_weights['amount_score'] +
            df['frequency_score'] * self.risk_weights['frequency_score'] +
            df['filing_delay_score'] * self.risk_weights['filing_delay_score'] +
            df['committee_correlation'] * self.risk_weights['committee_correlation'] +
            df['market_timing_score'] * self.risk_weights['market_timing_score'] +
            df['network_centrality'] * self.risk_weights['network_centrality']
        )
        
        # Normalize to 0-10 scale
        df['suspicion_score'] = df['suspicion_score'].clip(0, 10)
        
        # Add risk categories
        df['risk_category'] = pd.cut(
            df['suspicion_score'],
            bins=[0, 3, 6, 8, 10],
            labels=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
            include_lowest=True
        )
        
        logger.info("Composite suspicion scores calculated")
        return df
    
    def train_anomaly_detection_models(self, df: pd.DataFrame) -> Dict:
        """
        Train multiple anomaly detection models.
        
        Args:
            df: DataFrame with features and scores
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training anomaly detection models...")
        
        # Prepare features for ML
        feature_columns = [
            'amount_mid', 'filing_delay_days', 'member_trade_frequency',
            'member_symbol_frequency', 'member_avg_amount', 'price_change_7d',
            'price_change_30d', 'financial_committee_member', 'energy_committee_member',
            'tech_committee_member'
        ]
        
        # Handle missing values more robustly
        X = df[feature_columns].copy()
        
        # Fill NaN with median for numeric columns, 0 for any remaining
        for col in feature_columns:
            if X[col].dtype in ['float64', 'int64']:
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(0)
        
        # Final check - replace any remaining NaN/inf with 0
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler
        self.scalers['anomaly_scaler'] = scaler
        
        models = {}
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        iso_forest.fit(X_scaled)
        models['isolation_forest'] = iso_forest
        
        # 2. DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        models['dbscan'] = dbscan
        
        # 3. One-Class SVM (using Isolation Forest as substitute)
        from sklearn.svm import OneClassSVM
        one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        one_class_svm.fit(X_scaled)
        models['one_class_svm'] = one_class_svm
        
        # 4. Try XGBoost, fall back to Random Forest
        y_synthetic = (df['suspicion_score'] > df['suspicion_score'].quantile(0.9)).astype(int)
        
        if len(np.unique(y_synthetic)) > 1:  # Ensure we have both classes
            # Use Random Forest (XGBoost alternative)
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            rf_model.fit(X_scaled, y_synthetic)
            models['random_forest'] = rf_model
        
        # Store models
        self.models.update(models)
        
        logger.info(f"Trained {len(models)} anomaly detection models")
        return models
    
    def predict_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use trained models to predict anomalies.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with anomaly predictions
        """
        logger.info("Predicting anomalies using trained models...")
        
        # Prepare features
        feature_columns = [
            'amount_mid', 'filing_delay_days', 'member_trade_frequency',
            'member_symbol_frequency', 'member_avg_amount', 'price_change_7d',
            'price_change_30d', 'financial_committee_member', 'energy_committee_member',
            'tech_committee_member'
        ]
        
        # Handle missing values robustly
        X = df[feature_columns].copy()
        
        # Fill NaN with median for numeric columns, 0 for any remaining
        for col in feature_columns:
            if X[col].dtype in ['float64', 'int64']:
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(0)
        
        # Final check - replace any remaining NaN/inf with 0
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        X_scaled = self.scalers['anomaly_scaler'].transform(X)
        
        # Get predictions from each model
        if 'isolation_forest' in self.models:
            df['iso_forest_anomaly'] = self.models['isolation_forest'].predict(X_scaled)
            df['iso_forest_score'] = self.models['isolation_forest'].decision_function(X_scaled)
        
        if 'one_class_svm' in self.models:
            df['svm_anomaly'] = self.models['one_class_svm'].predict(X_scaled)
            df['svm_score'] = self.models['one_class_svm'].decision_function(X_scaled)
        
        if 'xgboost' in self.models:
            df['xgb_anomaly_prob'] = self.models['xgboost'].predict_proba(X_scaled)[:, 1]
            df['xgb_anomaly'] = (df['xgb_anomaly_prob'] > 0.5).astype(int)
        elif 'random_forest' in self.models:
            df['rf_anomaly_prob'] = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
            df['rf_anomaly'] = (df['rf_anomaly_prob'] > 0.5).astype(int)
        
        # Ensemble anomaly score
        anomaly_scores = []
        if 'iso_forest_score' in df.columns:
            anomaly_scores.append(df['iso_forest_score'])
        if 'svm_score' in df.columns:
            anomaly_scores.append(df['svm_score'])
        if 'xgb_anomaly_prob' in df.columns:
            anomaly_scores.append(df['xgb_anomaly_prob'])
        elif 'rf_anomaly_prob' in df.columns:
            anomaly_scores.append(df['rf_anomaly_prob'])
        
        if anomaly_scores:
            # Normalize and average anomaly scores
            normalized_scores = []
            for score in anomaly_scores:
                normalized = (score - score.min()) / (score.max() - score.min())
                normalized_scores.append(normalized)
            
            df['ml_anomaly_score'] = np.mean(normalized_scores, axis=0) * 10
        else:
            df['ml_anomaly_score'] = 5.0  # Default moderate score
        
        logger.info("Anomaly predictions completed")
        return df
    
    def generate_alerts(self, df: pd.DataFrame, threshold: float = 7.0) -> pd.DataFrame:
        """
        Generate high-priority alerts for suspicious trades.
        
        Args:
            df: DataFrame with suspicion scores
            threshold: Minimum suspicion score for alerts
            
        Returns:
            DataFrame with high-priority alerts
        """
        logger.info(f"Generating alerts for trades with suspicion score > {threshold}")
        
        # Filter high-suspicion trades
        alerts = df[df['suspicion_score'] >= threshold].copy()
        
        # Sort by suspicion score (highest first)
        alerts = alerts.sort_values('suspicion_score', ascending=False)
        
        # Add alert metadata
        alerts['alert_generated'] = datetime.now()
        alerts['alert_priority'] = pd.cut(
            alerts['suspicion_score'],
            bins=[threshold, 8, 9, 10],
            labels=['HIGH', 'CRITICAL', 'EXTREME'],
            include_lowest=True
        )
        
        # Generate alert descriptions
        alerts['alert_reason'] = alerts.apply(self._generate_alert_reason, axis=1)
        
        logger.info(f"Generated {len(alerts)} high-priority alerts")
        return alerts
    
    def _generate_alert_reason(self, row) -> str:
        """Generate human-readable alert reason."""
        reasons = []
        
        if row['timing_score'] > 7:
            reasons.append(f"Late filing ({row['filing_delay_days']} days)")
        
        if row['amount_score'] > 7:
            reasons.append(f"Large trade amount (${row['amount_mid']:,.0f})")
        
        if row['frequency_score'] > 7:
            reasons.append(f"High trading frequency ({row['member_trade_frequency']} trades)")
        
        if row['market_timing_score'] > 7:
            reasons.append("Exceptional market timing")
        
        if row['committee_correlation'] > 5:
            reasons.append("Committee jurisdiction correlation")
        
        if hasattr(row, 'ml_anomaly_score') and row['ml_anomaly_score'] > 7:
            reasons.append("ML anomaly detection")
        
        return " • ".join(reasons) if reasons else "Multiple risk factors"
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info("Saving trained models...")
        
        # Save models
        for name, model in self.models.items():
            model_path = self.model_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f"{name}.pkl"
            joblib.dump(scaler, scaler_path)
        
        # Save risk weights
        weights_path = self.model_dir / "risk_weights.pkl"
        with open(weights_path, 'wb') as f:
            pickle.dump(self.risk_weights, f)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models from disk."""
        logger.info("Loading trained models...")
        
        try:
            # Load models
            for model_file in self.model_dir.glob("*_model.pkl"):
                name = model_file.stem.replace('_model', '')
                self.models[name] = joblib.load(model_file)
            
            # Load scalers
            for scaler_file in self.model_dir.glob("*_scaler.pkl"):
                name = scaler_file.stem
                self.scalers[name] = joblib.load(scaler_file)
            
            # Load risk weights
            weights_path = self.model_dir / "risk_weights.pkl"
            if weights_path.exists():
                with open(weights_path, 'rb') as f:
                    self.risk_weights = pickle.load(f)
            
            logger.info(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers")
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete suspicious trading analysis.
        
        Returns:
            Tuple of (full_analysis_df, alerts_df)
        """
        logger.info("Running full suspicious trading analysis...")
        
        # Extract features
        df = self.extract_trading_features()
        if df.empty:
            logger.error("No trading data available for analysis")
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate risk scores
        df = self.calculate_risk_scores(df)
        
        # Calculate composite suspicion scores
        df = self.calculate_composite_suspicion_score(df)
        
        # Train and apply ML models
        self.train_anomaly_detection_models(df)
        df = self.predict_anomalies(df)
        
        # Generate alerts
        alerts = self.generate_alerts(df, threshold=7.0)
        
        # Save models
        self.save_models()
        
        logger.info("Full suspicious trading analysis completed")
        return df, alerts

def main():
    """Main execution function."""
    logger.info("Starting Congressional Trading Suspicious Pattern Detection...")
    
    detector = SuspiciousTradingDetector()
    
    # Run full analysis
    analysis_df, alerts_df = detector.run_full_analysis()
    
    if not analysis_df.empty:
        # Display summary statistics
        logger.info("=== ANALYSIS SUMMARY ===")
        logger.info(f"Total trades analyzed: {len(analysis_df)}")
        logger.info(f"Average suspicion score: {analysis_df['suspicion_score'].mean():.2f}")
        logger.info(f"High-risk trades (>7.0): {len(analysis_df[analysis_df['suspicion_score'] > 7.0])}")
        logger.info(f"Extreme-risk trades (>9.0): {len(analysis_df[analysis_df['suspicion_score'] > 9.0])}")
        
        # Display top suspicious trades
        logger.info("\n=== TOP 5 SUSPICIOUS TRADES ===")
        for _, trade in analysis_df.nlargest(5, 'suspicion_score').iterrows():
            logger.info(f"{trade['full_name']} ({trade['party']}-{trade['state']}) - "
                       f"{trade['symbol']} {trade['transaction_type']} ${trade['amount_mid']:,.0f} - "
                       f"Score: {trade['suspicion_score']:.1f}")
        
        # Display alerts
        if not alerts_df.empty:
            logger.info(f"\n=== GENERATED {len(alerts_df)} ALERTS ===")
            for _, alert in alerts_df.head(3).iterrows():
                logger.info(f"🚨 {alert['full_name']}: {alert['alert_reason']} "
                           f"(Score: {alert['suspicion_score']:.1f})")
    
    logger.info("Suspicious pattern detection completed successfully!")

if __name__ == "__main__":
    main()