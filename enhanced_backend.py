#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Enhanced Backend
Comprehensive data integration system with ML predictions and statistical analysis
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any
import time
from collections import defaultdict

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder='src/dashboard/templates',
    static_folder='src/dashboard/static'
)
CORS(app)

# Add current directory to Python path
sys.path.append('.')

# Global cache for data and models
_data_cache = {}
_model_cache = {}
_cache_timestamps = {}

class DataProcessor:
    """Comprehensive data processing for congressional trading analysis"""
    
    def __init__(self):
        self.members_df = None
        self.trades_df = None
        self.analysis_results = {}
        
    def load_data(self) -> bool:
        """Load congressional members and trading data"""
        try:
            # Load members data
            members_path = 'src/data/congressional_members_full.json'
            if os.path.exists(members_path):
                with open(members_path, 'r') as f:
                    members_data = json.load(f)
                self.members_df = pd.DataFrame(members_data)
                logger.info(f"Loaded {len(self.members_df)} congressional members")
            
            # Load trades data
            trades_path = 'src/data/congressional_trades_full.json'
            if os.path.exists(trades_path):
                with open(trades_path, 'r') as f:
                    trades_data = json.load(f)
                self.trades_df = pd.DataFrame(trades_data)
                
                # Convert date columns
                if 'transaction_date' in self.trades_df.columns:
                    self.trades_df['transaction_date'] = pd.to_datetime(
                        self.trades_df['transaction_date']
                    )
                if 'filing_date' in self.trades_df.columns:
                    self.trades_df['filing_date'] = pd.to_datetime(
                        self.trades_df['filing_date']
                    )
                
                # Calculate filing delays
                if 'transaction_date' in self.trades_df.columns and 'filing_date' in self.trades_df.columns:
                    self.trades_df['filing_delay_days'] = (
                        self.trades_df['filing_date'] - self.trades_df['transaction_date']
                    ).dt.days
                
                logger.info(f"Loaded {len(self.trades_df)} congressional trades")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def calculate_risk_scores(self) -> Dict[str, Any]:
        """Calculate comprehensive risk scores for members"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return {}
        
        risk_scores = {}
        
        for member_name in self.trades_df['member_name'].unique():
            member_trades = self.trades_df[self.trades_df['member_name'] == member_name]
            
            # Base risk factors
            trade_count = len(member_trades)
            avg_amount = member_trades[['amount_from', 'amount_to']].mean().mean()
            avg_delay = member_trades['filing_delay_days'].mean() if 'filing_delay_days' in member_trades.columns else 30
            
            # Risk score calculation
            risk_score = 0
            
            # Trade frequency (more trades = higher risk)
            if trade_count > 10:
                risk_score += 3
            elif trade_count > 5:
                risk_score += 2
            elif trade_count > 2:
                risk_score += 1
            
            # Trade size (larger trades = higher risk)
            if avg_amount > 1000000:
                risk_score += 3
            elif avg_amount > 250000:
                risk_score += 2
            elif avg_amount > 50000:
                risk_score += 1
            
            # Filing delays (late filings = higher risk)
            if avg_delay > 60:
                risk_score += 3
            elif avg_delay > 45:
                risk_score += 2
            elif avg_delay > 30:
                risk_score += 1
            
            # Additional risk factors
            unique_symbols = member_trades['symbol'].nunique()
            if unique_symbols > 20:
                risk_score += 2
            elif unique_symbols > 10:
                risk_score += 1
            
            risk_scores[member_name] = {
                'risk_score': min(risk_score, 10),  # Cap at 10
                'trade_count': trade_count,
                'avg_amount': avg_amount,
                'avg_delay': avg_delay,
                'unique_symbols': unique_symbols
            }
        
        return risk_scores
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous trading patterns using machine learning"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return []
        
        try:
            # Prepare features for anomaly detection
            features_df = self.trades_df.copy()
            
            # Create numerical features
            features_df['amount_avg'] = (features_df['amount_from'] + features_df['amount_to']) / 2
            features_df['amount_range'] = features_df['amount_to'] - features_df['amount_from']
            
            # Encode categorical variables
            features_df['party_encoded'] = features_df['party'].map({'D': 0, 'R': 1, 'I': 2})
            features_df['chamber_encoded'] = features_df['chamber'].map({'House': 0, 'Senate': 1})
            features_df['transaction_type_encoded'] = features_df['transaction_type'].map({
                'Purchase': 0, 'Sale': 1, 'Exchange': 2
            })
            
            # Select numeric features for anomaly detection
            feature_columns = [
                'amount_avg', 'amount_range', 'filing_delay_days',
                'party_encoded', 'chamber_encoded', 'transaction_type_encoded'
            ]
            
            # Fill NaN values
            for col in feature_columns:
                if col in features_df.columns:
                    features_df[col] = features_df[col].fillna(features_df[col].median())
            
            # Prepare feature matrix
            X = features_df[feature_columns].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            
            # Extract anomalous trades
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            anomalies = []
            
            for idx in anomaly_indices:
                trade = self.trades_df.iloc[idx]
                anomaly_score = iso_forest.score_samples(X_scaled[idx].reshape(1, -1))[0]
                
                anomalies.append({
                    'id': trade.get('id', idx),
                    'member_name': trade['member_name'],
                    'symbol': trade['symbol'],
                    'transaction_date': trade['transaction_date'].strftime('%Y-%m-%d'),
                    'amount_from': trade['amount_from'],
                    'amount_to': trade['amount_to'],
                    'anomaly_score': float(anomaly_score),
                    'filing_delay_days': trade.get('filing_delay_days', 0),
                    'reason': 'Statistical anomaly detected by ML model'
                })
            
            # Sort by anomaly score (most anomalous first)
            anomalies.sort(key=lambda x: x['anomaly_score'])
            
            logger.info(f"Detected {len(anomalies)} anomalous trades")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics"""
        stats = {}
        
        if self.trades_df is not None and len(self.trades_df) > 0:
            # Basic statistics
            stats['total_trades'] = len(self.trades_df)
            stats['unique_members'] = self.trades_df['member_name'].nunique()
            stats['total_volume'] = self.trades_df[['amount_from', 'amount_to']].mean(axis=1).sum()
            stats['avg_trade_size'] = self.trades_df[['amount_from', 'amount_to']].mean(axis=1).mean()
            
            # Party breakdown
            party_stats = self.trades_df['party'].value_counts().to_dict()
            stats['party_breakdown'] = party_stats
            
            # Chamber breakdown
            chamber_stats = self.trades_df['chamber'].value_counts().to_dict()
            stats['chamber_breakdown'] = chamber_stats
            
            # Filing delay statistics
            if 'filing_delay_days' in self.trades_df.columns:
                stats['avg_filing_delay'] = self.trades_df['filing_delay_days'].mean()
                stats['late_filings_count'] = len(self.trades_df[self.trades_df['filing_delay_days'] > 45])
                stats['compliance_rate'] = (1 - stats['late_filings_count'] / stats['total_trades']) * 100
            
            # Top trading symbols
            top_symbols = self.trades_df['symbol'].value_counts().head(10).to_dict()
            stats['top_symbols'] = top_symbols
            
            # Monthly trading volume
            if 'transaction_date' in self.trades_df.columns:
                monthly_volume = self.trades_df.groupby(
                    self.trades_df['transaction_date'].dt.to_period('M')
                )[['amount_from', 'amount_to']].mean().mean(axis=1).to_dict()
                
                # Convert period index to strings
                stats['monthly_volume'] = {
                    str(period): float(volume) for period, volume in monthly_volume.items()
                }
        
        if self.members_df is not None and len(self.members_df) > 0:
            stats['total_members_tracked'] = len(self.members_df)
            
            # Net worth statistics
            if 'net_worth' in self.members_df.columns:
                stats['avg_net_worth'] = self.members_df['net_worth'].mean()
                stats['median_net_worth'] = self.members_df['net_worth'].median()
        
        return stats

# Initialize data processor
data_processor = DataProcessor()

def cache_response(cache_key: str, ttl_minutes: int = 30):
    """Decorator to cache API responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            global _data_cache, _cache_timestamps
            
            # Check if cached data exists and is still valid
            if (cache_key in _data_cache and 
                cache_key in _cache_timestamps and
                datetime.now() - _cache_timestamps[cache_key] < timedelta(minutes=ttl_minutes)):
                logger.info(f"Returning cached data for {cache_key}")
                return _data_cache[cache_key]
            
            # Generate fresh data
            result = f(*args, **kwargs)
            
            # Cache the result
            _data_cache[cache_key] = result
            _cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Cached fresh data for {cache_key}")
            return result
        
        return decorated_function
    return decorator

@app.route('/api/v1/stats')
@cache_response('stats', 15)  # Cache for 15 minutes
def api_v1_stats():
    """Enhanced statistics API with comprehensive metrics"""
    try:
        # Ensure data is loaded
        if not data_processor.load_data():
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        # Calculate comprehensive statistics
        stats = data_processor.calculate_statistics()
        risk_scores = data_processor.calculate_risk_scores()
        
        # Add risk score statistics
        if risk_scores:
            risk_values = [member['risk_score'] for member in risk_scores.values()]
            stats['average_risk_score'] = np.mean(risk_values)
            stats['high_risk_members'] = len([r for r in risk_values if r >= 7])
            stats['medium_risk_members'] = len([r for r in risk_values if 4 <= r < 7])
            stats['low_risk_members'] = len([r for r in risk_values if r < 4])
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'data_quality': {
                'members_loaded': len(data_processor.members_df) if data_processor.members_df is not None else 0,
                'trades_loaded': len(data_processor.trades_df) if data_processor.trades_df is not None else 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in /api/v1/stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/v1/members')
@cache_response('members', 30)
def api_v1_members():
    """Enhanced members API with risk scores and detailed analysis"""
    try:
        if not data_processor.load_data():
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        party_filter = request.args.get('party')
        chamber_filter = request.args.get('chamber')
        risk_threshold = request.args.get('min_risk', 0, type=float)
        
        # Calculate risk scores
        risk_scores = data_processor.calculate_risk_scores()
        
        # Prepare member data
        members_list = []
        if data_processor.members_df is not None:
            members_data = data_processor.members_df.to_dict('records')
            
            for member in members_data:
                member_name = member.get('name', '')
                risk_data = risk_scores.get(member_name, {})
                
                # Apply filters
                if party_filter and member.get('party') != party_filter:
                    continue
                if chamber_filter and member.get('chamber') != chamber_filter:
                    continue
                if risk_data.get('risk_score', 0) < risk_threshold:
                    continue
                
                # Enhance member data with risk information
                enhanced_member = {
                    **member,
                    'risk_score': risk_data.get('risk_score', 0),
                    'trade_count': risk_data.get('trade_count', 0),
                    'avg_trade_amount': risk_data.get('avg_amount', 0),
                    'avg_filing_delay': risk_data.get('avg_delay', 0),
                    'unique_symbols_traded': risk_data.get('unique_symbols', 0),
                    'risk_level': (
                        'EXTREME' if risk_data.get('risk_score', 0) >= 9 else
                        'HIGH' if risk_data.get('risk_score', 0) >= 7 else
                        'MEDIUM' if risk_data.get('risk_score', 0) >= 4 else
                        'LOW'
                    )
                }
                
                members_list.append(enhanced_member)
        
        # Sort by risk score descending
        members_list.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        # Apply pagination
        total_count = len(members_list)
        paginated_members = members_list[offset:offset + limit]
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'members': paginated_members,
                'pagination': {
                    'total_count': total_count,
                    'returned_count': len(paginated_members),
                    'offset': offset,
                    'limit': limit,
                    'has_more': offset + limit < total_count
                },
                'filters_applied': {
                    'party': party_filter,
                    'chamber': chamber_filter,
                    'min_risk': risk_threshold
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in /api/v1/members: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/trades')
@cache_response('trades', 20)
def api_v1_trades():
    """Enhanced trades API with filtering and analysis"""
    try:
        if not data_processor.load_data():
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        member_name = request.args.get('member')
        symbol = request.args.get('symbol')
        min_amount = request.args.get('min_amount', 0, type=float)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        trades_data = []
        if data_processor.trades_df is not None:
            df = data_processor.trades_df.copy()
            
            # Apply filters
            if member_name:
                df = df[df['member_name'].str.contains(member_name, case=False, na=False)]
            if symbol:
                df = df[df['symbol'] == symbol.upper()]
            if min_amount > 0:
                df = df[df[['amount_from', 'amount_to']].mean(axis=1) >= min_amount]
            
            if start_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df['transaction_date'] >= start_dt]
                except:
                    pass
            
            if end_date:
                try:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df['transaction_date'] <= end_dt]
                except:
                    pass
            
            # Sort by transaction date descending
            df = df.sort_values('transaction_date', ascending=False)
            
            # Apply pagination
            total_count = len(df)
            paginated_df = df.iloc[offset:offset + limit]
            
            # Convert to list of dictionaries
            trades_data = []
            for _, trade in paginated_df.iterrows():
                trade_dict = trade.to_dict()
                
                # Convert timestamps to strings
                if 'transaction_date' in trade_dict and pd.notna(trade_dict['transaction_date']):
                    trade_dict['transaction_date'] = trade_dict['transaction_date'].strftime('%Y-%m-%d')
                if 'filing_date' in trade_dict and pd.notna(trade_dict['filing_date']):
                    trade_dict['filing_date'] = trade_dict['filing_date'].strftime('%Y-%m-%d')
                
                # Add computed fields
                trade_dict['avg_amount'] = (trade_dict.get('amount_from', 0) + trade_dict.get('amount_to', 0)) / 2
                trade_dict['amount_range'] = trade_dict.get('amount_to', 0) - trade_dict.get('amount_from', 0)
                
                # Compliance status
                filing_delay = trade_dict.get('filing_delay_days', 0)
                trade_dict['compliance_status'] = (
                    'LATE' if filing_delay > 45 else
                    'WARNING' if filing_delay > 30 else
                    'COMPLIANT'
                )
                
                trades_data.append(trade_dict)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'trades': trades_data,
                'pagination': {
                    'total_count': total_count if 'total_count' in locals() else 0,
                    'returned_count': len(trades_data),
                    'offset': offset,
                    'limit': limit,
                    'has_more': offset + limit < (total_count if 'total_count' in locals() else 0)
                },
                'filters_applied': {
                    'member': member_name,
                    'symbol': symbol,
                    'min_amount': min_amount,
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in /api/v1/trades: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/anomalies')
@cache_response('anomalies', 60)  # Cache for 1 hour
def api_v1_anomalies():
    """API endpoint for anomaly detection results"""
    try:
        if not data_processor.load_data():
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        anomalies = data_processor.detect_anomalies()
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'anomalies': anomalies,
                'total_anomalies': len(anomalies),
                'detection_method': 'Isolation Forest ML Algorithm',
                'model_parameters': {
                    'contamination': 0.1,
                    'features_used': [
                        'amount_avg', 'amount_range', 'filing_delay_days',
                        'party_encoded', 'chamber_encoded', 'transaction_type_encoded'
                    ]
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in /api/v1/anomalies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/export/<format>')
def api_v1_export(format):
    """Export data in various formats (CSV, JSON)"""
    try:
        if not data_processor.load_data():
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        export_type = request.args.get('type', 'members')  # members, trades, or analysis
        
        if format.lower() not in ['csv', 'json']:
            return jsonify({
                'success': False,
                'error': 'Unsupported format. Use csv or json.'
            }), 400
        
        # Prepare data based on type
        if export_type == 'members' and data_processor.members_df is not None:
            df = data_processor.members_df.copy()
            filename = f'congressional_members_{datetime.now().strftime("%Y%m%d")}'
        elif export_type == 'trades' and data_processor.trades_df is not None:
            df = data_processor.trades_df.copy()
            # Convert datetime columns to strings for export
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            filename = f'congressional_trades_{datetime.now().strftime("%Y%m%d")}'
        elif export_type == 'analysis':
            # Create analysis summary
            stats = data_processor.calculate_statistics()
            risk_scores = data_processor.calculate_risk_scores()
            
            analysis_data = []
            for member_name, risk_data in risk_scores.items():
                analysis_data.append({
                    'member_name': member_name,
                    'risk_score': risk_data['risk_score'],
                    'trade_count': risk_data['trade_count'],
                    'avg_amount': risk_data['avg_amount'],
                    'avg_delay': risk_data['avg_delay'],
                    'unique_symbols': risk_data['unique_symbols']
                })
            
            df = pd.DataFrame(analysis_data)
            filename = f'congressional_analysis_{datetime.now().strftime("%Y%m%d")}'
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid export type or no data available'
            }), 400
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports', exist_ok=True)
        
        if format.lower() == 'csv':
            file_path = f'exports/{filename}.csv'
            df.to_csv(file_path, index=False)
        else:  # json
            file_path = f'exports/{filename}.json'
            df.to_json(file_path, orient='records', indent=2)
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f'{filename}.{format.lower()}'
        )
        
    except Exception as e:
        logger.error(f"Error in /api/v1/export: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/predictions')
@cache_response('predictions', 120)  # Cache for 2 hours
def api_v1_predictions():
    """ML predictions for trading patterns"""
    try:
        if not data_processor.load_data():
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        if data_processor.trades_df is None or len(data_processor.trades_df) < 50:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for predictions'
            }), 400
        
        # Prepare features for prediction model
        df = data_processor.trades_df.copy()
        
        # Create features
        df['amount_avg'] = (df['amount_from'] + df['amount_to']) / 2
        df['amount_range'] = df['amount_to'] - df['amount_from']
        df['party_encoded'] = df['party'].map({'D': 0, 'R': 1, 'I': 2})
        df['chamber_encoded'] = df['chamber'].map({'House': 0, 'Senate': 1})
        df['month'] = df['transaction_date'].dt.month
        
        # Create target variable (high-risk trade: 1, low-risk: 0)
        df['high_risk'] = (
            (df['amount_avg'] > 100000) | 
            (df['filing_delay_days'] > 45)
        ).astype(int)
        
        # Prepare feature matrix
        feature_columns = [
            'amount_avg', 'amount_range', 'filing_delay_days',
            'party_encoded', 'chamber_encoded', 'month'
        ]
        
        # Fill NaN values
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        X = df[feature_columns].values
        y = df['high_risk'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Generate predictions for future scenarios
        future_predictions = []
        
        # Predict for each member based on their historical patterns
        for member_name in df['member_name'].unique()[:10]:  # Limit to top 10 for performance
            member_data = df[df['member_name'] == member_name].iloc[-1:]  # Last trade
            
            if len(member_data) > 0:
                member_features = member_data[feature_columns].values
                risk_probability = model.predict_proba(member_features)[0][1]  # Probability of high risk
                
                future_predictions.append({
                    'member_name': member_name,
                    'risk_probability': float(risk_probability),
                    'prediction': 'HIGH_RISK' if risk_probability > 0.6 else 'MODERATE' if risk_probability > 0.3 else 'LOW_RISK',
                    'confidence': float(max(risk_probability, 1 - risk_probability))
                })
        
        # Sort by risk probability
        future_predictions.sort(key=lambda x: x['risk_probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'model_performance': {
                    'accuracy': float(np.mean(y_pred == y_test)),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_importance': feature_importance
                },
                'future_predictions': future_predictions,
                'model_info': {
                    'algorithm': 'Random Forest Classifier',
                    'features_used': feature_columns,
                    'prediction_horizon': '30 days',
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in /api/v1/predictions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/refresh')
def api_v1_refresh():
    """Force refresh of all cached data"""
    global _data_cache, _cache_timestamps
    
    # Clear all caches
    _data_cache.clear()
    _cache_timestamps.clear()
    
    # Reload data
    success = data_processor.load_data()
    
    return jsonify({
        'success': success,
        'message': 'Data cache refreshed successfully' if success else 'Failed to refresh data',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Enhanced health check with system status"""
    try:
        # Check data availability
        data_status = data_processor.load_data()
        
        # Get basic statistics
        stats = {
            'members_count': len(data_processor.members_df) if data_processor.members_df is not None else 0,
            'trades_count': len(data_processor.trades_df) if data_processor.trades_df is not None else 0,
            'cache_entries': len(_data_cache),
            'uptime': 'OK'
        }
        
        return jsonify({
            'status': 'healthy' if data_status else 'degraded',
            'service': 'congressional-trading-intelligence-enhanced',
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'data_status': stats,
            'features': [
                'Enhanced Statistics API',
                'ML-powered Anomaly Detection',
                'Risk Score Calculation',
                'Data Export Capabilities',
                'Predictive Analytics',
                'Real-time Caching',
                'Advanced Filtering'
            ]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Legacy endpoints for backward compatibility
@app.route('/api/stats')
def legacy_stats():
    """Legacy stats endpoint - redirects to v1"""
    return api_v1_stats()

@app.route('/api/members')
def legacy_members():
    """Legacy members endpoint - redirects to v1"""
    return api_v1_members()

@app.route('/api/trades')
def legacy_trades():
    """Legacy trades endpoint - redirects to v1"""
    return api_v1_trades()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # Initialize data on startup
    logger.info("Initializing Congressional Trading Intelligence Enhanced Backend...")
    data_processor.load_data()
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)