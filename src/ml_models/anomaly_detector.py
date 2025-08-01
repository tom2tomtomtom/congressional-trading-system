#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Suspicious Pattern Detection
Advanced anomaly detection for identifying potentially problematic trading patterns.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
from scipy import stats
import joblib

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    TIMING_ANOMALY = "timing_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    COORDINATION_PATTERN = "coordination_pattern"
    INSIDER_TIMING = "insider_timing"
    COMMITTEE_OVERLAP = "committee_overlap"
    UNUSUAL_PERFORMANCE = "unusual_performance"

class SeverityLevel(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Anomaly:
    """Data model for detected anomalies."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    member_id: str
    symbol: str
    detection_date: str
    trade_date: str
    description: str
    confidence_score: float
    evidence: Dict[str, Any]
    related_members: List[str]
    committee_context: Optional[str]

@dataclass
class SuspicionScore:
    """Comprehensive suspicion scoring for trades."""
    trade_id: str
    overall_score: float
    component_scores: Dict[str, float]
    risk_factors: List[str]
    anomalies: List[Anomaly]
    recommendations: List[str]

class TimingAnomalyDetector:
    """Detects unusual timing patterns in congressional trades."""
    
    def __init__(self):
        """Initialize timing anomaly detector."""
        self.model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, trading_data: pd.DataFrame):
        """
        Fit the timing anomaly detection model.
        
        Args:
            trading_data: DataFrame with trading data including dates and events
        """
        logger.info("Fitting timing anomaly detection model")
        
        # Create timing features
        features = self._create_timing_features(trading_data)
        
        # Fit scaler and model
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        self.is_fitted = True
        
        logger.info("Timing anomaly model fitted successfully")
    
    def _create_timing_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create timing-based features for anomaly detection."""
        features = []
        
        for _, trade in df.iterrows():
            trade_date = pd.to_datetime(trade.get('transaction_date', datetime.now()))
            filing_date = pd.to_datetime(trade.get('filing_date', datetime.now()))
            
            # Days between trade and filing
            filing_delay = (filing_date - trade_date).days
            
            # Day of week (0=Monday, 6=Sunday)
            day_of_week = trade_date.weekday()
            
            # Days until quarter end
            quarter_end = pd.Timestamp(trade_date.year, ((trade_date.quarter * 3)), 1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            days_to_quarter_end = (quarter_end - trade_date).days
            
            # Time since last trade (simulated)
            days_since_last_trade = np.random.exponential(30)  # Average 30 days
            
            # Market hours (simulated - actual would check if trade during market hours)
            market_hours = np.random.choice([0, 1], p=[0.1, 0.9])  # 90% during market hours
            
            features.append([
                filing_delay,
                day_of_week,
                days_to_quarter_end,
                days_since_last_trade,
                market_hours
            ])
        
        return np.array(features)
    
    def detect_anomalies(self, trading_data: pd.DataFrame) -> List[Anomaly]:
        """
        Detect timing anomalies in trading data.
        
        Args:
            trading_data: DataFrame with trading data
            
        Returns:
            List of detected timing anomalies
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        logger.info(f"Detecting timing anomalies in {len(trading_data)} trades")
        
        features = self._create_timing_features(trading_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(features_scaled)
        anomaly_scores = self.model.decision_function(features_scaled)
        
        anomalies = []
        for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
            if prediction == -1:  # Anomaly detected
                trade = trading_data.iloc[i]
                
                # Determine severity based on score
                if score < -0.5:
                    severity = SeverityLevel.CRITICAL
                elif score < -0.3:
                    severity = SeverityLevel.HIGH
                elif score < -0.1:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW
                
                anomaly = Anomaly(
                    anomaly_id=f"timing_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    anomaly_type=AnomalyType.TIMING_ANOMALY,
                    severity=severity,
                    member_id=trade.get('bioguide_id', 'UNKNOWN'),
                    symbol=trade.get('symbol', 'UNKNOWN'),
                    detection_date=datetime.now().isoformat(),
                    trade_date=str(trade.get('transaction_date', '')),
                    description=f"Unusual timing pattern detected (score: {score:.3f})",
                    confidence_score=abs(score),
                    evidence={
                        'anomaly_score': score,
                        'timing_features': features[i].tolist(),
                        'filing_delay': features[i][0],
                        'day_of_week': features[i][1]
                    },
                    related_members=[],
                    committee_context=None
                )
                anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} timing anomalies")
        return anomalies

class VolumeAnomalyDetector:
    """Detects unusual trading volume patterns."""
    
    def __init__(self):
        """Initialize volume anomaly detector."""
        self.member_profiles = {}  # Member trading profiles
        self.volume_thresholds = {}  # Dynamic thresholds per member
    
    def analyze_member_patterns(self, trading_data: pd.DataFrame):
        """
        Analyze historical trading patterns for each member.
        
        Args:
            trading_data: Historical trading data
        """
        logger.info("Analyzing member trading patterns")
        
        for member_id in trading_data['bioguide_id'].unique():
            member_trades = trading_data[trading_data['bioguide_id'] == member_id]
            
            # Calculate statistics
            trade_amounts = member_trades['amount_mid'].dropna()
            
            profile = {
                'total_trades': len(member_trades),
                'avg_amount': trade_amounts.mean() if len(trade_amounts) > 0 else 0,
                'median_amount': trade_amounts.median() if len(trade_amounts) > 0 else 0,
                'std_amount': trade_amounts.std() if len(trade_amounts) > 0 else 0,
                'max_amount': trade_amounts.max() if len(trade_amounts) > 0 else 0,
                'trading_frequency': len(member_trades) / 365,  # Trades per day
                'preferred_symbols': member_trades['symbol'].value_counts().head(5).to_dict()
            }
            
            self.member_profiles[member_id] = profile
            
            # Set dynamic thresholds (mean + 2*std)
            if profile['std_amount'] > 0:
                self.volume_thresholds[member_id] = profile['avg_amount'] + 2 * profile['std_amount']
            else:
                self.volume_thresholds[member_id] = profile['avg_amount'] * 3  # Fallback
        
        logger.info(f"Analyzed patterns for {len(self.member_profiles)} members")
    
    def detect_volume_anomalies(self, member_id: str, recent_trades: pd.DataFrame) -> List[Anomaly]:
        """
        Detect volume anomalies for a specific member.
        
        Args:
            member_id: Congressional member ID
            recent_trades: Recent trading data for the member
            
        Returns:
            List of volume anomalies
        """
        if member_id not in self.member_profiles:
            logger.warning(f"No profile found for member {member_id}")
            return []
        
        profile = self.member_profiles[member_id]
        threshold = self.volume_thresholds.get(member_id, 100000)  # Default threshold
        
        anomalies = []
        
        for _, trade in recent_trades.iterrows():
            trade_amount = trade.get('amount_mid', 0)
            
            # Check if trade exceeds threshold
            if trade_amount > threshold:
                # Calculate z-score
                if profile['std_amount'] > 0:
                    z_score = (trade_amount - profile['avg_amount']) / profile['std_amount']
                else:
                    z_score = 3.0  # Default high score
                
                # Determine severity
                if z_score > 4:
                    severity = SeverityLevel.CRITICAL
                elif z_score > 3:
                    severity = SeverityLevel.HIGH
                elif z_score > 2:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW
                
                anomaly = Anomaly(
                    anomaly_id=f"volume_{member_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=severity,
                    member_id=member_id,
                    symbol=trade.get('symbol', 'UNKNOWN'),
                    detection_date=datetime.now().isoformat(),
                    trade_date=str(trade.get('transaction_date', '')),
                    description=f"Unusually large trade: ${trade_amount:,.0f} (z-score: {z_score:.2f})",
                    confidence_score=min(z_score / 4.0, 1.0),
                    evidence={
                        'trade_amount': trade_amount,
                        'member_avg': profile['avg_amount'],
                        'member_max': profile['max_amount'],
                        'z_score': z_score,
                        'threshold': threshold
                    },
                    related_members=[],
                    committee_context=None
                )
                anomalies.append(anomaly)
        
        return anomalies

class CoordinationDetector:
    """Detects coordinated trading patterns among congressional members."""
    
    def __init__(self):
        """Initialize coordination detector."""
        self.similarity_threshold = 0.7
        self.time_window_days = 7  # Look for trades within 7 days
    
    def detect_coordination_patterns(self, trading_data: pd.DataFrame) -> List[Anomaly]:
        """
        Detect coordinated trading patterns.
        
        Args:
            trading_data: Trading data across multiple members
            
        Returns:
            List of coordination anomalies
        """
        logger.info(f"Detecting coordination patterns in {len(trading_data)} trades")
        
        # Group trades by symbol and time windows
        coordination_groups = self._find_coordination_groups(trading_data)
        
        anomalies = []
        for group in coordination_groups:
            if len(group['members']) >= 2:  # At least 2 members coordinating
                
                # Calculate coordination strength
                strength = self._calculate_coordination_strength(group)
                
                if strength > self.similarity_threshold:
                    # Determine severity based on group size and strength
                    if len(group['members']) >= 5 and strength > 0.9:
                        severity = SeverityLevel.CRITICAL
                    elif len(group['members']) >= 3 and strength > 0.8:
                        severity = SeverityLevel.HIGH
                    elif strength > 0.75:
                        severity = SeverityLevel.MEDIUM
                    else:
                        severity = SeverityLevel.LOW
                    
                    # Create anomaly for each member in the group
                    for member_id in group['members']:
                        anomaly = Anomaly(
                            anomaly_id=f"coord_{group['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{member_id}",
                            anomaly_type=AnomalyType.COORDINATION_PATTERN,
                            severity=severity,
                            member_id=member_id,
                            symbol=group['symbol'],
                            detection_date=datetime.now().isoformat(),
                            trade_date=group['date_range'],
                            description=f"Coordinated trading detected: {len(group['members'])} members trading {group['symbol']} within {self.time_window_days} days",
                            confidence_score=strength,
                            evidence={
                                'coordination_strength': strength,
                                'group_size': len(group['members']),
                                'time_window': self.time_window_days,
                                'trade_dates': group['trade_dates'],
                                'amounts': group['amounts']
                            },
                            related_members=[m for m in group['members'] if m != member_id],
                            committee_context=group.get('committee_overlap')
                        )
                        anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} coordination anomalies")
        return anomalies
    
    def _find_coordination_groups(self, trading_data: pd.DataFrame) -> List[Dict]:
        """Find groups of potentially coordinated trades."""
        groups = []
        
        # Group by symbol
        for symbol in trading_data['symbol'].unique():
            symbol_trades = trading_data[trading_data['symbol'] == symbol].copy()
            symbol_trades['transaction_date'] = pd.to_datetime(symbol_trades['transaction_date'])
            symbol_trades = symbol_trades.sort_values('transaction_date')
            
            # Find trades within time windows
            for i, trade1 in symbol_trades.iterrows():
                window_start = trade1['transaction_date']
                window_end = window_start + timedelta(days=self.time_window_days)
                
                window_trades = symbol_trades[
                    (symbol_trades['transaction_date'] >= window_start) &
                    (symbol_trades['transaction_date'] <= window_end)
                ]
                
                if len(window_trades) >= 2:
                    group = {
                        'symbol': symbol,
                        'members': window_trades['bioguide_id'].unique().tolist(),
                        'trade_dates': window_trades['transaction_date'].dt.strftime('%Y-%m-%d').tolist(),
                        'amounts': window_trades['amount_mid'].tolist(),
                        'date_range': f"{window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}",
                        'committee_overlap': None  # Would be populated with actual committee data
                    }
                    groups.append(group)
        
        return groups
    
    def _calculate_coordination_strength(self, group: Dict) -> float:
        """Calculate coordination strength score."""
        # Simple coordination metric based on:
        # 1. Number of members (more = higher coordination)
        # 2. Time clustering (closer in time = higher coordination)
        # 3. Amount similarity (similar amounts = higher coordination)
        
        base_score = min(len(group['members']) / 10.0, 1.0)  # Normalize by 10 members max
        
        # Time clustering bonus
        dates = pd.to_datetime(group['trade_dates'])
        if len(dates) > 1:
            date_spread = (dates.max() - dates.min()).days
            time_score = max(0, 1.0 - date_spread / self.time_window_days)
        else:
            time_score = 1.0
        
        # Amount similarity bonus
        amounts = [a for a in group['amounts'] if pd.notna(a)]
        if len(amounts) > 1:
            amount_cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 1.0
            amount_score = max(0, 1.0 - amount_cv)
        else:
            amount_score = 0.5
        
        # Weighted combination
        coordination_strength = (0.5 * base_score + 0.3 * time_score + 0.2 * amount_score)
        
        return coordination_strength

class SuspiciousPatternDetector:
    """
    Main class for detecting suspicious trading patterns in congressional trades.
    Combines multiple detection methods for comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize suspicious pattern detector."""
        self.timing_detector = TimingAnomalyDetector()
        self.volume_detector = VolumeAnomalyDetector()
        self.coordination_detector = CoordinationDetector()
        
        self.is_trained = False
    
    def train(self, historical_data: pd.DataFrame):
        """
        Train all detection models on historical data.
        
        Args:
            historical_data: Historical trading data for training
        """
        logger.info("Training suspicious pattern detection models")
        
        # Train timing anomaly detector
        self.timing_detector.fit(historical_data)
        
        # Analyze member patterns for volume detector
        self.volume_detector.analyze_member_patterns(historical_data)
        
        self.is_trained = True
        logger.info("All detection models trained successfully")
    
    def detect_all_anomalies(self, trading_data: pd.DataFrame) -> List[Anomaly]:
        """
        Run comprehensive anomaly detection on trading data.
        
        Args:
            trading_data: Trading data to analyze
            
        Returns:
            List of all detected anomalies
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before use")
        
        logger.info(f"Running comprehensive anomaly detection on {len(trading_data)} trades")
        
        all_anomalies = []
        
        # Timing anomalies
        try:
            timing_anomalies = self.timing_detector.detect_anomalies(trading_data)
            all_anomalies.extend(timing_anomalies)
            logger.info(f"Found {len(timing_anomalies)} timing anomalies")
        except Exception as e:
            logger.error(f"Error in timing detection: {e}")
        
        # Volume anomalies (per member)
        try:
            volume_anomalies = []
            for member_id in trading_data['bioguide_id'].unique():
                member_trades = trading_data[trading_data['bioguide_id'] == member_id]
                member_anomalies = self.volume_detector.detect_volume_anomalies(member_id, member_trades)
                volume_anomalies.extend(member_anomalies)
            
            all_anomalies.extend(volume_anomalies)
            logger.info(f"Found {len(volume_anomalies)} volume anomalies")
        except Exception as e:
            logger.error(f"Error in volume detection: {e}")
        
        # Coordination patterns
        try:
            coordination_anomalies = self.coordination_detector.detect_coordination_patterns(trading_data)
            all_anomalies.extend(coordination_anomalies)
            logger.info(f"Found {len(coordination_anomalies)} coordination anomalies")
        except Exception as e:
            logger.error(f"Error in coordination detection: {e}")
        
        logger.info(f"Total anomalies detected: {len(all_anomalies)}")
        return all_anomalies
    
    def calculate_suspicion_score(self, trade_data: Dict[str, Any], 
                                 detected_anomalies: List[Anomaly]) -> SuspicionScore:
        """
        Calculate comprehensive suspicion score for a trade.
        
        Args:
            trade_data: Individual trade data
            detected_anomalies: Anomalies detected for this trade
            
        Returns:
            Comprehensive suspicion score
        """
        trade_id = trade_data.get('id', 'unknown')
        
        # Component scores (0-1 scale)
        component_scores = {
            'timing_score': 0.0,
            'volume_score': 0.0,
            'coordination_score': 0.0,
            'committee_overlap_score': 0.0,
            'performance_score': 0.0
        }
        
        risk_factors = []
        
        # Process detected anomalies
        for anomaly in detected_anomalies:
            if anomaly.anomaly_type == AnomalyType.TIMING_ANOMALY:
                component_scores['timing_score'] = max(component_scores['timing_score'], anomaly.confidence_score)
                risk_factors.append("Unusual timing pattern")
            
            elif anomaly.anomaly_type == AnomalyType.VOLUME_ANOMALY:
                component_scores['volume_score'] = max(component_scores['volume_score'], anomaly.confidence_score)
                risk_factors.append("Unusually large trade volume")
            
            elif anomaly.anomaly_type == AnomalyType.COORDINATION_PATTERN:
                component_scores['coordination_score'] = max(component_scores['coordination_score'], anomaly.confidence_score)
                risk_factors.append("Potential coordination with other members")
        
        # Add committee overlap analysis (simplified)
        if trade_data.get('committee_relevance', False):
            component_scores['committee_overlap_score'] = 0.6
            risk_factors.append("Trade in sector under member's committee jurisdiction")
        
        # Calculate overall score (weighted average)
        weights = {
            'timing_score': 0.25,
            'volume_score': 0.20,
            'coordination_score': 0.30,
            'committee_overlap_score': 0.15,
            'performance_score': 0.10
        }
        
        overall_score = sum(component_scores[key] * weights[key] for key in weights)
        
        # Generate recommendations
        recommendations = []
        if overall_score > 0.7:
            recommendations.append("Recommend detailed review by ethics committee")
        if component_scores['coordination_score'] > 0.6:
            recommendations.append("Investigate potential coordination with related members")
        if component_scores['timing_score'] > 0.5:
            recommendations.append("Review timing relative to relevant events")
        
        return SuspicionScore(
            trade_id=trade_id,
            overall_score=overall_score,
            component_scores=component_scores,
            risk_factors=risk_factors,
            anomalies=detected_anomalies,
            recommendations=recommendations
        )
    
    def generate_anomaly_report(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Generate comprehensive anomaly report."""
        # Count by type and severity
        type_counts = {}
        severity_counts = {}
        
        for anomaly in anomalies:
            # Count by type
            anomaly_type = anomaly.anomaly_type.value
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            
            # Count by severity
            severity = anomaly.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Find most suspicious members
        member_anomaly_counts = {}
        for anomaly in anomalies:
            member_id = anomaly.member_id
            member_anomaly_counts[member_id] = member_anomaly_counts.get(member_id, 0) + 1
        
        most_suspicious = sorted(member_anomaly_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_anomalies': len(anomalies),
            'by_type': type_counts,
            'by_severity': severity_counts,
            'most_suspicious_members': most_suspicious,
            'critical_anomalies': [a for a in anomalies if a.severity == SeverityLevel.CRITICAL],
            'generated_at': datetime.now().isoformat()
        }

def main():
    """Test function for suspicious pattern detector."""
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_trades = 1000
    
    sample_data = []
    members = [f'M{i:03d}' for i in range(100)]
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    for i in range(n_trades):
        trade_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        filing_date = trade_date + timedelta(days=np.random.randint(1, 45))
        
        sample_data.append({
            'id': i,
            'bioguide_id': np.random.choice(members),
            'symbol': np.random.choice(symbols),
            'transaction_date': trade_date,
            'filing_date': filing_date,
            'amount_mid': np.random.lognormal(10, 1),  # Log-normal distribution for amounts
            'transaction_type': np.random.choice(['Purchase', 'Sale'], p=[0.6, 0.4])
        })
    
    trading_data = pd.DataFrame(sample_data)
    
    print("Initializing and training suspicious pattern detector...")
    detector = SuspiciousPatternDetector()
    
    # Split data for training and testing
    train_data = trading_data.sample(frac=0.8, random_state=42)
    test_data = trading_data.drop(train_data.index)
    
    # Train detector
    detector.train(train_data)
    
    print(f"Testing on {len(test_data)} trades...")
    anomalies = detector.detect_all_anomalies(test_data)
    
    print(f"\nDetected {len(anomalies)} anomalies:")
    
    # Show summary by type and severity
    report = detector.generate_anomaly_report(anomalies)
    print(f"By type: {report['by_type']}")
    print(f"By severity: {report['by_severity']}")
    
    # Show top 5 anomalies
    print(f"\nTop 5 most suspicious anomalies:")
    sorted_anomalies = sorted(anomalies, key=lambda x: x.confidence_score, reverse=True)[:5]
    
    for i, anomaly in enumerate(sorted_anomalies, 1):
        print(f"{i}. {anomaly.anomaly_type.value} - {anomaly.severity.value}")
        print(f"   Member: {anomaly.member_id}, Symbol: {anomaly.symbol}")
        print(f"   Confidence: {anomaly.confidence_score:.3f}")
        print(f"   Description: {anomaly.description}")
        print()

if __name__ == "__main__":
    main()