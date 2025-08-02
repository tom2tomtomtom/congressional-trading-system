#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Advanced Pattern Analysis
Enhanced analysis capabilities for deep pattern recognition and behavioral clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SectorRotationAnalyzer:
    """Analyze how congressional members rotate between sectors over time."""
    
    def __init__(self):
        self.sector_mapping = {
            'NVDA': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'AAPL': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Technology', 'COIN': 'Technology', 'RBLX': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'UNH': 'Healthcare', 'PFE': 'Healthcare', 'JNJ': 'Healthcare',
            'DIS': 'Media', 'NFLX': 'Media', 'WMG': 'Media',
            'HCA': 'Healthcare', 'VFC': 'Consumer', 'ICE': 'Financial',
            'CARR': 'Industrial', 'ZM': 'Technology'
        }
    
    def analyze_sector_rotation(self, trades_df):
        """
        Analyze sector rotation patterns for each congressional member.
        """
        # Add sector information
        trades_df['sector'] = trades_df['symbol'].map(self.sector_mapping)
        trades_df['sector'] = trades_df['sector'].fillna('Other')
        
        # Sort by date
        trades_df = trades_df.sort_values('transactionDate')
        
        rotation_analysis = {}
        
        for member in trades_df['name'].unique():
            member_trades = trades_df[trades_df['name'] == member].copy()
            
            if len(member_trades) < 2:
                continue
                
            # Track sector sequence
            sectors = member_trades['sector'].tolist()
            dates = member_trades['transactionDate'].tolist()
            amounts = member_trades['avg_amount'].tolist()
            
            # Calculate rotation metrics
            rotation_analysis[member] = {
                'sector_sequence': sectors,
                'trade_dates': dates,
                'trade_amounts': amounts,
                'unique_sectors': len(set(sectors)),
                'sector_switches': self._count_sector_switches(sectors),
                'rotation_score': self._calculate_rotation_score(sectors, amounts),
                'dominant_sector': max(set(sectors), key=sectors.count),
                'sector_concentration': max([sectors.count(s) for s in set(sectors)]) / len(sectors)
            }
        
        return rotation_analysis
    
    def _count_sector_switches(self, sectors):
        """Count the number of times a member switches sectors."""
        switches = 0
        for i in range(1, len(sectors)):
            if sectors[i] != sectors[i-1]:
                switches += 1
        return switches
    
    def _calculate_rotation_score(self, sectors, amounts):
        """Calculate a rotation suspicion score (higher = more suspicious)."""
        if len(sectors) <= 1:
            return 0
        
        # Base score on sector diversity and timing
        diversity_score = len(set(sectors)) / len(sectors)
        switch_frequency = self._count_sector_switches(sectors) / len(sectors)
        
        # Weight by trade amounts (larger amounts = more suspicious rotations)
        avg_amount = np.mean(amounts)
        amount_weight = min(2.0, avg_amount / 100000)  # Cap at 2x weight
        
        rotation_score = (diversity_score + switch_frequency) * amount_weight * 10
        return min(10, rotation_score)  # Cap at 10

class VolumeAnomalyDetector:
    """Detect unusual trading volume patterns that may indicate insider knowledge."""
    
    def __init__(self):
        self.z_score_threshold = 2.5  # Standard deviations for anomaly detection
    
    def detect_volume_anomalies(self, trades_df):
        """
        Detect trading volume anomalies for each member and stock.
        """
        anomalies = []
        
        # Group by member and stock
        for (member, symbol), group in trades_df.groupby(['name', 'symbol']):
            if len(group) < 3:  # Need at least 3 trades for statistical analysis
                continue
            
            amounts = group['avg_amount'].values
            dates = group['transactionDate'].values
            
            # Calculate statistical measures
            mean_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            
            if std_amount == 0:  # All trades same amount
                continue
            
            # Detect anomalies using z-score
            for i, (amount, date) in enumerate(zip(amounts, dates)):
                z_score = abs((amount - mean_amount) / std_amount)
                
                if z_score > self.z_score_threshold:
                    anomalies.append({
                        'member': member,
                        'symbol': symbol,
                        'date': date,
                        'amount': amount,
                        'mean_amount': mean_amount,
                        'z_score': z_score,
                        'anomaly_type': 'volume_spike' if amount > mean_amount else 'volume_drop',
                        'suspicion_level': self._classify_suspicion_level(z_score)
                    })
        
        return pd.DataFrame(anomalies)
    
    def _classify_suspicion_level(self, z_score):
        """Classify suspicion level based on z-score."""
        if z_score > 4:
            return 'EXTREME'
        elif z_score > 3:
            return 'HIGH'
        elif z_score > 2.5:
            return 'MEDIUM'
        else:
            return 'LOW'

class BehaviorClusterAnalyzer:
    """Cluster congressional members by trading behavior patterns."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def cluster_trading_behavior(self, trades_df, n_clusters=5):
        """
        Cluster members based on trading behavior features.
        """
        # Create feature matrix for each member
        features = []
        member_names = []
        
        for member in trades_df['name'].unique():
            member_trades = trades_df[trades_df['name'] == member]
            
            if len(member_trades) == 0:
                continue
            
            # Calculate behavioral features
            feature_vector = self._extract_behavioral_features(member_trades)
            features.append(feature_vector)
            member_names.append(member)
        
        if len(features) < n_clusters:
            return None, None, None
        
        # Convert to numpy array and handle NaN values
        features = np.array(features)
        
        # Replace NaN and infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'member': member_names,
            'cluster': cluster_labels
        })
        
        # Add cluster characteristics
        cluster_characteristics = self._analyze_cluster_characteristics(
            features, cluster_labels, member_names, n_clusters
        )
        
        return results_df, cluster_characteristics, silhouette_avg
    
    def _extract_behavioral_features(self, member_trades):
        """Extract behavioral features for clustering."""
        features = []
        
        # Trading frequency (normalized)
        features.append(len(member_trades))
        
        # Average trade amount (log-scaled)
        avg_amount = member_trades['avg_amount'].mean()
        features.append(np.log10(max(1, avg_amount)))
        
        # Trade amount volatility
        amount_std = member_trades['avg_amount'].std()
        features.append(amount_std / avg_amount if avg_amount > 0 else 0)
        
        # Filing delay patterns
        avg_delay = member_trades['filing_delay_days'].mean()
        features.append(avg_delay)
        
        # Purchase vs sale ratio
        purchases = len(member_trades[member_trades['transactionType'] == 'Purchase'])
        total_trades = len(member_trades)
        features.append(purchases / total_trades if total_trades > 0 else 0)
        
        # Sector diversity
        unique_sectors = member_trades['symbol'].map(
            lambda x: self._get_sector(x)
        ).nunique()
        features.append(unique_sectors)
        
        # Timing consistency (coefficient of variation of trade intervals)
        if len(member_trades) > 1:
            dates = pd.to_datetime(member_trades['transactionDate']).sort_values()
            intervals = [(dates.iloc[i+1] - dates.iloc[i]).days 
                        for i in range(len(dates)-1)]
            if len(intervals) > 1 and np.mean(intervals) > 0:
                timing_cv = np.std(intervals) / np.mean(intervals)
            else:
                timing_cv = 0
        else:
            timing_cv = 0
        features.append(timing_cv)
        
        return features
    
    def _get_sector(self, symbol):
        """Get sector for a stock symbol."""
        sector_mapping = {
            'NVDA': 'Tech', 'GOOGL': 'Tech', 'MSFT': 'Tech', 'AAPL': 'Tech',
            'META': 'Tech', 'AMZN': 'Tech', 'TSLA': 'Tech', 'COIN': 'Tech',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'PFE': 'Healthcare',
            'UNH': 'Healthcare', 'DIS': 'Media'
        }
        return sector_mapping.get(symbol, 'Other')
    
    def _analyze_cluster_characteristics(self, features, labels, member_names, n_clusters):
        """Analyze characteristics of each cluster."""
        feature_names = [
            'trade_frequency', 'avg_amount_log', 'amount_volatility',
            'avg_filing_delay', 'purchase_ratio', 'sector_diversity',
            'timing_consistency'
        ]
        
        characteristics = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_members = [member_names[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_features) == 0:
                continue
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i, feature_name in enumerate(feature_names):
                feature_values = cluster_features[:, i]
                cluster_stats[feature_name] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'median': np.median(feature_values)
                }
            
            characteristics[cluster_id] = {
                'members': cluster_members,
                'size': len(cluster_members),
                'statistics': cluster_stats,
                'profile': self._generate_cluster_profile(cluster_stats)
            }
        
        return characteristics
    
    def _generate_cluster_profile(self, stats):
        """Generate a human-readable profile for the cluster."""
        profile = []
        
        # Trading frequency
        freq = stats['trade_frequency']['mean']
        if freq > 5:
            profile.append("Active traders")
        elif freq > 2:
            profile.append("Moderate traders")
        else:
            profile.append("Infrequent traders")
        
        # Trade amounts
        avg_amount = 10 ** stats['avg_amount_log']['mean']
        if avg_amount > 500000:
            profile.append("Large positions")
        elif avg_amount > 100000:
            profile.append("Medium positions")
        else:
            profile.append("Small positions")
        
        # Filing behavior
        delay = stats['avg_filing_delay']['mean']
        if delay > 60:
            profile.append("Late filers")
        elif delay > 30:
            profile.append("Moderate filing delays")
        else:
            profile.append("Timely filers")
        
        return " | ".join(profile)

class TimingCorrelationAnalyzer:
    """Analyze timing correlations between trades and market events."""
    
    def __init__(self):
        self.market_events = [
            {'date': '2020-03-20', 'event': 'COVID-19 Market Crash', 'type': 'crisis'},
            {'date': '2020-11-09', 'event': 'Vaccine Announcement', 'type': 'positive'},
            {'date': '2022-03-15', 'event': 'Fed Rate Hike', 'type': 'policy'},
            {'date': '2023-11-30', 'event': 'AI Executive Order', 'type': 'policy'},
            {'date': '2024-01-15', 'event': 'Tech Earnings Season', 'type': 'earnings'}
        ]
    
    def analyze_event_timing(self, trades_df, window_days=30):
        """
        Analyze correlation between trades and major market events.
        """
        correlations = []
        
        for event in self.market_events:
            event_date = pd.to_datetime(event['date'])
            window_start = event_date - timedelta(days=window_days)
            window_end = event_date + timedelta(days=window_days)
            
            # Find trades within the event window
            event_trades = trades_df[
                (trades_df['transactionDate'] >= window_start) &
                (trades_df['transactionDate'] <= window_end)
            ].copy()
            
            if len(event_trades) == 0:
                continue
            
            # Calculate days relative to event
            event_trades['days_to_event'] = (
                event_date - event_trades['transactionDate']
            ).dt.days
            
            # Analyze pre-event vs post-event trading
            pre_event = event_trades[event_trades['days_to_event'] > 0]
            post_event = event_trades[event_trades['days_to_event'] <= 0]
            
            correlation_data = {
                'event': event['event'],
                'event_date': event['date'],
                'event_type': event['type'],
                'total_trades': len(event_trades),
                'pre_event_trades': len(pre_event),
                'post_event_trades': len(post_event),
                'pre_event_volume': pre_event['avg_amount'].sum() if len(pre_event) > 0 else 0,
                'post_event_volume': post_event['avg_amount'].sum() if len(post_event) > 0 else 0,
                'suspicious_timing_score': self._calculate_timing_suspicion_score(
                    pre_event, post_event, event['type']
                )
            }
            
            correlations.append(correlation_data)
        
        return pd.DataFrame(correlations)
    
    def _calculate_timing_suspicion_score(self, pre_trades, post_trades, event_type):
        """Calculate suspicion score based on timing relative to events."""
        if len(pre_trades) == 0 and len(post_trades) == 0:
            return 0
        
        score = 0
        
        # Pre-event trading (potentially more suspicious)
        if len(pre_trades) > 0:
            pre_volume = pre_trades['avg_amount'].sum()
            pre_avg_days = pre_trades['days_to_event'].mean()
            
            # Score based on proximity to event and volume
            proximity_score = max(0, (30 - pre_avg_days) / 30) * 5  # Max 5 points
            volume_score = min(3, np.log10(pre_volume / 100000))  # Max 3 points
            
            score += proximity_score + volume_score
        
        # Event type adjustments
        if event_type == 'crisis':
            score *= 1.5  # Crisis trading more suspicious
        elif event_type == 'policy':
            score *= 1.3  # Policy trading moderately suspicious
        
        return min(10, score)  # Cap at 10

def create_advanced_visualizations(sector_analysis, volume_anomalies, cluster_results, timing_correlations):
    """Create comprehensive visualizations for advanced pattern analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Advanced Congressional Trading Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sector Rotation Analysis
    if sector_analysis:
        members = list(sector_analysis.keys())
        rotation_scores = [sector_analysis[m]['rotation_score'] for m in members]
        sector_switches = [sector_analysis[m]['sector_switches'] for m in members]
        
        scatter = axes[0,0].scatter(sector_switches, rotation_scores, 
                                  c=rotation_scores, cmap='Reds', s=100, alpha=0.7)
        axes[0,0].set_xlabel('Number of Sector Switches')
        axes[0,0].set_ylabel('Rotation Suspicion Score')
        axes[0,0].set_title('Sector Rotation Patterns')
        
        # Add member labels for high scores
        for i, member in enumerate(members):
            if rotation_scores[i] > 5:
                axes[0,0].annotate(member.split()[0], 
                                 (sector_switches[i], rotation_scores[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 2. Volume Anomaly Detection
    if len(volume_anomalies) > 0:
        anomaly_counts = volume_anomalies['member'].value_counts()
        axes[0,1].bar(range(len(anomaly_counts)), anomaly_counts.values)
        axes[0,1].set_xlabel('Congressional Members')
        axes[0,1].set_ylabel('Number of Volume Anomalies')
        axes[0,1].set_title('Volume Anomaly Detection')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_xticks(range(len(anomaly_counts)))
        axes[0,1].set_xticklabels([name.split()[0] for name in anomaly_counts.index])
    
    # 3. Behavior Clustering
    if cluster_results[0] is not None:
        cluster_df, characteristics, silhouette = cluster_results
        cluster_counts = cluster_df['cluster'].value_counts().sort_index()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        axes[1,0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                     autopct='%1.1f%%', colors=colors)
        axes[1,0].set_title(f'Trading Behavior Clusters\n(Silhouette Score: {silhouette:.3f})')
    
    # 4. Event Timing Correlations
    if len(timing_correlations) > 0:
        events = timing_correlations['event'].values
        scores = timing_correlations['suspicious_timing_score'].values
        
        bars = axes[1,1].bar(range(len(events)), scores, 
                           color=['red' if s > 7 else 'orange' if s > 4 else 'yellow' for s in scores])
        axes[1,1].set_xlabel('Market Events')
        axes[1,1].set_ylabel('Timing Suspicion Score')
        axes[1,1].set_title('Event Timing Correlation Analysis')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_xticks(range(len(events)))
        axes[1,1].set_xticklabels([event.split()[0] for event in events])
    
    plt.tight_layout()
    plt.savefig('advanced_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_advanced_analysis(trades_df):
    """Run comprehensive advanced pattern analysis."""
    
    print("ADVANCED CONGRESSIONAL TRADING PATTERN ANALYSIS")
    print("=" * 60)
    print()
    
    # Initialize analyzers
    sector_analyzer = SectorRotationAnalyzer()
    volume_detector = VolumeAnomalyDetector()
    cluster_analyzer = BehaviorClusterAnalyzer()
    timing_analyzer = TimingCorrelationAnalyzer()
    
    # 1. Sector Rotation Analysis
    print("1. SECTOR ROTATION ANALYSIS")
    print("-" * 30)
    sector_analysis = sector_analyzer.analyze_sector_rotation(trades_df)
    
    if sector_analysis:
        # Sort by rotation score
        sorted_members = sorted(sector_analysis.items(), 
                              key=lambda x: x[1]['rotation_score'], reverse=True)
        
        for member, data in sorted_members[:5]:  # Top 5
            print(f"{member}:")
            print(f"  Rotation Score: {data['rotation_score']:.1f}/10")
            print(f"  Sectors Traded: {data['unique_sectors']}")
            print(f"  Sector Switches: {data['sector_switches']}")
            print(f"  Dominant Sector: {data['dominant_sector']}")
            print(f"  Concentration: {data['sector_concentration']:.1%}")
            print()
    
    # 2. Volume Anomaly Detection
    print("2. VOLUME ANOMALY DETECTION")
    print("-" * 30)
    volume_anomalies = volume_detector.detect_volume_anomalies(trades_df)
    
    if len(volume_anomalies) > 0:
        print(f"Detected {len(volume_anomalies)} volume anomalies:")
        for _, anomaly in volume_anomalies.head(5).iterrows():
            print(f"  {anomaly['member']} - {anomaly['symbol']}")
            print(f"    Date: {anomaly['date']}")
            print(f"    Amount: ${anomaly['amount']:,.0f} (avg: ${anomaly['mean_amount']:,.0f})")
            print(f"    Z-Score: {anomaly['z_score']:.2f} ({anomaly['suspicion_level']})")
            print()
    else:
        print("No volume anomalies detected with current dataset.")
        print()
    
    # 3. Behavior Clustering
    print("3. TRADING BEHAVIOR CLUSTERING")
    print("-" * 30)
    cluster_results = cluster_analyzer.cluster_trading_behavior(trades_df)
    
    if cluster_results[0] is not None:
        cluster_df, characteristics, silhouette = cluster_results
        print(f"Clustering Quality (Silhouette Score): {silhouette:.3f}")
        print()
        
        for cluster_id, data in characteristics.items():
            print(f"Cluster {cluster_id}: {data['profile']}")
            print(f"  Members ({data['size']}): {', '.join([m.split()[0] for m in data['members']])}")
            print()
    else:
        print("Insufficient data for clustering analysis.")
        print()
    
    # 4. Event Timing Correlation
    print("4. MARKET EVENT TIMING ANALYSIS")
    print("-" * 30)
    timing_correlations = timing_analyzer.analyze_event_timing(trades_df)
    
    if len(timing_correlations) > 0:
        sorted_correlations = timing_correlations.sort_values('suspicious_timing_score', ascending=False)
        
        for _, correlation in sorted_correlations.iterrows():
            print(f"{correlation['event']} ({correlation['event_date']}):")
            print(f"  Timing Suspicion Score: {correlation['suspicious_timing_score']:.1f}/10")
            print(f"  Total Trades: {correlation['total_trades']}")
            print(f"  Pre-event: {correlation['pre_event_trades']} trades, ${correlation['pre_event_volume']:,.0f}")
            print(f"  Post-event: {correlation['post_event_trades']} trades, ${correlation['post_event_volume']:,.0f}")
            print()
    
    # Create visualizations
    create_advanced_visualizations(sector_analysis, volume_anomalies, cluster_results, timing_correlations)
    
    return {
        'sector_analysis': sector_analysis,
        'volume_anomalies': volume_anomalies,
        'cluster_results': cluster_results,
        'timing_correlations': timing_correlations
    }

if __name__ == "__main__":
    # Import base data
    import sys
    sys.path.append('..')
    from congressional_analysis import get_congressional_trades_sample, analyze_trading_patterns
    
    # Load and process data
    trades = get_congressional_trades_sample()
    df, analysis = analyze_trading_patterns(trades)
    
    # Run advanced analysis
    results = run_advanced_analysis(df)
    
    print("\nAdvanced analysis complete! Visualization saved as 'advanced_pattern_analysis.png'")