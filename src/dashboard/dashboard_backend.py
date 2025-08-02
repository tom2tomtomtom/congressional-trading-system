#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Dashboard Backend
Flask API service to power the enhanced HTML dashboard with real analysis data.
"""

import json
import sys
import os
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our analysis modules
from analysis.congressional_analysis import (
    get_congressional_trades_sample, 
    get_committee_assignments, 
    get_current_legislation,
    analyze_trading_patterns,
    calculate_suspicious_score
)
from analysis.advanced_pattern_analyzer import (
    SectorRotationAnalyzer,
    VolumeAnomalyDetector,
    BehaviorClusterAnalyzer,
    TimingCorrelationAnalyzer,
    run_advanced_analysis
)
from analysis.predictive_intelligence import (
    TradePredictionEngine,
    MarketImpactPredictor,
    LegislationOutcomePredictor,
    run_predictive_analysis
)

app = Flask(__name__)
CORS(app)

# Global data cache
_data_cache = {
    'trades_df': None,
    'committee_data': None,
    'legislation_data': None,
    'advanced_analysis': None,
    'predictive_analysis': None,
    'last_updated': None
}

def refresh_data_cache():
    """Refresh the data cache with latest analysis results."""
    global _data_cache
    
    print("Refreshing data cache...")
    
    # Load base data
    trades = get_congressional_trades_sample()
    committee_data = get_committee_assignments()
    legislation_data = get_current_legislation()
    
    # Process trades data
    df = pd.DataFrame(trades)
    df['transactionDate'] = pd.to_datetime(df['transactionDate'])
    df['filingDate'] = pd.to_datetime(df['filingDate'])
    df['filing_delay_days'] = (df['filingDate'] - df['transactionDate']).dt.days
    df['avg_amount'] = (df['amountFrom'] + df['amountTo']) / 2
    
    # Run advanced analysis
    advanced_results = run_advanced_analysis(df)
    
    # Run predictive analysis
    predictive_results = run_predictive_analysis(df, committee_data, legislation_data)
    
    # Update cache
    _data_cache.update({
        'trades_df': df,
        'committee_data': committee_data,
        'legislation_data': legislation_data,
        'advanced_analysis': advanced_results,
        'predictive_analysis': predictive_results,
        'last_updated': datetime.now()
    })
    
    print(f"Data cache refreshed at {_data_cache['last_updated']}")

@app.route('/')
def dashboard():
    """Serve the enhanced dashboard."""
    return render_template('enhanced_dashboard.html')

@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get overview statistics for the dashboard."""
    if _data_cache['trades_df'] is None:
        refresh_data_cache()
    
    df = _data_cache['trades_df']
    advanced = _data_cache['advanced_analysis']
    predictive = _data_cache['predictive_analysis']
    
    # Calculate statistics
    stats = {
        'total_members': df['name'].nunique(),
        'total_volume': f"${df['avg_amount'].sum() / 1000000:.1f}M",
        'avg_suspicion': f"{df.apply(lambda x: calculate_suspicious_score(pd.DataFrame([x])), axis=1).mean():.1f}",
        'anomalies_detected': len(advanced['volume_anomalies']) if len(advanced['volume_anomalies']) > 0 else 0,
        'prediction_accuracy': f"{predictive['model_performance']['trade_accuracy']:.1%}" if predictive['model_performance']['trade_accuracy'] else "N/A",
        'active_clusters': len(advanced['cluster_results'][1]) if advanced['cluster_results'][1] else 0,
        'last_updated': _data_cache['last_updated'].strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(stats)

@app.route('/api/analysis/advanced')
def get_advanced_analysis():
    """Get advanced pattern analysis results."""
    if _data_cache['advanced_analysis'] is None:
        refresh_data_cache()
    
    advanced = _data_cache['advanced_analysis']
    
    # Format sector rotation analysis
    sector_analysis = []
    if advanced['sector_analysis']:
        for member, data in advanced['sector_analysis'].items():
            sector_analysis.append({
                'member': member,
                'rotation_score': round(data['rotation_score'], 1),
                'unique_sectors': data['unique_sectors'],
                'sector_switches': data['sector_switches'],
                'dominant_sector': data['dominant_sector'],
                'concentration': round(data['sector_concentration'], 2)
            })
        sector_analysis.sort(key=lambda x: x['rotation_score'], reverse=True)
    
    # Format timing correlations
    timing_data = []
    if len(advanced['timing_correlations']) > 0:
        for _, correlation in advanced['timing_correlations'].iterrows():
            timing_data.append({
                'event': correlation['event'],
                'event_date': correlation['event_date'],
                'timing_score': round(correlation['suspicious_timing_score'], 1),
                'total_trades': int(correlation['total_trades']),
                'pre_event_trades': int(correlation['pre_event_trades']),
                'pre_event_volume': correlation['pre_event_volume']
            })
    
    return jsonify({
        'sector_rotation': sector_analysis[:10],  # Top 10
        'timing_correlations': timing_data,
        'pattern_summary': {
            'sector_patterns': len(sector_analysis),
            'timing_patterns': len(timing_data),
            'high_risk_patterns': len([s for s in sector_analysis if s['rotation_score'] > 5])
        }
    })

@app.route('/api/analysis/predictions')
def get_prediction_analysis():
    """Get ML prediction results."""
    if _data_cache['predictive_analysis'] is None:
        refresh_data_cache()
    
    predictive = _data_cache['predictive_analysis']
    
    # Format trade predictions
    trade_predictions = []
    for member, probability in predictive['trade_predictions'].items():
        risk_level = 'high' if probability > 0.7 else 'medium' if probability > 0.4 else 'low'
        trade_predictions.append({
            'member': member,
            'probability': round(probability, 3),
            'risk_level': risk_level,
            'confidence': 'Based on historical patterns and committee activity'
        })
    trade_predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    # Format impact predictions
    impact_predictions = []
    for member, impact in predictive['impact_predictions'].items():
        impact_predictions.append({
            'member': member,
            'predicted_impact': round(impact, 2),
            'impact_level': 'extreme' if impact > 7 else 'high' if impact > 4 else 'medium'
        })
    impact_predictions.sort(key=lambda x: x['predicted_impact'], reverse=True)
    
    # Format legislation correlations
    legislation_correlations = []
    for bill, analysis in predictive['legislation_correlations'].items():
        legislation_correlations.append({
            'bill': bill,
            'outcome_confidence': round(analysis['prediction_confidence'], 2),
            'trading_volume': analysis['total_volume'],
            'unique_traders': analysis['unique_members'],
            'suspicion_score': round(analysis['avg_suspicion_score'], 1)
        })
    
    return jsonify({
        'trade_predictions': trade_predictions[:10],
        'impact_predictions': impact_predictions[:10],
        'legislation_correlations': legislation_correlations,
        'model_performance': predictive['model_performance']
    })

@app.route('/api/analysis/clustering')
def get_clustering_analysis():
    """Get behavior clustering results."""
    if _data_cache['advanced_analysis'] is None:
        refresh_data_cache()
    
    cluster_results = _data_cache['advanced_analysis']['cluster_results']
    
    if cluster_results[0] is None:
        return jsonify({'error': 'Clustering analysis not available'})
    
    cluster_df, characteristics, silhouette = cluster_results
    
    # Format cluster data
    clusters = []
    for cluster_id, data in characteristics.items():
        clusters.append({
            'cluster_id': cluster_id,
            'profile': data['profile'],
            'members': data['members'],
            'size': data['size'],
            'statistics': {
                'avg_frequency': round(data['statistics']['trade_frequency']['mean'], 1),
                'avg_amount': round(10 ** data['statistics']['avg_amount_log']['mean'], 0),
                'avg_delay': round(data['statistics']['avg_filing_delay']['mean'], 1)
            }
        })
    
    # Member cluster assignments
    member_clusters = []
    for _, row in cluster_df.iterrows():
        member_clusters.append({
            'member': row['member'],
            'cluster': int(row['cluster'])
        })
    
    return jsonify({
        'clusters': clusters,
        'member_assignments': member_clusters,
        'silhouette_score': round(silhouette, 3),
        'summary': {
            'total_clusters': len(clusters),
            'quality_score': round(silhouette, 3),
            'members_clustered': len(member_clusters)
        }
    })

@app.route('/api/analysis/anomalies')
def get_anomaly_analysis():
    """Get volume anomaly detection results."""
    if _data_cache['advanced_analysis'] is None:
        refresh_data_cache()
    
    volume_anomalies = _data_cache['advanced_analysis']['volume_anomalies']
    
    if len(volume_anomalies) == 0:
        return jsonify({'anomalies': [], 'summary': {'total': 0}})
    
    # Format anomalies
    anomalies = []
    for _, anomaly in volume_anomalies.iterrows():
        anomalies.append({
            'member': anomaly['member'],
            'symbol': anomaly['symbol'],
            'date': anomaly['date'].strftime('%Y-%m-%d') if hasattr(anomaly['date'], 'strftime') else str(anomaly['date']),
            'amount': anomaly['amount'],
            'mean_amount': anomaly['mean_amount'],
            'z_score': round(anomaly['z_score'], 2),
            'anomaly_type': anomaly['anomaly_type'],
            'suspicion_level': anomaly['suspicion_level']
        })
    
    # Sort by z-score
    anomalies.sort(key=lambda x: x['z_score'], reverse=True)
    
    # Summary statistics
    summary = {
        'total': len(anomalies),
        'extreme': len([a for a in anomalies if a['suspicion_level'] == 'EXTREME']),
        'high': len([a for a in anomalies if a['suspicion_level'] == 'HIGH']),
        'medium': len([a for a in anomalies if a['suspicion_level'] == 'MEDIUM']),
        'avg_z_score': round(np.mean([a['z_score'] for a in anomalies]), 2)
    }
    
    return jsonify({
        'anomalies': anomalies,
        'summary': summary
    })

@app.route('/api/analysis/correlations')
def get_correlation_analysis():
    """Get event timing correlation results."""
    if _data_cache['advanced_analysis'] is None:
        refresh_data_cache()
    
    timing_correlations = _data_cache['advanced_analysis']['timing_correlations']
    
    if len(timing_correlations) == 0:
        return jsonify({'correlations': [], 'summary': {}})
    
    # Format correlations
    correlations = []
    for _, correlation in timing_correlations.iterrows():
        correlations.append({
            'event': correlation['event'],
            'event_date': correlation['event_date'],
            'event_type': correlation['event_type'],
            'total_trades': int(correlation['total_trades']),
            'pre_event_trades': int(correlation['pre_event_trades']),
            'post_event_trades': int(correlation['post_event_trades']),
            'pre_event_volume': correlation['pre_event_volume'],
            'post_event_volume': correlation['post_event_volume'],
            'timing_score': round(correlation['suspicious_timing_score'], 1)
        })
    
    # Sort by timing score
    correlations.sort(key=lambda x: x['timing_score'], reverse=True)
    
    # Summary
    summary = {
        'events_analyzed': len(correlations),
        'high_risk_events': len([c for c in correlations if c['timing_score'] > 7]),
        'total_pre_event_trades': sum([c['pre_event_trades'] for c in correlations]),
        'avg_timing_score': round(np.mean([c['timing_score'] for c in correlations]), 2)
    }
    
    return jsonify({
        'correlations': correlations,
        'summary': summary
    })

@app.route('/api/research/export/<format>')
def export_research_data(format):
    """Export research data in various formats."""
    if _data_cache['trades_df'] is None:
        refresh_data_cache()
    
    df = _data_cache['trades_df']
    
    if format == 'csv':
        # Prepare CSV data
        csv_data = df.to_csv(index=False)
        return jsonify({
            'format': 'csv',
            'data': csv_data,
            'filename': f'congressional_trades_{datetime.now().strftime("%Y%m%d")}.csv'
        })
    
    elif format == 'json':
        # Prepare JSON data
        json_data = df.to_dict('records')
        return jsonify({
            'format': 'json',
            'data': json_data,
            'filename': f'congressional_trades_{datetime.now().strftime("%Y%m%d")}.json'
        })
    
    elif format == 'summary':
        # Statistical summary
        summary = {
            'overview': {
                'total_trades': len(df),
                'unique_members': df['name'].nunique(),
                'date_range': f"{df['transactionDate'].min().strftime('%Y-%m-%d')} to {df['transactionDate'].max().strftime('%Y-%m-%d')}",
                'total_volume': df['avg_amount'].sum()
            },
            'statistics': {
                'mean_trade_amount': df['avg_amount'].mean(),
                'median_trade_amount': df['avg_amount'].median(),
                'std_trade_amount': df['avg_amount'].std(),
                'mean_filing_delay': df['filing_delay_days'].mean(),
                'median_filing_delay': df['filing_delay_days'].median(),
                'late_filings_pct': (df['filing_delay_days'] > 45).mean() * 100
            },
            'by_member': df.groupby('name').agg({
                'avg_amount': ['count', 'mean', 'sum'],
                'filing_delay_days': 'mean'
            }).round(2).to_dict(),
            'by_transaction_type': df['transactionType'].value_counts().to_dict(),
            'by_symbol': df['symbol'].value_counts().to_dict()
        }
        return jsonify(summary)
    
    else:
        return jsonify({'error': 'Unsupported format'}), 400

@app.route('/api/refresh')
def refresh_cache():
    """Manually refresh the data cache."""
    try:
        refresh_data_cache()
        return jsonify({
            'status': 'success',
            'message': 'Data cache refreshed successfully',
            'timestamp': _data_cache['last_updated'].isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to refresh cache: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_last_updated': _data_cache['last_updated'].isoformat() if _data_cache['last_updated'] else None,
        'cache_status': 'loaded' if _data_cache['trades_df'] is not None else 'empty'
    })

if __name__ == '__main__':
    # Initialize data cache on startup
    refresh_data_cache()
    
    # Run Flask app
    print("Starting Congressional Trading Intelligence Dashboard Backend...")
    print("Dashboard will be available at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/")
    
    app.run(host='0.0.0.0', port=5000, debug=True)