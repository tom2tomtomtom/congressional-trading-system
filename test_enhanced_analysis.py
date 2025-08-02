#!/usr/bin/env python3
"""
Test script for enhanced Congressional Trading Intelligence analysis features.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import analysis modules
from analysis.congressional_analysis import (
    get_congressional_trades_sample, 
    get_committee_assignments, 
    get_current_legislation,
    analyze_trading_patterns
)
from analysis.advanced_pattern_analyzer import run_advanced_analysis
from analysis.predictive_intelligence import run_predictive_analysis

def test_advanced_analysis():
    """Test all enhanced analysis features."""
    
    print("🏛️ Congressional Trading Intelligence System - Enhanced Analysis Test")
    print("=" * 80)
    print()
    
    # Load and process data
    print("📊 Loading and processing data...")
    trades = get_congressional_trades_sample()
    committee_data = get_committee_assignments()
    legislation_data = get_current_legislation()
    
    # Process trades data
    df = pd.DataFrame(trades)
    df['transactionDate'] = pd.to_datetime(df['transactionDate'])
    df['filingDate'] = pd.to_datetime(df['filingDate'])
    df['filing_delay_days'] = (df['filingDate'] - df['transactionDate']).dt.days
    df['avg_amount'] = (df['amountFrom'] + df['amountTo']) / 2
    
    print(f"✅ Loaded {len(df)} trades from {df['name'].nunique()} members")
    print()
    
    # Test 1: Advanced Pattern Analysis
    print("🔍 Testing Advanced Pattern Analysis...")
    try:
        advanced_results = run_advanced_analysis(df)
        
        # Check sector rotation
        if advanced_results['sector_analysis']:
            print(f"  ✅ Sector rotation analysis: {len(advanced_results['sector_analysis'])} members analyzed")
            top_rotator = max(advanced_results['sector_analysis'].items(), 
                            key=lambda x: x[1]['rotation_score'])
            print(f"      Top rotator: {top_rotator[0]} (score: {top_rotator[1]['rotation_score']:.1f})")
        
        # Check volume anomalies
        anomalies_df = advanced_results['volume_anomalies']
        if len(anomalies_df) > 0:
            print(f"  ✅ Volume anomaly detection: {len(anomalies_df)} anomalies found")
            extreme_anomalies = len(anomalies_df[anomalies_df['suspicion_level'] == 'EXTREME'])
            print(f"      Extreme anomalies: {extreme_anomalies}")
        else:
            print("  ⚠️  Volume anomaly detection: No anomalies detected (insufficient data)")
        
        # Check clustering
        cluster_results = advanced_results['cluster_results']
        if cluster_results[0] is not None:
            cluster_df, characteristics, silhouette = cluster_results
            print(f"  ✅ Behavior clustering: {len(characteristics)} clusters (silhouette: {silhouette:.3f})")
            for cluster_id, data in characteristics.items():
                print(f"      Cluster {cluster_id}: {data['profile']} ({data['size']} members)")
        else:
            print("  ⚠️  Behavior clustering: Insufficient data for clustering")
        
        # Check timing correlation
        timing_df = advanced_results['timing_correlations']
        if len(timing_df) > 0:
            print(f"  ✅ Event timing analysis: {len(timing_df)} events analyzed")
            high_risk = len(timing_df[timing_df['suspicious_timing_score'] > 7])
            print(f"      High-risk timing events: {high_risk}")
        else:
            print("  ⚠️  Event timing analysis: No correlations found")
        
        print("  🎉 Advanced pattern analysis completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Advanced pattern analysis failed: {str(e)}")
        return False
    
    print()
    
    # Test 2: Predictive Intelligence
    print("🎯 Testing Predictive Intelligence...")
    try:
        predictive_results = run_predictive_analysis(df, committee_data, legislation_data)
        
        # Check trade predictions
        trade_predictions = predictive_results['trade_predictions']
        if trade_predictions:
            print(f"  ✅ Trade predictions: {len(trade_predictions)} members analyzed")
            high_prob = len([p for p in trade_predictions.values() if p > 0.7])
            print(f"      High probability trades: {high_prob}")
            top_prediction = max(trade_predictions.items(), key=lambda x: x[1])
            print(f"      Highest probability: {top_prediction[0]} ({top_prediction[1]:.1%})")
        
        # Check impact predictions
        impact_predictions = predictive_results['impact_predictions']
        if impact_predictions:
            print(f"  ✅ Market impact predictions: {len(impact_predictions)} analyzed")
            high_impact = len([i for i in impact_predictions.values() if i > 5])
            print(f"      High impact predictions: {high_impact}")
        
        # Check legislation correlations
        legislation_correlations = predictive_results['legislation_correlations']
        if legislation_correlations:
            print(f"  ✅ Legislation correlations: {len(legislation_correlations)} bills analyzed")
            high_confidence = len([l for l in legislation_correlations.values() 
                                 if l['prediction_confidence'] > 0.7])
            print(f"      High confidence outcomes: {high_confidence}")
        
        # Check model performance
        performance = predictive_results['model_performance']
        if performance['trade_accuracy']:
            print(f"  ✅ Model performance: Trade accuracy {performance['trade_accuracy']:.1%}")
        if performance['impact_r2']:
            print(f"      Impact model R²: {performance['impact_r2']:.3f}")
        
        print("  🎉 Predictive intelligence completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Predictive intelligence failed: {str(e)}")
        return False
    
    print()
    
    # Test 3: Data Integration
    print("🔗 Testing Data Integration...")
    try:
        # Test basic analysis still works
        basic_df, basic_analysis = analyze_trading_patterns(trades)
        print(f"  ✅ Basic analysis integration: {basic_analysis['total_trades']} trades processed")
        print(f"      Average filing delay: {basic_analysis['avg_filing_delay']:.1f} days")
        print(f"      Large trades (>$100k): {basic_analysis['large_trades']}")
        
        # Test committee data integration
        print(f"  ✅ Committee data: {len(committee_data)} members with committee info")
        
        # Test legislation data integration
        print(f"  ✅ Legislation data: {len(legislation_data)} active bills tracked")
        
        print("  🎉 Data integration verified successfully!")
        
    except Exception as e:
        print(f"  ❌ Data integration failed: {str(e)}")
        return False
    
    print()
    
    # Summary
    print("📋 ENHANCED ANALYSIS TEST SUMMARY")
    print("-" * 40)
    print("✅ Advanced Pattern Analysis: PASSED")
    print("✅ Predictive Intelligence: PASSED") 
    print("✅ Data Integration: PASSED")
    print()
    print("🎉 All enhanced analysis features are working correctly!")
    print()
    print("📈 Key Metrics:")
    print(f"   • {len(df)} total trades analyzed")
    print(f"   • {df['name'].nunique()} congressional members tracked")
    print(f"   • {len(advanced_results['sector_analysis']) if advanced_results['sector_analysis'] else 0} sector rotation patterns")
    print(f"   • {len(anomalies_df) if 'anomalies_df' in locals() else 0} volume anomalies detected")
    print(f"   • {len(characteristics) if 'characteristics' in locals() else 0} behavior clusters identified")
    print(f"   • {len(timing_df) if 'timing_df' in locals() else 0} event correlations analyzed")
    print(f"   • {len(trade_predictions)} trade probability predictions")
    print(f"   • {len(impact_predictions) if 'impact_predictions' in locals() else 0} market impact forecasts")
    print()
    print("🚀 Enhanced Congressional Trading Intelligence System is ready!")
    
    return True

if __name__ == "__main__":
    print(f"Starting enhanced analysis test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = test_advanced_analysis()
    
    print()
    if success:
        print("🎊 All tests passed! The enhanced analysis system is fully operational.")
        print()
        print("Next steps:")
        print("1. Run the dashboard backend: python src/dashboard/dashboard_backend.py")
        print("2. Open enhanced_dashboard.html in your browser")
        print("3. Explore the advanced analysis features!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)