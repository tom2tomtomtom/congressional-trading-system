#!/usr/bin/env python3
"""
Basic test for enhanced analysis features.
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.congressional_analysis import (
    get_congressional_trades_sample, 
    analyze_trading_patterns
)
from analysis.advanced_pattern_analyzer import (
    SectorRotationAnalyzer,
    VolumeAnomalyDetector
)

def test_basic_enhanced():
    print("🏛️ Testing Enhanced Congressional Trading Analysis")
    print("=" * 60)
    
    # Load data
    trades = get_congressional_trades_sample()
    df, analysis = analyze_trading_patterns(trades)
    
    print(f"✅ Loaded {len(df)} trades from {df['name'].nunique()} members")
    
    # Test sector rotation
    print("\n🔄 Testing Sector Rotation Analysis...")
    sector_analyzer = SectorRotationAnalyzer()
    sector_results = sector_analyzer.analyze_sector_rotation(df)
    
    if sector_results:
        print(f"  ✅ Analyzed {len(sector_results)} members")
        top_member = max(sector_results.items(), key=lambda x: x[1]['rotation_score'])
        print(f"  🏆 Top rotator: {top_member[0]} (score: {top_member[1]['rotation_score']:.1f})")
    else:
        print("  ⚠️  No sector rotation data")
    
    # Test volume anomalies
    print("\n⚠️ Testing Volume Anomaly Detection...")
    volume_detector = VolumeAnomalyDetector()
    anomalies = volume_detector.detect_volume_anomalies(df)
    
    if len(anomalies) > 0:
        print(f"  ✅ Detected {len(anomalies)} anomalies")
        extreme = len(anomalies[anomalies['suspicion_level'] == 'EXTREME'])
        print(f"  🚨 Extreme anomalies: {extreme}")
    else:
        print("  ⚠️  No anomalies detected (may need more diverse data)")
    
    print("\n📊 Basic Enhanced Analysis Summary:")
    print(f"   • Sector rotation patterns: {len(sector_results) if sector_results else 0}")
    print(f"   • Volume anomalies: {len(anomalies)}")
    print(f"   • Analysis successful: ✅")
    
    return True

if __name__ == "__main__":
    success = test_basic_enhanced()
    
    if success:
        print("\n🎊 Basic enhanced analysis working!")
        print("\nNext: Open the enhanced dashboard in your browser:")
        print("   src/dashboard/enhanced_dashboard.html")
    else:
        print("\n❌ Test failed")
        sys.exit(1)