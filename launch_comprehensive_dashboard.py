#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Comprehensive Dashboard Launcher
Launch the full-featured dashboard with detailed explanations and data exploration.
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

def launch_comprehensive_dashboard():
    """Launch the comprehensive dashboard with all features."""
    
    print("🏛️ Congressional Trading Intelligence System - Comprehensive Dashboard")
    print("=" * 75)
    print()
    
    # Check if comprehensive dashboard exists
    dashboard_path = Path("src/dashboard/comprehensive_dashboard.html")
    
    if not dashboard_path.exists():
        print("❌ Comprehensive dashboard not found!")
        print(f"   Expected at: {dashboard_path.absolute()}")
        return False
    
    print("🚀 Launching Comprehensive Analysis Dashboard...")
    print()
    print("📊 Dashboard Features:")
    print()
    
    print("   📋 TAB 1: MEMBER ANALYSIS")
    print("      • Complete list of all 531 congressional members")
    print("      • Individual trading profiles and risk scores")
    print("      • Filing compliance analysis with STOCK Act requirements")
    print("      • Trading volume distribution and patterns")
    print()
    
    print("   🔍 TAB 2: TRADE EXPLORER")
    print("      • Detailed view of all congressional trades")
    print("      • Interactive filters by member, risk level, amount")
    print("      • Trade-by-trade risk assessment and timing analysis")
    print("      • Stock sector analysis and concentration patterns")
    print()
    
    print("   📈 TAB 3: PATTERN ANALYSIS")
    print("      • Advanced statistical pattern recognition")
    print("      • Machine learning behavioral clustering")
    print("      • Volume anomaly detection with Z-score analysis")
    print("      • Market event timing correlation analysis")
    print()
    
    print("   🎯 TAB 4: ML PREDICTIONS")
    print("      • Trade probability forecasting for each member")
    print("      • Market impact predictions when trades are disclosed")
    print("      • Legislation outcome confidence based on trading patterns")
    print("      • Model performance metrics and validation")
    print()
    
    print("   📊 TAB 5: EVENT CORRELATIONS")
    print("      • Trading timeline vs major market events")
    print("      • Statistical correlation strength analysis")
    print("      • COVID-19, AI regulation, and policy timing analysis")
    print()
    
    print("   🏛️ TAB 6: COMMITTEE ANALYSIS")
    print("      • Committee assignment vs trading pattern correlation")
    print("      • Conflict of interest risk assessment")
    print("      • Trading activity by committee type")
    print()
    
    print("   📜 TAB 7: ACTIVE LEGISLATION")
    print("      • Bills currently affecting traded stocks")
    print("      • Legislative timeline and trading volume correlation")
    print("      • Risk assessment for pending legislation")
    print()
    
    print("   📚 TAB 8: RESEARCH TOOLS")
    print("      • Statistical summary and methodology")
    print("      • Data export capabilities (CSV, JSON)")
    print("      • Academic citation information")
    print("      • Research methodology documentation")
    print()
    
    print("✨ ENHANCED FEATURES:")
    print("   🔍 Chart Explanations: Every chart includes detailed explanations")
    print("   📖 Interpretations: Clear insights and what patterns mean")
    print("   ❓ Help Tooltips: Hover over ❓ symbols for context")
    print("   📊 Interactive Filters: Customize views by member, risk, amount")
    print("   📈 Real-time Insights: Live calculations and risk assessments")
    print("   🎨 Professional Design: Publication-quality visualizations")
    print()
    
    # Open dashboard in browser
    dashboard_url = f"file://{dashboard_path.absolute()}"
    
    print(f"🌐 Opening comprehensive dashboard...")
    print(f"   URL: {dashboard_url}")
    print()
    
    try:
        webbrowser.open(dashboard_url)
        print("✅ Comprehensive dashboard opened successfully!")
        print()
        print("📖 HOW TO USE THE DASHBOARD:")
        print()
        print("   1️⃣ START WITH MEMBER ANALYSIS")
        print("      • Review the complete list of congressional members")
        print("      • Check individual risk scores and trading patterns")
        print()
        print("   2️⃣ EXPLORE INDIVIDUAL TRADES")
        print("      • Switch to Trade Explorer tab")
        print("      • Use filters to focus on specific members or risk levels")
        print("      • Read detailed explanations for each trade")
        print()
        print("   3️⃣ UNDERSTAND THE PATTERNS")
        print("      • Pattern Analysis tab shows statistical insights")
        print("      • Each chart includes 'What This Shows' explanations")
        print("      • Look for 'Key Insights' interpretation boxes")
        print()
        print("   4️⃣ CHECK PREDICTIONS")
        print("      • ML Predictions tab shows future trade probabilities")
        print("      • Market impact forecasts for potential trades")
        print()
        print("   5️⃣ ANALYZE CORRELATIONS")
        print("      • Event Correlations shows timing with market events")
        print("      • Committee Analysis reveals potential conflicts")
        print()
        print("🎯 KEY FEATURES TO EXPLORE:")
        print("   • Hover over ❓ symbols for detailed explanations")
        print("   • Read 'Chart Explanation' boxes below each visualization")
        print("   • Check 'Key Insights' for interpretations")
        print("   • Use filters in Trade Explorer to customize views")
        print("   • Export data from Research Tools tab")
        print()
        print("🔍 UNDERSTANDING RISK SCORES:")
        print("   • 0-2: Low Risk (routine investment activity)")
        print("   • 3-5: Medium Risk (requires monitoring)")
        print("   • 6-8: High Risk (potential conflicts of interest)")
        print("   • 9-10: Extreme Risk (requires investigation)")
        print()
        print("📊 INTERPRETING CHARTS:")
        print("   • Red/Orange: High risk or concerning patterns")
        print("   • Blue: Standard or expected patterns")
        print("   • Green: Low risk or compliant behavior")
        print("   • Size often indicates trade volume or impact")
        print()
        print("🎊 Comprehensive Congressional Trading Intelligence System is ready!")
        print("   Explore all 8 tabs for complete analysis capabilities!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to open dashboard: {e}")
        print()
        print("MANUAL LAUNCH:")
        print(f"   Open this file in your browser:")
        print(f"   {dashboard_path.absolute()}")
        return False

if __name__ == "__main__":
    print("Starting Comprehensive Dashboard Launcher...")
    print()
    
    success = launch_comprehensive_dashboard()
    
    if success:
        print()
        print("🌟 Dashboard launched successfully!")
        print("   Enjoy exploring the comprehensive analysis features!")
    else:
        print()
        print("❌ Launch failed. Please try manual launch.")
        sys.exit(1)