#!/usr/bin/env python3
"""
Enhanced Congressional Trading Intelligence System - Dashboard Launcher
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

def launch_enhanced_dashboard():
    """Launch the enhanced dashboard."""
    
    print("🏛️ Congressional Trading Intelligence System - Enhanced Dashboard")
    print("=" * 70)
    print()
    
    # Check if enhanced dashboard exists
    dashboard_path = Path("src/dashboard/enhanced_dashboard.html")
    
    if not dashboard_path.exists():
        print("❌ Enhanced dashboard not found!")
        print(f"   Expected at: {dashboard_path.absolute()}")
        return False
    
    print("🚀 Launching Enhanced Dashboard...")
    print()
    print("📊 Available Analysis Features:")
    print("   ✅ Advanced Pattern Analysis")
    print("      • Sector rotation analysis")
    print("      • Volume anomaly detection") 
    print("      • Behavior clustering")
    print("      • Event timing correlation")
    print()
    print("   🎯 ML Predictions")
    print("      • Trade probability predictions")
    print("      • Market impact forecasting")
    print("      • Legislation outcome analysis")
    print() 
    print("   📈 Enhanced Visualizations")
    print("      • Interactive charts and graphs")
    print("      • Real-time data updates")
    print("      • Export capabilities")
    print()
    print("   🔬 Research Tools")
    print("      • Statistical analysis")
    print("      • Data export (CSV, JSON)")
    print("      • Academic reporting")
    print()
    
    # Open dashboard in browser
    dashboard_url = f"file://{dashboard_path.absolute()}"
    
    print(f"🌐 Opening dashboard at: {dashboard_url}")
    print()
    
    try:
        webbrowser.open(dashboard_url)
        print("✅ Dashboard opened in your default browser!")
        print()
        print("📝 Usage Notes:")
        print("   • Click on different tabs to explore analysis features")
        print("   • Charts will load automatically when you switch tabs")
        print("   • Some features show sample data for demonstration")
        print("   • Real implementation would connect to live data sources")
        print()
        print("🔧 For Dynamic Data (Optional):")
        print("   • Run: python3 src/dashboard/dashboard_backend.py")
        print("   • Then visit: http://localhost:5000")
        print("   • This provides live API-powered data")
        print()
        print("🎊 Enjoy exploring the enhanced Congressional Trading Intelligence System!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to open dashboard: {e}")
        print()
        print("Manual launch:")
        print(f"   Open this file in your browser: {dashboard_path.absolute()}")
        return False

if __name__ == "__main__":
    print("Starting Enhanced Dashboard Launcher...")
    print()
    
    success = launch_enhanced_dashboard()
    
    if not success:
        sys.exit(1)
    
    print()
    print("Dashboard launched successfully! 🎉")