#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Enhanced Launch Script
Comprehensive launcher for the complete data integration system
"""

import os
import sys
import json
import logging
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSystemLauncher:
    """Complete launcher for Congressional Trading Intelligence System"""
    
    def __init__(self):
        self.system_components = {
            'data_analyzer': {
                'script': 'src/processing/data_analyzer.py',
                'description': 'Comprehensive data analysis engine',
                'required': True
            },
            'ml_engine': {
                'script': 'src/ml_models/prediction_engine.py',
                'description': 'Machine learning prediction models',
                'required': True
            },
            'statistical_analyzer': {
                'script': 'src/analysis/statistical_analyzer.py',
                'description': 'Advanced statistical analysis',
                'required': True
            },
            'enhanced_backend': {
                'script': 'enhanced_backend.py',
                'description': 'Enhanced API backend with comprehensive endpoints',
                'required': True,
                'port': 5000
            },
            'data_updater': {
                'script': 'src/data_pipeline/automated_updater.py',
                'description': 'Automated data update system',
                'required': False
            }
        }
        
        self.system_status = {}
        self.processes = {}
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed"""
        logger.info("🔍 Checking system dependencies...")
        
        required_packages = [
            'flask', 'flask-cors', 'pandas', 'numpy', 'scikit-learn',
            'xgboost', 'scipy', 'statsmodels', 'requests', 'schedule'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"  ✅ {package}")
            except ImportError:
                logger.error(f"  ❌ {package} - MISSING")
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("💡 Install missing packages with:")
            logger.info(f"   pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("✅ All dependencies satisfied")
        return True
    
    def check_data_files(self) -> Dict[str, bool]:
        """Check if required data files exist"""
        logger.info("📂 Checking data files...")
        
        required_files = [
            'src/data/congressional_members_full.json',
            'src/data/congressional_trades_full.json'
        ]
        
        file_status = {}
        
        for file_path in required_files:
            exists = os.path.exists(file_path)
            file_status[file_path] = exists
            
            if exists:
                # Check file size
                size = os.path.getsize(file_path)
                logger.info(f"  ✅ {file_path} ({size:,} bytes)")
            else:
                logger.warning(f"  ⚠️ {file_path} - NOT FOUND")
        
        return file_status
    
    def run_initial_analysis(self) -> Dict[str, Any]:
        """Run initial data analysis to warm up the system"""
        logger.info("🔄 Running initial system analysis...")
        
        results = {}
        
        try:
            # Run data analyzer
            logger.info("📊 Running data analysis...")
            from src.processing.data_analyzer import CongressionalTradingAnalyzer
            
            analyzer = CongressionalTradingAnalyzer()
            analysis_results = analyzer.run_comprehensive_analysis()
            
            if analysis_results:
                results['data_analysis'] = {
                    'status': 'success',
                    'total_members': analysis_results.get('metadata', {}).get('total_members', 0),
                    'total_trades': analysis_results.get('metadata', {}).get('total_trades', 0)
                }
                logger.info(f"  ✅ Analyzed {results['data_analysis']['total_trades']} trades from {results['data_analysis']['total_members']} members")
            else:
                results['data_analysis'] = {'status': 'failed', 'error': 'No analysis results'}
                logger.error("  ❌ Data analysis failed")
        
        except Exception as e:
            results['data_analysis'] = {'status': 'error', 'error': str(e)}
            logger.error(f"  ❌ Data analysis error: {e}")
        
        try:
            # Run ML training
            logger.info("🤖 Training ML models...")
            from src.ml_models.prediction_engine import TradingPredictionEngine
            
            ml_engine = TradingPredictionEngine()
            training_results = ml_engine.run_full_training_pipeline()
            
            if 'error' not in training_results:
                results['ml_training'] = {
                    'status': 'success',
                    'models_trained': len(training_results.get('saved_models', {}))
                }
                logger.info(f"  ✅ Trained {results['ml_training']['models_trained']} ML models")
            else:
                results['ml_training'] = {'status': 'failed', 'error': training_results.get('error')}
                logger.error(f"  ❌ ML training failed: {training_results.get('error')}")
        
        except Exception as e:
            results['ml_training'] = {'status': 'error', 'error': str(e)}
            logger.error(f"  ❌ ML training error: {e}")
        
        try:
            # Run statistical analysis
            logger.info("📈 Running statistical analysis...")
            from src.analysis.statistical_analyzer import CongressionalStatisticalAnalyzer
            
            stat_analyzer = CongressionalStatisticalAnalyzer()
            stat_results = stat_analyzer.run_comprehensive_statistical_analysis()
            
            if 'error' not in stat_results:
                results['statistical_analysis'] = {
                    'status': 'success',
                    'insights_count': len(stat_results.get('key_insights', []))
                }
                logger.info(f"  ✅ Generated {results['statistical_analysis']['insights_count']} statistical insights")
            else:
                results['statistical_analysis'] = {'status': 'failed', 'error': stat_results.get('error')}
                logger.error(f"  ❌ Statistical analysis failed: {stat_results.get('error')}")
        
        except Exception as e:
            results['statistical_analysis'] = {'status': 'error', 'error': str(e)}
            logger.error(f"  ❌ Statistical analysis error: {e}")
        
        return results
    
    def start_enhanced_backend(self) -> bool:
        """Start the enhanced backend server"""
        logger.info("🚀 Starting enhanced backend server...")
        
        try:
            # Check if enhanced_backend.py exists
            backend_script = 'enhanced_backend.py'
            if not os.path.exists(backend_script):
                logger.error(f"❌ Backend script not found: {backend_script}")
                return False
            
            # Start the backend process
            process = subprocess.Popen([
                sys.executable, backend_script
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['enhanced_backend'] = process
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info("  ✅ Enhanced backend server started successfully")
                logger.info("  🌐 Backend available at: http://localhost:5000")
                logger.info("  📊 API endpoints available at: http://localhost:5000/api/v1/")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"  ❌ Backend failed to start")
                logger.error(f"  Error: {stderr.decode()}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def display_system_status(self, analysis_results: Dict[str, Any]) -> None:
        """Display comprehensive system status"""
        print("\n" + "="*80)
        print("🏛️  CONGRESSIONAL TRADING INTELLIGENCE SYSTEM - ENHANCED")
        print("="*80)
        
        print(f"🕐 Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System Components Status
        print(f"\n📊 SYSTEM COMPONENTS:")
        
        # Data Analysis
        data_status = analysis_results.get('data_analysis', {})
        if data_status.get('status') == 'success':
            print(f"  ✅ Data Analysis Engine")
            print(f"     📈 {data_status.get('total_trades', 0)} trades analyzed")
            print(f"     👥 {data_status.get('total_members', 0)} members tracked")
        else:
            print(f"  ❌ Data Analysis Engine - {data_status.get('error', 'Failed')}")
        
        # ML Models
        ml_status = analysis_results.get('ml_training', {})
        if ml_status.get('status') == 'success':
            print(f"  ✅ Machine Learning Engine")
            print(f"     🤖 {ml_status.get('models_trained', 0)} models trained")
            print(f"     🎯 Risk prediction, anomaly detection, clustering ready")
        else:
            print(f"  ❌ Machine Learning Engine - {ml_status.get('error', 'Failed')}")
        
        # Statistical Analysis
        stat_status = analysis_results.get('statistical_analysis', {})
        if stat_status.get('status') == 'success':
            print(f"  ✅ Statistical Analysis Engine")
            print(f"     📊 {stat_status.get('insights_count', 0)} statistical insights generated")
            print(f"     🔬 Hypothesis testing, correlation analysis ready")
        else:
            print(f"  ❌ Statistical Analysis Engine - {stat_status.get('error', 'Failed')}")
        
        # Backend Server
        if 'enhanced_backend' in self.processes:
            print(f"  ✅ Enhanced Backend Server")
            print(f"     🌐 API server running on http://localhost:5000")
            print(f"     🔗 Comprehensive endpoints with ML integration")
        else:
            print(f"  ❌ Enhanced Backend Server - Not started")
        
        # Available Features
        print(f"\n🎯 AVAILABLE FEATURES:")
        print(f"  • Enhanced REST API with comprehensive endpoints")
        print(f"  • Real-time ML predictions and risk scoring")
        print(f"  • Advanced statistical analysis and hypothesis testing")
        print(f"  • Anomaly detection with multiple algorithms")
        print(f"  • Data export in CSV/JSON formats")
        print(f"  • Performance optimized with caching")
        print(f"  • Comprehensive error handling and validation")
        
        # API Endpoints
        print(f"\n🔗 KEY API ENDPOINTS:")
        base_url = "http://localhost:5000"
        endpoints = [
            ("/api/v1/stats", "Enhanced system statistics"),
            ("/api/v1/members", "Congressional members with risk scores"),
            ("/api/v1/trades", "Trading data with advanced filtering"),
            ("/api/v1/anomalies", "ML-powered anomaly detection"),
            ("/api/v1/predictions", "ML predictions and forecasting"),
            ("/api/v1/export/csv", "Data export in CSV format"),
            ("/api/v1/export/json", "Data export in JSON format"),
            ("/health", "System health and status check")
        ]
        
        for endpoint, description in endpoints:
            print(f"  • {base_url}{endpoint}")
            print(f"    {description}")
        
        # Dashboard Information
        print(f"\n📊 DASHBOARD ACCESS:")
        print(f"  • Main Dashboard: https://apextrading.up.railway.app/dashboard")
        print(f"  • Local Development: http://localhost:5000/dashboard/comprehensive")
        print(f"  • Analysis Page: http://localhost:5000/analysis")
        
        print("="*80)
        print("🚀 SYSTEM READY - Congressional Trading Intelligence Enhanced")
        print("💡 Access the API documentation and test endpoints above")
        print("🔧 Use Ctrl+C to shutdown the system")
        print("="*80)
    
    def launch_system(self) -> bool:
        """Launch the complete enhanced system"""
        logger.info("🏛️ Launching Congressional Trading Intelligence System - Enhanced")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ System launch aborted due to missing dependencies")
            return False
        
        # Step 2: Check data files
        data_files = self.check_data_files()
        missing_files = [f for f, exists in data_files.items() if not exists]
        
        if missing_files:
            logger.warning(f"⚠️ Missing data files: {', '.join(missing_files)}")
            logger.info("💡 The system will run with available data or sample data")
        
        # Step 3: Run initial analysis
        analysis_results = self.run_initial_analysis()
        
        # Step 4: Start backend server
        if not self.start_enhanced_backend():
            logger.error("❌ Failed to start backend server")
            return False
        
        # Step 5: Display system status
        self.display_system_status(analysis_results)
        
        return True
    
    def shutdown_system(self) -> None:
        """Gracefully shutdown all system components"""
        logger.info("🛑 Shutting down Congressional Trading Intelligence System...")
        
        for component, process in self.processes.items():
            try:
                logger.info(f"  Stopping {component}...")
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"  ✅ {component} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"  ⚠️ Force killing {component}...")
                process.kill()
            except Exception as e:
                logger.error(f"  ❌ Error stopping {component}: {e}")
        
        logger.info("✅ System shutdown complete")
    
    def run_interactive_mode(self) -> None:
        """Run the system in interactive mode"""
        try:
            if not self.launch_system():
                return
            
            # Keep the system running
            print("\n💡 System is running. Available commands:")
            print("  - Press 's' + Enter for system status")
            print("  - Press 'h' + Enter for health check")
            print("  - Press 'q' + Enter or Ctrl+C to quit")
            
            while True:
                try:
                    command = input("\nCommand: ").strip().lower()
                    
                    if command == 'q' or command == 'quit':
                        break
                    elif command == 's' or command == 'status':
                        self._show_runtime_status()
                    elif command == 'h' or command == 'health':
                        self._run_health_check()
                    elif command == 'help':
                        print("Available commands: status (s), health (h), quit (q)")
                    else:
                        print("Unknown command. Type 'help' for available commands.")
                
                except EOFError:
                    break
        
        except KeyboardInterrupt:
            print("\n")
            logger.info("🛑 Received shutdown signal")
        
        finally:
            self.shutdown_system()
    
    def _show_runtime_status(self) -> None:
        """Show current runtime status"""
        print("\n📊 RUNTIME STATUS:")
        
        for component, process in self.processes.items():
            if process.poll() is None:
                print(f"  ✅ {component} - Running (PID: {process.pid})")
            else:
                print(f"  ❌ {component} - Stopped")
        
        # Check API availability
        try:
            import requests
            response = requests.get('http://localhost:5000/health', timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"  🌐 API Health: {health_data.get('status', 'Unknown')}")
                print(f"  📊 Data: {health_data.get('data_status', {}).get('trades_count', 0)} trades, {health_data.get('data_status', {}).get('members_count', 0)} members")
            else:
                print(f"  ⚠️ API Health: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ API Health: Connection failed - {e}")
    
    def _run_health_check(self) -> None:
        """Run comprehensive health check"""
        print("\n🔍 HEALTH CHECK:")
        
        # Check processes
        all_healthy = True
        for component, process in self.processes.items():
            if process.poll() is None:
                print(f"  ✅ {component} process healthy")
            else:
                print(f"  ❌ {component} process stopped")
                all_healthy = False
        
        # Check API endpoints
        try:
            import requests
            
            endpoints_to_check = [
                '/health',
                '/api/v1/stats',
                '/api/v1/members',
                '/api/v1/trades'
            ]
            
            for endpoint in endpoints_to_check:
                try:
                    response = requests.get(f'http://localhost:5000{endpoint}', timeout=5)
                    if response.status_code == 200:
                        print(f"  ✅ {endpoint} - OK")
                    else:
                        print(f"  ⚠️ {endpoint} - HTTP {response.status_code}")
                        all_healthy = False
                except Exception as e:
                    print(f"  ❌ {endpoint} - {e}")
                    all_healthy = False
        
        except ImportError:
            print("  ⚠️ Cannot check API endpoints (requests not available)")
        
        # Overall status
        if all_healthy:
            print("  🎉 Overall System Health: EXCELLENT")
        else:
            print("  ⚠️ Overall System Health: NEEDS ATTENTION")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Congressional Trading Intelligence System - Enhanced Launcher'
    )
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'server', 'analysis'], 
        default='interactive',
        help='Launch mode (default: interactive)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Server port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Create launcher instance
    launcher = EnhancedSystemLauncher()
    
    if args.mode == 'interactive':
        launcher.run_interactive_mode()
    elif args.mode == 'server':
        if launcher.launch_system():
            try:
                print("🚀 Server mode - press Ctrl+C to stop")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n")
                launcher.shutdown_system()
    elif args.mode == 'analysis':
        # Run analysis only
        launcher.check_dependencies()
        results = launcher.run_initial_analysis()
        
        print("\n📊 ANALYSIS RESULTS:")
        for component, result in results.items():
            print(f"  {component}: {result.get('status', 'unknown')}")
            if result.get('error'):
                print(f"    Error: {result['error']}")

if __name__ == '__main__':
    main()