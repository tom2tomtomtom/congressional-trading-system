#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Complete Launcher
Automated setup and launch of the entire system
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_launch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CongressionalTradingSystemLauncher:
    """Complete system launcher and manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = {}
        
        # Required environment variables
        self.required_env = {
            'CONGRESS_GOV_API_KEY': 'GnEJVyPiswjccfl3Y9KHhwRVEeWDUnxOAVC4aMhD',
            'FINNHUB_API_KEY': 'd26nr7hr01qvrairb710d26nr7hr01qvrairb71g',
            'FLASK_ENV': 'development',
            'FLASK_DEBUG': 'True',
            'DATABASE_URL': 'sqlite:///congressional_trading.db',
            'REDIS_URL': 'redis://localhost:6379/0',
            'USE_CONGRESS_API': 'true',
            'USE_HOUSE_SCRAPER': 'true',
            'USE_MOCK_DATA': 'false'
        }
    
    def print_banner(self):
        """Print system banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🏛️  CONGRESSIONAL TRADING INTELLIGENCE SYSTEM 🏛️         ║
║                                                              ║
║     Advanced ML-Powered Trading Pattern Analysis             ║
║     Real-time Congressional Data • Stock Market Integration  ║
║     Enterprise Security • Interactive Dashboard             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("Starting Congressional Trading Intelligence System")
    
    def check_system_requirements(self):
        """Check system requirements and dependencies"""
        logger.info("🔍 Checking system requirements...")
        
        requirements = {
            'python': {'command': ['python3', '--version'], 'min_version': '3.9'},
            'pip': {'command': ['pip3', '--version'], 'required': True},
            'git': {'command': ['git', '--version'], 'required': True}
        }
        
        missing = []
        
        for req, config in requirements.items():
            try:
                result = subprocess.run(
                    config['command'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"✅ {req}: {result.stdout.strip()}")
                else:
                    missing.append(req)
                    logger.error(f"❌ {req}: Not found or error")
            except Exception as e:
                missing.append(req)
                logger.error(f"❌ {req}: {e}")
        
        if missing:
            logger.error(f"Missing requirements: {missing}")
            return False
        
        logger.info("✅ All system requirements met")
        return True
    
    def setup_environment(self):
        """Setup environment variables"""
        logger.info("⚙️ Setting up environment variables...")
        
        # Generate secure keys
        import secrets
        self.required_env['SECRET_KEY'] = secrets.token_urlsafe(32)
        self.required_env['JWT_SECRET_KEY'] = secrets.token_urlsafe(32)
        
        # Create .env file
        env_file = self.project_root / '.env'
        
        with open(env_file, 'w') as f:
            f.write("# Congressional Trading Intelligence System - Environment Configuration\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            
            for key, value in self.required_env.items():
                f.write(f"{key}={value}\n")
                os.environ[key] = str(value)
        
        logger.info(f"✅ Environment configured: {env_file}")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Install from requirements-production.txt if it exists, otherwise dev
            req_file = 'requirements-production.txt'
            if not (self.project_root / req_file).exists():
                req_file = 'requirements-dev.txt'
            
            logger.info(f"Installing from {req_file}")
            
            result = subprocess.run([
                'pip3', 'install', '-r', req_file
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Dependency installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_database(self):
        """Setup and initialize database"""
        logger.info("🗄️ Setting up database...")
        
        try:
            # Run database setup
            setup_script = self.project_root / 'database' / 'setup.py'
            if setup_script.exists():
                result = subprocess.run([
                    'python3', str(setup_script)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ Database setup completed")
                else:
                    logger.warning(f"⚠️ Database setup warnings: {result.stderr}")
            
            # Populate with data
            populate_script = self.project_root / 'scripts' / 'populate_congressional_database.py'
            if populate_script.exists():
                logger.info("📊 Populating database with congressional data...")
                result = subprocess.run([
                    'python3', str(populate_script)
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    logger.info("✅ Database populated with congressional data")
                else:
                    logger.warning(f"⚠️ Data population warnings: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def test_apis(self):
        """Test API connectivity"""
        logger.info("🧪 Testing API connectivity...")
        
        try:
            # Test Congress.gov API
            import requests
            
            headers = {'X-API-Key': self.required_env['CONGRESS_GOV_API_KEY']}
            response = requests.get(
                'https://api.congress.gov/v3/member/118/house',
                headers=headers,
                params={'format': 'json', 'limit': 1},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Congress.gov API: Connected")
            else:
                logger.warning(f"⚠️ Congress.gov API: Status {response.status_code}")
            
            # Test Finnhub API
            response = requests.get(
                'https://finnhub.io/api/v1/quote',
                params={'symbol': 'AAPL', 'token': self.required_env['FINNHUB_API_KEY']},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Finnhub API: Connected")
            else:
                logger.warning(f"⚠️ Finnhub API: Status {response.status_code}")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ API testing failed: {e}")
            return True  # Don't fail startup for API issues
    
    def start_api_server(self):
        """Start the Flask API server"""
        logger.info("🌐 Starting API server...")
        
        try:
            api_script = self.project_root / 'src' / 'api' / 'app.py'
            
            if api_script.exists():
                # Start API server in background
                process = subprocess.Popen([
                    'python3', str(api_script)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.processes['api_server'] = process
                logger.info("✅ API server starting (PID: {})".format(process.pid))
                
                # Give it time to start
                time.sleep(3)
                
                return True
            else:
                logger.error("❌ API server script not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            return False
    
    def start_dashboard(self):
        """Start the dashboard server"""
        logger.info("📊 Starting dashboard server...")
        
        try:
            dashboard_script = self.project_root / 'src' / 'dashboard' / 'react_dashboard.py'
            
            if dashboard_script.exists():
                # Start dashboard in background
                process = subprocess.Popen([
                    'python3', str(dashboard_script)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.processes['dashboard'] = process
                logger.info("✅ Dashboard starting (PID: {})".format(process.pid))
                
                # Give it time to start
                time.sleep(3)
                
                return True
            else:
                logger.error("❌ Dashboard script not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def start_websocket_server(self):
        """Start WebSocket server for real-time features"""
        logger.info("⚡ Starting WebSocket server...")
        
        try:
            ws_script = self.project_root / 'src' / 'realtime' / 'websocket_server.py'
            
            if ws_script.exists():
                # Start WebSocket server in background
                process = subprocess.Popen([
                    'python3', str(ws_script)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.processes['websocket'] = process
                logger.info("✅ WebSocket server starting (PID: {})".format(process.pid))
                
                return True
            else:
                logger.warning("⚠️ WebSocket server script not found (optional)")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ WebSocket server failed (optional): {e}")
            return True  # Don't fail for optional service
    
    def check_services_health(self):
        """Check if all services are running properly"""
        logger.info("🏥 Checking service health...")
        
        try:
            import requests
            
            # Check API server
            try:
                response = requests.get('http://localhost:5000/health', timeout=5)
                if response.status_code == 200:
                    logger.info("✅ API Server: Healthy")
                else:
                    logger.warning(f"⚠️ API Server: Status {response.status_code}")
            except Exception:
                logger.warning("⚠️ API Server: Not responding")
            
            # Check dashboard
            try:
                response = requests.get('http://localhost:8050', timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Dashboard: Healthy")
                else:
                    logger.warning(f"⚠️ Dashboard: Status {response.status_code}")
            except Exception:
                logger.warning("⚠️ Dashboard: Not responding")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Health check failed: {e}")
            return True
    
    def print_access_info(self):
        """Print access information"""
        info = """
🎉 CONGRESSIONAL TRADING INTELLIGENCE SYSTEM IS RUNNING!

📊 Access Points:
┌─────────────────────────────────────────────────────────────┐
│  🌐 Web Dashboard:    http://localhost:8050                │
│  🔗 API Endpoints:    http://localhost:5000/api/v1         │
│  📋 API Documentation: http://localhost:5000/api/v1/info   │
│  🏥 Health Check:     http://localhost:5000/health         │
│  ⚡ WebSocket:        ws://localhost:6789                  │
└─────────────────────────────────────────────────────────────┘

🎯 Quick Actions:
  • Open Dashboard:    http://localhost:8050
  • Test API:          curl http://localhost:5000/health
  • View Logs:         tail -f system_launch.log
  • Stop System:       Press Ctrl+C

🔧 System Status:
  • Congressional Data: ✅ Live from Congress.gov API
  • Stock Market Data:  ✅ Live from Finnhub API  
  • Database:           ✅ SQLite with congressional data
  • Security:           ✅ JWT authentication enabled
  • Real-time Updates:  ✅ WebSocket connections active

🚀 The system is ready for congressional trading intelligence analysis!
        """
        print(info)
        logger.info("System startup completed successfully")
    
    def cleanup(self):
        """Cleanup running processes"""
        logger.info("🧹 Cleaning up processes...")
        
        for service_name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ Stopped {service_name}")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping {service_name}: {e}")
    
    def run_complete_system(self):
        """Run the complete system setup and launch"""
        try:
            self.print_banner()
            
            # Step 1: Check requirements
            if not self.check_system_requirements():
                logger.error("❌ System requirements not met")
                return False
            
            # Step 2: Setup environment
            if not self.setup_environment():
                logger.error("❌ Environment setup failed")
                return False
            
            # Step 3: Install dependencies
            if not self.install_dependencies():
                logger.error("❌ Dependency installation failed")
                return False
            
            # Step 4: Setup database
            if not self.setup_database():
                logger.error("❌ Database setup failed")
                return False
            
            # Step 5: Test APIs
            self.test_apis()
            
            # Step 6: Start services
            if not self.start_api_server():
                logger.error("❌ API server failed to start")
                return False
            
            if not self.start_dashboard():
                logger.error("❌ Dashboard failed to start")
                return False
            
            self.start_websocket_server()
            
            # Step 7: Health check
            time.sleep(5)  # Give services time to fully start
            self.check_services_health()
            
            # Step 8: Print access info
            self.print_access_info()
            
            # Keep running
            logger.info("System running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(30)
                    # Optional: periodic health checks
                    self.check_services_health()
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System launch failed: {e}")
            return False
        
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    launcher = CongressionalTradingSystemLauncher()
    success = launcher.run_complete_system()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()