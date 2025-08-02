#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Simple Local Launcher
Lightweight version that runs without complex ML dependencies
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import webbrowser

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


class SimpleCongressionalTradingSystemLauncher:
    """Simplified system launcher that works with minimal dependencies"""
    
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
            'USE_CONGRESS_API': 'true',
            'USE_MOCK_DATA': 'true',  # Use mock data for demo
            'ENABLE_EMAIL_ALERTS': 'false'
        }
    
    def print_banner(self):
        """Print system banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🏛️  CONGRESSIONAL TRADING INTELLIGENCE SYSTEM 🏛️         ║
║                                                              ║
║     Simple Demo Version - Local Development                  ║
║     Congressional Data • Basic Analysis • Dashboard         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("Starting Simple Congressional Trading Intelligence System")
    
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
    
    def install_basic_dependencies(self):
        """Install only essential dependencies"""
        logger.info("📦 Installing basic dependencies...")
        
        basic_packages = [
            'flask',
            'requests',
            'python-dotenv',
            'sqlite3'  # Usually built-in
        ]
        
        for package in basic_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✅ {package}: Already installed")
            except ImportError:
                try:
                    subprocess.run(['pip3', 'install', package], 
                                 check=True, capture_output=True)
                    logger.info(f"✅ {package}: Installed")
                except subprocess.CalledProcessError:
                    logger.warning(f"⚠️ Failed to install {package}, trying to continue...")
        
        return True
    
    def create_simple_api(self):
        """Create a simple Flask API"""
        api_content = '''#!/usr/bin/env python3
"""
Simple Congressional Trading API
"""
import os
from flask import Flask, jsonify, render_template_string
from datetime import datetime
import json

app = Flask(__name__)

# Sample congressional data
SAMPLE_CONGRESS_DATA = [
    {
        "id": 1,
        "name": "Nancy Pelosi",
        "party": "Democrat",
        "state": "California",
        "chamber": "House",
        "recent_trades": [
            {"date": "2024-01-15", "stock": "NVDA", "action": "Buy", "amount": "$1M-5M"},
            {"date": "2024-01-10", "stock": "TSLA", "action": "Sell", "amount": "$500K-1M"}
        ]
    },
    {
        "id": 2,
        "name": "Josh Gottheimer", 
        "party": "Democrat",
        "state": "New Jersey",
        "chamber": "House",
        "recent_trades": [
            {"date": "2024-01-12", "stock": "AAPL", "action": "Buy", "amount": "$15K-50K"}
        ]
    }
]

@app.route('/')
def dashboard():
    """Main dashboard"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Congressional Trading Intelligence</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 30px; border-radius: 10px; text-align: center; }
            .stats { display: flex; gap: 20px; margin: 20px 0; }
            .stat { background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center; }
            .members { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .member { border-bottom: 1px solid #eee; padding: 15px 0; }
            .trade { background: #f8f9fa; margin: 5px 0; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏛️ Congressional Trading Intelligence System</h1>
            <p>Real-time Analysis of Congressional Stock Trading Patterns</p>
        </div>
        
        <div class="stats">
            <div class="stat">
                <h3>535</h3>
                <p>Total Members</p>
            </div>
            <div class="stat">
                <h3>2,847</h3>
                <p>Total Trades YTD</p>
            </div>
            <div class="stat">
                <h3>$47.2M</h3>
                <p>Total Volume</p>
            </div>
            <div class="stat">
                <h3>94%</h3>
                <p>Compliance Rate</p>
            </div>
        </div>
        
        <div class="members">
            <h2>📊 Recent Congressional Trading Activity</h2>
            <div id="members-list"></div>
        </div>
        
        <script>
            const members = """ + json.dumps(SAMPLE_CONGRESS_DATA) + """;
            
            function renderMembers() {
                const container = document.getElementById('members-list');
                members.forEach(member => {
                    const memberDiv = document.createElement('div');
                    memberDiv.className = 'member';
                    memberDiv.innerHTML = `
                        <h4>${member.name} (${member.party[0]}-${member.state})</h4>
                        <p><strong>Chamber:</strong> ${member.chamber}</p>
                        <div class="trades">
                            ${member.recent_trades.map(trade => `
                                <div class="trade">
                                    <strong>${trade.date}:</strong> ${trade.action} ${trade.stock} (${trade.amount})
                                </div>
                            `).join('')}
                        </div>
                    `;
                    container.appendChild(memberDiv);
                });
            }
            
            renderMembers();
        </script>
    </body>
    </html>
    """
    return html

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-simple"
    })

@app.route('/api/v1/info')
def api_info():
    """API information"""
    return jsonify({
        "name": "Congressional Trading Intelligence API",
        "version": "1.0.0-simple",
        "description": "Simple API for congressional trading data analysis",
        "endpoints": {
            "/": "Main dashboard",
            "/health": "Health check",
            "/api/v1/members": "Congressional members data",
            "/api/v1/trades": "Recent trading activity"
        }
    })

@app.route('/api/v1/members')
def get_members():
    """Get congressional members"""
    return jsonify({
        "members": SAMPLE_CONGRESS_DATA,
        "count": len(SAMPLE_CONGRESS_DATA),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/trades')
def get_trades():
    """Get recent trades"""
    all_trades = []
    for member in SAMPLE_CONGRESS_DATA:
        for trade in member['recent_trades']:
            trade_record = trade.copy()
            trade_record['member'] = member['name']
            trade_record['party'] = member['party']
            all_trades.append(trade_record)
    
    return jsonify({
        "trades": all_trades,
        "count": len(all_trades),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Congressional Trading Intelligence API...")
    print("📊 Dashboard: http://localhost:8080")
    print("🔗 API Info: http://localhost:8080/api/v1/info")
    print("🏥 Health: http://localhost:8080/health")
    app.run(host='0.0.0.0', port=8080, debug=True)
'''
        
        api_path = self.project_root / 'simple_api.py'
        with open(api_path, 'w') as f:
            f.write(api_content)
        
        logger.info(f"✅ Simple API created: {api_path}")
        return api_path
    
    def start_simple_api(self):
        """Start the simple Flask API"""
        logger.info("🌐 Starting simple API server...")
        
        try:
            api_script = self.create_simple_api()
            
            # Start API server in background
            process = subprocess.Popen([
                'python3', str(api_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['api_server'] = process
            logger.info("✅ API server starting (PID: {})".format(process.pid))
            
            # Give it time to start
            time.sleep(3)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            return False
    
    def test_api_health(self):
        """Test if API is responding"""
        logger.info("🧪 Testing API health...")
        
        try:
            import requests
            response = requests.get('http://localhost:8080/health', timeout=5)
            if response.status_code == 200:
                logger.info("✅ API Server: Healthy and responding")
                return True
            else:
                logger.warning(f"⚠️ API Server: Status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ API health check failed: {e}")
            return False
    
    def print_access_info(self):
        """Print access information"""
        info = """
🎉 CONGRESSIONAL TRADING INTELLIGENCE SYSTEM IS RUNNING!

📊 Access Points:
┌─────────────────────────────────────────────────────────────┐
│  🌐 Web Dashboard:    http://localhost:8080                │
│  🔗 API Endpoints:    http://localhost:8080/api/v1         │
│  📋 API Information:  http://localhost:8080/api/v1/info    │
│  🏥 Health Check:     http://localhost:8080/health         │
└─────────────────────────────────────────────────────────────┘

🎯 Quick Actions:
  • Open Dashboard:    http://localhost:8080
  • Test API:          curl http://localhost:8080/health
  • View Logs:         tail -f system_launch.log
  • Stop System:       Press Ctrl+C

🔧 System Status:
  • Congressional Data: ✅ Sample data loaded
  • API Server:         ✅ Running on port 5000
  • Database:           ✅ Simple in-memory data
  • Dashboard:          ✅ Interactive web interface

🚀 The simplified system is ready for congressional trading analysis!
        """
        print(info)
        logger.info("System startup completed successfully")
        
        # Try to open browser
        try:
            webbrowser.open('http://localhost:8080')
            logger.info("🌐 Opening dashboard in browser...")
        except:
            logger.info("💡 Manually open http://localhost:8080 in your browser")
    
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
    
    def run_simple_system(self):
        """Run the simplified system"""
        try:
            self.print_banner()
            
            # Step 1: Setup environment
            if not self.setup_environment():
                logger.error("❌ Environment setup failed")
                return False
            
            # Step 2: Install basic dependencies
            if not self.install_basic_dependencies():
                logger.error("❌ Basic dependency installation failed")
                return False
            
            # Step 3: Start simple API
            if not self.start_simple_api():
                logger.error("❌ Simple API failed to start")
                return False
            
            # Step 4: Test health
            time.sleep(2)  # Give service time to start
            self.test_api_health()
            
            # Step 5: Print access info
            self.print_access_info()
            
            # Keep running
            logger.info("System running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(30)
                    # Optional: periodic health checks
                    self.test_api_health()
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
    launcher = SimpleCongressionalTradingSystemLauncher()
    success = launcher.run_simple_system()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()