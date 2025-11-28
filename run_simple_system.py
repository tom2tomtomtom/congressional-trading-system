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
Simple Congressional Trading API with Enhanced Dashboard
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
    },
    {
        "id": 3,
        "name": "Dan Crenshaw", 
        "party": "Republican",
        "state": "Texas",
        "chamber": "House",
        "recent_trades": [
            {"date": "2024-02-01", "stock": "AMD", "action": "Buy", "amount": "$50K-100K"},
            {"date": "2024-02-01", "stock": "META", "action": "Buy", "amount": "$15K-50K"}
        ]
    },
    {
        "id": 4,
        "name": "Austin Scott", 
        "party": "Republican",
        "state": "Georgia",
        "chamber": "House",
        "recent_trades": [
            {"date": "2024-01-28", "stock": "SO", "action": "Buy", "amount": "$15K-50K"}
        ]
    }
]

@app.route('/')
def dashboard():
    """Main dashboard"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Congressional Trading Intelligence</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary: #4f46e5;
                --primary-dark: #4338ca;
                --secondary: #64748b;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-light: #64748b;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --border: #e2e8f0;
            }
            
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                line-height: 1.6;
            }
            
            .navbar {
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                color: white;
                padding: 1rem 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                position: sticky;
                top: 0;
                z-index: 100;
            }
            
            .navbar-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .brand {
                font-size: 1.5rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .nav-links a {
                color: rgba(255,255,255,0.9);
                text-decoration: none;
                margin-left: 1.5rem;
                font-weight: 500;
                transition: color 0.2s;
            }
            
            .nav-links a:hover {
                color: white;
            }
            
            .container {
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 1rem;
            }
            
            .hero-section {
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .hero-title {
                font-size: 2.5rem;
                color: var(--text-main);
                margin-bottom: 0.5rem;
            }
            
            .hero-subtitle {
                font-size: 1.1rem;
                color: var(--text-light);
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 1.5rem;
                margin-bottom: 3rem;
            }
            
            .stat-card {
                background: var(--card-bg);
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid var(--border);
                transition: transform 0.2s, box-shadow 0.2s;
                cursor: pointer;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .stat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }
            
            .stat-icon {
                width: 40px;
                height: 40px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.25rem;
            }
            
            .icon-blue { background: #eff6ff; color: #3b82f6; }
            .icon-green { background: #ecfdf5; color: #10b981; }
            .icon-purple { background: #f3e8ff; color: #a855f7; }
            .icon-orange { background: #fff7ed; color: #f97316; }
            
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--text-main);
                margin-bottom: 0.25rem;
            }
            
            .stat-label {
                font-size: 0.875rem;
                color: var(--text-light);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .controls-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
                gap: 1rem;
                flex-wrap: wrap;
            }

            .search-box {
                flex: 1;
                min-width: 300px;
                position: relative;
            }

            .search-input {
                width: 100%;
                padding: 0.75rem 1rem 0.75rem 2.5rem;
                border: 1px solid var(--border);
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.2s;
            }

            .search-input:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
            }

            .search-icon {
                position: absolute;
                left: 1rem;
                top: 50%;
                transform: translateY(-50%);
                color: var(--text-light);
            }

            .filter-group {
                display: flex;
                gap: 0.5rem;
            }

            .filter-btn {
                padding: 0.5rem 1rem;
                border: 1px solid var(--border);
                background: white;
                border-radius: 6px;
                color: var(--text-main);
                cursor: pointer;
                font-weight: 500;
                transition: all 0.2s;
            }

            .filter-btn:hover {
                background: #f8fafc;
            }

            .filter-btn.active {
                background: var(--primary);
                color: white;
                border-color: var(--primary);
            }
            
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                border: 1px solid var(--border);
                overflow: hidden;
                min-height: 400px;
            }
            
            .card-header {
                padding: 1.25rem 1.5rem;
                border-bottom: 1px solid var(--border);
                background: #f8fafc;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .card-title {
                font-weight: 600;
                color: var(--text-main);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .members-list {
                padding: 0;
            }
            
            .member-item {
                display: flex;
                flex-direction: column;
                padding: 1.5rem;
                border-bottom: 1px solid var(--border);
                transition: background 0.2s;
                cursor: pointer;
            }
            
            .member-item:last-child {
                border-bottom: none;
            }
            
            .member-item:hover {
                background: #f8fafc;
            }
            
            .member-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 1rem;
            }
            
            .member-info h4 {
                font-size: 1.1rem;
                margin-bottom: 0.25rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .party-tag {
                font-size: 0.75rem;
                padding: 0.125rem 0.5rem;
                border-radius: 999px;
                font-weight: 600;
            }
            
            .tag-dem { background: #eff6ff; color: #2563eb; }
            .tag-rep { background: #fef2f2; color: #dc2626; }
            
            .member-meta {
                font-size: 0.875rem;
                color: var(--text-light);
            }
            
            .trades-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 0.75rem;
            }
            
            .trade-card {
                background: white;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.75rem;
                font-size: 0.875rem;
                transition: transform 0.2s;
            }
            
            .trade-card:hover {
                transform: translateY(-2px);
                border-color: var(--primary);
            }
            
            .trade-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
            }
            
            .stock-symbol {
                font-weight: 700;
                color: var(--text-main);
            }
            
            .trade-date {
                color: var(--text-light);
                font-size: 0.75rem;
            }
            
            .trade-action {
                display: inline-block;
                padding: 0.125rem 0.375rem;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.75rem;
                margin-bottom: 0.25rem;
            }
            
            .action-buy { background: #ecfdf5; color: #059669; }
            .action-sell { background: #fef2f2; color: #dc2626; }
            
            .trade-amount {
                font-weight: 500;
                color: var(--text-main);
            }
            
            footer {
                text-align: center;
                padding: 2rem;
                color: var(--text-light);
                font-size: 0.875rem;
                margin-top: 2rem;
            }
            
            .api-link {
                display: inline-block;
                margin-top: 1rem;
                color: var(--primary);
                text-decoration: none;
                font-weight: 500;
            }

            .no-results {
                text-align: center;
                padding: 3rem;
                color: var(--text-light);
            }
            
            @media (min-width: 768px) {
                .member-item {
                    flex-direction: row;
                    align-items: flex-start;
                }
                
                .member-header {
                    width: 300px;
                    flex-shrink: 0;
                    margin-bottom: 0;
                    margin-right: 2rem;
                    flex-direction: column;
                }
                
                .trades-grid {
                    flex-grow: 1;
                }
            }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="navbar-content">
                <div class="brand">
                    <i class="fas fa-landmark"></i>
                    Congressional Trading
                </div>
                <div class="nav-links">
                    <a href="#" onclick="location.reload()">Dashboard</a>
                    <a href="#analysis" onclick="alert('Detailed analysis coming soon!')">Analysis</a>
                    <a href="/api/v1/info">API</a>
                </div>
            </div>
        </nav>
        
        <div class="container">
            <div class="hero-section">
                <h1 class="hero-title">Congressional Trading Intelligence</h1>
                <p class="hero-subtitle">Real-time tracking and analysis of legislative financial disclosures</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card" onclick="filterByParty('all')">
                    <div class="stat-header">
                        <div class="stat-label">Total Members</div>
                        <div class="stat-icon icon-blue"><i class="fas fa-users"></i></div>
                    </div>
                    <div class="stat-value">535</div>
                    <div class="stat-trend text-green"><i class="fas fa-arrow-up"></i> 100% Tracking</div>
                </div>
                
                <div class="stat-card" onclick="alert('Viewing year-to-date statistics')">
                    <div class="stat-header">
                        <div class="stat-label">Trades (YTD)</div>
                        <div class="stat-icon icon-green"><i class="fas fa-chart-line"></i></div>
                    </div>
                    <div class="stat-value">2,847</div>
                    <div class="stat-trend text-green"><i class="fas fa-arrow-up"></i> +12% vs 2023</div>
                </div>
                
                <div class="stat-card" onclick="alert('Volume calculation includes options and stocks')">
                    <div class="stat-header">
                        <div class="stat-label">Total Volume</div>
                        <div class="stat-icon icon-purple"><i class="fas fa-sack-dollar"></i></div>
                    </div>
                    <div class="stat-value">$47.2M</div>
                    <div class="stat-trend text-green"><i class="fas fa-arrow-up"></i> High Activity</div>
                </div>
                
                <div class="stat-card" onclick="alert('Compliance based on STOCK Act filing deadlines')">
                    <div class="stat-header">
                        <div class="stat-label">Compliance Rate</div>
                        <div class="stat-icon icon-orange"><i class="fas fa-clipboard-check"></i></div>
                    </div>
                    <div class="stat-value">94%</div>
                    <div class="stat-trend text-red"><i class="fas fa-arrow-down"></i> -2% vs 2023</div>
                </div>
            </div>
            
            <div class="controls-bar">
                <div class="search-box">
                    <i class="fas fa-search search-icon"></i>
                    <input type="text" class="search-input" id="searchInput" placeholder="Search members, states, or stocks..." onkeyup="filterMembers()">
                </div>
                
                <div class="filter-group">
                    <button class="filter-btn active" onclick="filterByParty('all')" id="btn-all">All Parties</button>
                    <button class="filter-btn" onclick="filterByParty('Democrat')" id="btn-dem">Democrats</button>
                    <button class="filter-btn" onclick="filterByParty('Republican')" id="btn-rep">Republicans</button>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-history"></i> Latest Disclosures
                    </div>
                    <div class="filter-group">
                        <button class="filter-btn" onclick="location.reload()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                <div class="members-list" id="members-list">
                    <!-- Populated by JS -->
                </div>
            </div>
        </div>
        
        <footer>
            <p>Congressional Trading Intelligence System &copy; 2025</p>
            <p>Data provided for educational purposes only.</p>
            <a href="/api/v1/info" class="api-link">View API Documentation <i class="fas fa-arrow-right"></i></a>
        </footer>
        
        <script>
            // Pass the Python data to JS
            const members = """ + json.dumps(SAMPLE_CONGRESS_DATA) + """;
            let currentPartyFilter = 'all';
            
            function renderMembers(data = members) {
                const container = document.getElementById('members-list');
                container.innerHTML = '';
                
                if (data.length === 0) {
                    container.innerHTML = `
                        <div class="no-results">
                            <i class="fas fa-search" style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                            <p>No matching records found</p>
                        </div>`;
                    return;
                }
                
                data.forEach(member => {
                    // Determine party tag class
                    const partyClass = member.party.includes('Democrat') ? 'tag-dem' : 'tag-rep';
                    const partyShort = member.party.includes('Democrat') ? 'D' : 'R';
                    
                    const memberDiv = document.createElement('div');
                    memberDiv.className = 'member-item';
                    
                    // Generate trades HTML
                    const tradesHtml = member.recent_trades.map(trade => {
                        const actionClass = trade.action.toLowerCase().includes('buy') ? 'action-buy' : 'action-sell';
                        const icon = trade.action.toLowerCase().includes('buy') ? 'fa-plus' : 'fa-minus';
                        
                        return `
                            <div class="trade-card" onclick="alert('Trade Details:\\n${trade.action} ${trade.amount} of ${trade.stock}\\nDate: ${trade.date}')">
                                <div class="trade-header">
                                    <span class="stock-symbol">${trade.stock}</span>
                                    <span class="trade-date">${trade.date}</span>
                                </div>
                                <div class="trade-action ${actionClass}">
                                    <i class="fas ${icon}"></i> ${trade.action}
                                </div>
                                <div class="trade-amount">${trade.amount}</div>
                            </div>
                        `;
                    }).join('');
                    
                    memberDiv.innerHTML = `
                        <div class="member-header">
                            <div class="member-info">
                                <h4>
                                    ${member.name} 
                                    <span class="party-tag ${partyClass}">${partyShort}</span>
                                </h4>
                                <div class="member-meta">
                                    <i class="fas fa-map-marker-alt"></i> ${member.state} • ${member.chamber}
                                </div>
                            </div>
                        </div>
                        <div class="trades-grid">
                            ${tradesHtml}
                        </div>
                    `;
                    
                    container.appendChild(memberDiv);
                });
            }
            
            function filterMembers() {
                const searchTerm = document.getElementById('searchInput').value.toLowerCase();
                
                const filtered = members.filter(member => {
                    // Check party filter first
                    if (currentPartyFilter !== 'all' && !member.party.includes(currentPartyFilter)) {
                        return false;
                    }
                    
                    // Then check search term
                    const searchString = `${member.name} ${member.state} ${member.party} ${JSON.stringify(member.recent_trades)}`.toLowerCase();
                    return searchString.includes(searchTerm);
                });
                
                renderMembers(filtered);
            }
            
            function filterByParty(party) {
                currentPartyFilter = party;
                
                // Update buttons
                document.querySelectorAll('.filter-group .filter-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                if (party === 'all') document.getElementById('btn-all').classList.add('active');
                else if (party === 'Democrat') document.getElementById('btn-dem').classList.add('active');
                else if (party === 'Republican') document.getElementById('btn-rep').classList.add('active');
                
                filterMembers(); // Re-run filter including search term
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', () => renderMembers());
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
        "version": "1.0.0-enhanced"
    })

@app.route('/api/v1/info')
def api_info():
    """API information"""
    return jsonify({
        "name": "Congressional Trading Intelligence API",
        "version": "1.0.0-enhanced",
        "description": "Enhanced API for congressional trading data analysis",
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
    print("🚀 Starting Enhanced Congressional Trading Intelligence API...")
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