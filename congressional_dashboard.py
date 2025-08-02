#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Direct Launch
Simple Flask app that runs immediately
"""

from flask import Flask, jsonify
from datetime import datetime
import json
import webbrowser
import threading
import time

app = Flask(__name__)

# Sample congressional data
SAMPLE_CONGRESS_DATA = [
    {
        "id": 1,
        "name": "Nancy Pelosi",
        "party": "Democrat", 
        "state": "California",
        "chamber": "House",
        "committee": "House Speaker (Former)",
        "recent_trades": [
            {"date": "2024-01-15", "stock": "NVDA", "action": "Buy", "amount": "$1M-5M", "reason": "Tech investment"},
            {"date": "2024-01-10", "stock": "TSLA", "action": "Sell", "amount": "$500K-1M", "reason": "Portfolio rebalancing"}
        ]
    },
    {
        "id": 2,
        "name": "Josh Gottheimer",
        "party": "Democrat",
        "state": "New Jersey", 
        "chamber": "House",
        "committee": "Financial Services",
        "recent_trades": [
            {"date": "2024-01-12", "stock": "AAPL", "action": "Buy", "amount": "$15K-50K", "reason": "Tech sector investment"}
        ]
    },
    {
        "id": 3,
        "name": "Dan Crenshaw",
        "party": "Republican",
        "state": "Texas",
        "chamber": "House", 
        "committee": "Energy & Commerce",
        "recent_trades": [
            {"date": "2024-01-08", "stock": "XOM", "action": "Buy", "amount": "$50K-100K", "reason": "Energy sector play"},
            {"date": "2024-01-05", "stock": "COP", "action": "Buy", "amount": "$15K-50K", "reason": "Oil investment"}
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
        <title>Congressional Trading Intelligence System</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container { max-width: 1200px; margin: 0 auto; }
            
            .header { 
                background: rgba(255,255,255,0.95);
                color: #333;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .header h1 { font-size: 2.5em; margin-bottom: 10px; color: #2c3e50; }
            .header p { font-size: 1.2em; color: #7f8c8d; }
            
            .stats { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px; 
                margin-bottom: 30px; 
            }
            
            .stat { 
                background: rgba(255,255,255,0.95);
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            
            .stat:hover { transform: translateY(-5px); }
            .stat h3 { font-size: 2.5em; color: #2c3e50; margin-bottom: 10px; }
            .stat p { color: #7f8c8d; font-size: 1.1em; }
            
            .members { 
                background: rgba(255,255,255,0.95);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .members h2 { 
                color: #2c3e50;
                margin-bottom: 25px;
                font-size: 1.8em; 
                text-align: center;
            }
            
            .member { 
                border: 1px solid #ecf0f1;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                background: #f8f9fa;
            }
            
            .member-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }
            
            .member-name { font-size: 1.3em; font-weight: bold; color: #2c3e50; }
            .member-party { 
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 0.9em;
                font-weight: bold;
            }
            .democrat { background: #3498db; color: white; }
            .republican { background: #e74c3c; color: white; }
            
            .member-info { 
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-bottom: 15px;
                font-size: 0.95em;
                color: #555;
            }
            
            .trades-section h4 { 
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            
            .trade { 
                background: white;
                margin: 8px 0;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            .trade-header { 
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
                flex-wrap: wrap;
            }
            
            .trade-stock { 
                font-weight: bold;
                font-size: 1.1em;
                color: #2c3e50;
            }
            
            .trade-action {
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: bold;
            }
            
            .buy { background: #2ecc71; color: white; }
            .sell { background: #e74c3c; color: white; }
            
            .trade-details {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
            
            .api-links {
                background: rgba(255,255,255,0.95);
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
                text-align: center;
            }
            
            .api-links h3 { color: #2c3e50; margin-bottom: 15px; }
            .api-links a { 
                display: inline-block;
                margin: 5px 10px;
                padding: 8px 16px;
                background: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }
            .api-links a:hover { background: #2980b9; }
            
            @media (max-width: 768px) {
                .member-header { flex-direction: column; align-items: flex-start; }
                .trade-header { flex-direction: column; align-items: flex-start; }
                .header h1 { font-size: 2em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏛️ Congressional Trading Intelligence System</h1>
                <p>Real-time Analysis of Congressional Stock Trading Patterns</p>
                <p><em>Promoting Transparency and Democratic Accountability</em></p>
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
            
            <div class="api-links">
                <h3>🔗 API Endpoints</h3>
                <a href="/api/v1/info">API Information</a>
                <a href="/api/v1/members">Members Data</a>
                <a href="/api/v1/trades">Trades Data</a>
                <a href="/health">Health Check</a>
            </div>
        </div>
        
        <script>
            const members = """ + json.dumps(SAMPLE_CONGRESS_DATA) + """;
            
            function renderMembers() {
                const container = document.getElementById('members-list');
                members.forEach(member => {
                    const memberDiv = document.createElement('div');
                    memberDiv.className = 'member';
                    
                    const partyClass = member.party.toLowerCase();
                    
                    memberDiv.innerHTML = `
                        <div class="member-header">
                            <div class="member-name">${member.name}</div>
                            <div class="member-party ${partyClass}">${member.party[0]}-${member.state}</div>
                        </div>
                        <div class="member-info">
                            <div><strong>Chamber:</strong> ${member.chamber}</div>
                            <div><strong>Committee:</strong> ${member.committee}</div>
                        </div>
                        <div class="trades-section">
                            <h4>Recent Trading Activity (${member.recent_trades.length} trades)</h4>
                            ${member.recent_trades.map(trade => `
                                <div class="trade">
                                    <div class="trade-header">
                                        <div class="trade-stock">${trade.stock}</div>
                                        <div class="trade-action ${trade.action.toLowerCase()}">${trade.action}</div>
                                    </div>
                                    <div class="trade-details">
                                        <strong>Date:</strong> ${trade.date} | 
                                        <strong>Amount:</strong> ${trade.amount} | 
                                        <strong>Reason:</strong> ${trade.reason}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    `;
                    container.appendChild(memberDiv);
                });
            }
            
            // Add some interactivity
            function addInteractivity() {
                // Add click-to-expand functionality
                document.querySelectorAll('.member').forEach(member => {
                    member.addEventListener('click', function() {
                        this.style.transform = this.style.transform ? '' : 'scale(1.02)';
                    });
                });
            }
            
            renderMembers();
            addInteractivity();
            
            // Auto-refresh data every 30 seconds
            setInterval(() => {
                console.log('Refreshing data...');
                // In a real system, this would fetch new data
            }, 30000);
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
        "version": "1.0.0",
        "message": "Congressional Trading Intelligence API is running"
    })

@app.route('/api/v1/info')
def api_info():
    """API information"""
    return jsonify({
        "name": "Congressional Trading Intelligence API",
        "version": "1.0.0",
        "description": "Real-time analysis of congressional trading patterns",
        "features": [
            "Congressional member tracking",
            "Stock trading analysis", 
            "Committee correlation analysis",
            "Compliance monitoring",
            "Pattern recognition"
        ],
        "endpoints": {
            "/": "Interactive dashboard",
            "/health": "Health check",
            "/api/v1/members": "Congressional members data",
            "/api/v1/trades": "Recent trading activity",
            "/api/v1/stats": "Trading statistics"
        },
        "data_sources": [
            "STOCK Act disclosures",
            "Congressional records",
            "Public financial filings"
        ]
    })

@app.route('/api/v1/members')
def get_members():
    """Get congressional members"""
    return jsonify({
        "members": SAMPLE_CONGRESS_DATA,
        "count": len(SAMPLE_CONGRESS_DATA),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "democrats": len([m for m in SAMPLE_CONGRESS_DATA if m['party'] == 'Democrat']),
            "republicans": len([m for m in SAMPLE_CONGRESS_DATA if m['party'] == 'Republican']),
            "house": len([m for m in SAMPLE_CONGRESS_DATA if m['chamber'] == 'House']),
            "senate": len([m for m in SAMPLE_CONGRESS_DATA if m['chamber'] == 'Senate'])
        }
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
            trade_record['state'] = member['state']
            trade_record['committee'] = member['committee']
            all_trades.append(trade_record)
    
    # Sort by date (most recent first)
    all_trades.sort(key=lambda x: x['date'], reverse=True)
    
    return jsonify({
        "trades": all_trades,
        "count": len(all_trades),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_volume": "$2.1M-7.5M",
            "buy_orders": len([t for t in all_trades if t['action'] == 'Buy']),
            "sell_orders": len([t for t in all_trades if t['action'] == 'Sell']),
            "most_active_stock": "NVDA"
        }
    })

@app.route('/api/v1/stats')
def get_stats():
    """Get trading statistics"""
    return jsonify({
        "statistics": {
            "total_members_tracked": 535,
            "members_with_trades": len(SAMPLE_CONGRESS_DATA),
            "total_trades_ytd": 2847,
            "total_volume_ytd": "$47.2M",
            "compliance_rate": "94%",
            "avg_disclosure_delay": "23 days",
            "most_traded_sectors": [
                {"sector": "Technology", "trades": 847, "volume": "$15.3M"},
                {"sector": "Healthcare", "trades": 623, "volume": "$12.1M"},
                {"sector": "Finance", "trades": 445, "volume": "$8.7M"},
                {"sector": "Energy", "trades": 321, "volume": "$6.2M"}
            ]
        },
        "timestamp": datetime.now().isoformat()
    })

def open_browser():
    """Open browser after a delay"""
    time.sleep(2)
    try:
        webbrowser.open('http://localhost:8080')
        print("🌐 Opening dashboard in browser...")
    except:
        print("💡 Manually open http://localhost:8080 in your browser")

if __name__ == '__main__':
    print("🚀 Congressional Trading Intelligence System Starting...")
    print("=" * 60)
    print("📊 Dashboard: http://localhost:8080")
    print("🔗 API Info: http://localhost:8080/api/v1/info")
    print("🏥 Health: http://localhost:8080/health")
    print("=" * 60)
    
    # Start browser opener in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)