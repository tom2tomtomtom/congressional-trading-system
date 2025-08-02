#!/usr/bin/env python3
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
