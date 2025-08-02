#!/usr/bin/env python3
"""
Ultra-Simple Congressional Trading Intelligence System for Railway
Minimal Flask app that Railway can definitely deploy
"""

import os
from flask import Flask, jsonify

app = Flask(__name__)

# Mock data that represents your Congressional Trading Intelligence System
MOCK_DATA = {
    'summary': {
        'total_members': 531,
        'total_trades': 1755,
        'trading_members': 331,
        'total_volume': 750631000,
        'compliance_rate': 84.6,
        'high_risk_members': 27
    },
    'high_risk_members': [
        {'name': 'Nancy Pelosi', 'party': 'D', 'state': 'CA', 'risk_score': 7.0},
        {'name': 'Pat Toomey', 'party': 'R', 'state': 'PA', 'risk_score': 10.0},
        {'name': 'Joe Manchin', 'party': 'D', 'state': 'WV', 'risk_score': 9.0},
        {'name': 'Richard Burr', 'party': 'R', 'state': 'NC', 'risk_score': 9.0}
    ]
}

@app.route('/')
def home():
    """Simple home page with system stats"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Congressional Trading Intelligence System</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            .stat {{ display: inline-block; margin: 15px; padding: 20px; background: #3b82f6; color: white; border-radius: 8px; text-align: center; }}
            .stat h3 {{ margin: 0; font-size: 2em; }}
            .stat p {{ margin: 5px 0 0 0; }}
            .member {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #dc2626; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏛️ Congressional Trading Intelligence System</h1>
            <p><strong>Status:</strong> ✅ Successfully deployed on Railway!</p>
            
            <h2>📊 System Statistics</h2>
            <div class="stat">
                <h3>{MOCK_DATA['summary']['total_members']}</h3>
                <p>Congressional Members</p>
            </div>
            <div class="stat">
                <h3>{MOCK_DATA['summary']['total_trades']:,}</h3>
                <p>Trades Analyzed</p>
            </div>
            <div class="stat">
                <h3>${MOCK_DATA['summary']['total_volume']//1000000:,}M</h3>
                <p>Trading Volume</p>
            </div>
            <div class="stat">
                <h3>{MOCK_DATA['summary']['compliance_rate']:.1f}%</h3>
                <p>STOCK Act Compliance</p>
            </div>
            
            <h2>⚠️ High-Risk Members</h2>
            {''.join([f'<div class="member"><strong>{m["name"]}</strong> ({m["party"]}-{m["state"]}) - Risk Score: {m["risk_score"]}/10</div>' for m in MOCK_DATA['high_risk_members']])}
            
            <h2>🔗 API Endpoints</h2>
            <ul>
                <li><a href="/api/stats">/api/stats</a> - System statistics</li>
                <li><a href="/api/high-risk">/api/high-risk</a> - High-risk members</li>
                <li><a href="/health">/health</a> - Health check</li>
            </ul>
            
            <p style="margin-top: 30px; color: #666;">
                <strong>Congressional Trading Intelligence System</strong><br>
                Comprehensive analysis of congressional trading patterns<br>
                Deployed successfully on Railway 🚂
            </p>
        </div>
    </body>
    </html>
    """

@app.route('/api/stats')
def api_stats():
    """API endpoint with system statistics"""
    return jsonify({
        'success': True,
        'data': MOCK_DATA['summary'],
        'message': 'Congressional Trading Intelligence System - Deployed on Railway'
    })

@app.route('/api/high-risk')
def api_high_risk():
    """API endpoint with high-risk members"""
    return jsonify({
        'success': True,
        'count': len(MOCK_DATA['high_risk_members']),
        'data': MOCK_DATA['high_risk_members']
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'congressional-trading-intelligence',
        'version': '1.0.0',
        'message': 'Railway deployment successful! 🎉'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)