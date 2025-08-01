#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Phase 2
React-based Interactive Dashboard

This module implements a modern React-style dashboard using Dash framework
for advanced data visualization and real-time congressional trading intelligence.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Dash and Plotly for React-style dashboard
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Data processing
import pandas as pd
import numpy as np

# Database
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml

# Our intelligence modules
sys.path.append(str(Path(__file__).parent.parent))
from intelligence.suspicious_trading_detector import SuspiciousTradingDetector
from intelligence.network_analyzer import NetworkAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CongressionalDashboard:
    """
    Modern React-style dashboard for congressional trading intelligence.
    Built with Dash for interactive data visualization and real-time updates.
    """
    
    def __init__(self, config_path: str = "config/database.yml"):
        """Initialize the dashboard."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.db_config = self.config.get('development', {}).get('database', {})
        
        # Initialize intelligence modules
        self.detector = SuspiciousTradingDetector(config_path)
        self.network_analyzer = NetworkAnalyzer(config_path)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
            ],
            suppress_callback_exceptions=True
        )
        
        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        # Data cache
        self.data_cache = {
            'last_update': None,
            'trading_data': None,
            'analysis_results': None,
            'network_data': None
        }
        
    def _load_config(self) -> Dict:
        """Load database configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            database=self.db_config.get('name', 'congressional_trading_dev'),
            user=self.db_config.get('user', 'postgres'),
            password=self.db_config.get('password', 'password')
        )
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-building-columns me-3"),
                        "Congressional Trading Intelligence"
                    ], className="text-primary mb-0"),
                    html.P("Advanced Real-Time Trading Pattern Analysis", 
                          className="text-muted lead")
                ], width=8),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Refresh Data"
                    ], id="refresh-btn", color="primary", size="lg", className="float-end")
                ], width=4)
            ], className="mb-4"),
            
            # Key Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                                html.Span(id="high-risk-count", children="--")
                            ]),
                            html.P("High-Risk Alerts", className="text-muted mb-0")
                        ])
                    ], className="h-100")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-dollar-sign text-success me-2"),
                                html.Span(id="total-volume", children="--")
                            ]),
                            html.P("Total Trade Volume", className="text-muted mb-0")
                        ])
                    ], className="h-100")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-users text-info me-2"),
                                html.Span(id="active-members", children="--")
                            ]),
                            html.P("Active Members", className="text-muted mb-0")
                        ])
                    ], className="h-100")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-line text-warning me-2"),
                                html.Span(id="avg-suspicion", children="--")
                            ]),
                            html.P("Avg Suspicion Score", className="text-muted mb-0")
                        ])
                    ], className="h-100")
                ], width=3)
            ], className="mb-4"),
            
            # Main Content Tabs
            dbc.Tabs([
                dbc.Tab(label="🚨 Real-Time Alerts", tab_id="alerts"),
                dbc.Tab(label="📊 Trading Analysis", tab_id="analysis"),
                dbc.Tab(label="🕸️ Network Insights", tab_id="network"),
                dbc.Tab(label="🔍 Member Profiles", tab_id="profiles"),
                dbc.Tab(label="📈 Market Impact", tab_id="market")
            ], id="main-tabs", active_tab="alerts", className="mb-4"),
            
            # Tab Content
            html.Div(id="tab-content"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            ),
            
            # Data stores
            dcc.Store(id='trading-data-store'),
            dcc.Store(id='analysis-data-store'),
            dcc.Store(id='network-data-store')
            
        ], fluid=True, className="py-4")
    
    def _create_alerts_tab(self, analysis_data: Optional[pd.DataFrame] = None) -> html.Div:
        """Create the real-time alerts tab."""
        if analysis_data is None or analysis_data.empty:
            return html.Div([
                dbc.Alert("No analysis data available. Click 'Refresh Data' to load.", 
                         color="warning", className="text-center")
            ])
        
        # Get high-risk trades
        high_risk_trades = analysis_data[analysis_data['suspicion_score'] >= 6.0].nlargest(10, 'suspicion_score')
        
        alert_cards = []
        for _, trade in high_risk_trades.iterrows():
            # Determine alert level
            if trade['suspicion_score'] >= 8.0:
                alert_color = "danger"
                alert_icon = "fas fa-exclamation-circle"
                alert_level = "EXTREME"
            elif trade['suspicion_score'] >= 7.0:
                alert_color = "warning"
                alert_icon = "fas fa-exclamation-triangle"
                alert_level = "HIGH"
            else:
                alert_color = "info"
                alert_icon = "fas fa-info-circle"
                alert_level = "MEDIUM"
            
            alert_cards.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.Span([
                            html.I(className=f"{alert_icon} me-2"),
                            f"{alert_level} RISK ALERT"
                        ], className=f"badge bg-{alert_color} fs-6"),
                        html.Span(f"Score: {trade['suspicion_score']:.1f}/10", 
                                className="float-end text-muted")
                    ]),
                    dbc.CardBody([
                        html.H5(f"{trade['full_name']} ({trade['party']}-{trade['state']})", 
                               className="card-title"),
                        html.P([
                            html.Strong(f"{trade['symbol']} {trade['transaction_type']}: "),
                            f"${trade['amount_mid']:,.0f}"
                        ], className="card-text"),
                        html.Small([
                            f"Trade Date: {trade['transaction_date']} | ",
                            f"Filing Date: {trade['filing_date']} | ",
                            f"Delay: {trade.get('filing_delay_days', 0)} days"
                        ], className="text-muted")
                    ])
                ], className="mb-3")
            )
        
        return html.Div([
            html.H4("🚨 High-Priority Trading Alerts", className="mb-3"),
            html.Div(alert_cards) if alert_cards else dbc.Alert(
                "No high-risk alerts detected.", color="success", className="text-center"
            )
        ])
    
    def _create_analysis_tab(self, analysis_data: Optional[pd.DataFrame] = None) -> html.Div:
        """Create the trading analysis tab."""
        if analysis_data is None or analysis_data.empty:
            return html.Div([
                dbc.Alert("No analysis data available.", color="warning", className="text-center")
            ])
        
        # Create suspicion score distribution
        fig_distribution = px.histogram(
            analysis_data,
            x='suspicion_score',
            nbins=20,
            title='Suspicion Score Distribution',
            labels={'suspicion_score': 'Suspicion Score', 'count': 'Number of Trades'}
        )
        fig_distribution.update_layout(showlegend=False)
        
        # Create party analysis
        party_analysis = analysis_data.groupby('party').agg({
            'suspicion_score': 'mean',
            'amount_mid': 'sum',
            'id': 'count'
        }).round(2)
        party_analysis.columns = ['Avg Suspicion Score', 'Total Volume', 'Trade Count']
        
        fig_party = px.bar(
            party_analysis.reset_index(),
            x='party',
            y='Avg Suspicion Score',
            title='Average Suspicion Score by Party'
        )
        
        # Create timeline chart
        analysis_data['transaction_date'] = pd.to_datetime(analysis_data['transaction_date'])
        timeline_data = analysis_data.groupby([
            analysis_data['transaction_date'].dt.to_period('M'), 'party'
        ]).agg({
            'suspicion_score': 'mean',
            'amount_mid': 'sum'
        }).reset_index()
        timeline_data['transaction_date'] = timeline_data['transaction_date'].astype(str)
        
        fig_timeline = px.line(
            timeline_data,
            x='transaction_date',
            y='suspicion_score',
            color='party',
            title='Suspicion Score Trends Over Time'
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_distribution)
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=fig_party)
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_timeline)
                ], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.H5("Party Analysis Summary"),
                    dash_table.DataTable(
                        data=party_analysis.reset_index().to_dict('records'),
                        columns=[{"name": i, "id": i} for i in party_analysis.reset_index().columns],
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]
                    )
                ], width=12)
            ])
        ])
    
    def _create_network_tab(self) -> html.Div:
        """Create the network analysis tab."""
        # This would show network visualizations
        # For now, showing placeholder content
        return html.Div([
            html.H4("🕸️ Network Analysis", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Committee Networks", className="card-title"),
                            html.P("Visualization of committee membership relationships and trading correlations.", 
                                  className="card-text"),
                            dbc.Button("Generate Network Graph", color="primary", disabled=True)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Influence Metrics", className="card-title"),
                            html.P("Analysis of member influence based on network position and trading activity.", 
                                  className="card-text"),
                            dbc.Button("Calculate Influence", color="secondary", disabled=True)
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_profiles_tab(self, analysis_data: Optional[pd.DataFrame] = None) -> html.Div:
        """Create the member profiles tab."""
        if analysis_data is None or analysis_data.empty:
            return html.Div([
                dbc.Alert("No member data available.", color="warning", className="text-center")
            ])
        
        # Create member summary
        member_summary = analysis_data.groupby(['bioguide_id', 'full_name', 'party', 'state']).agg({
            'suspicion_score': 'mean',
            'amount_mid': ['sum', 'count'],
            'filing_delay_days': 'mean'
        }).round(2)
        
        member_summary.columns = ['Avg Suspicion', 'Total Volume', 'Trade Count', 'Avg Filing Delay']
        member_summary = member_summary.reset_index().sort_values('Avg Suspicion', ascending=False)
        
        return html.Div([
            html.H4("👥 Member Trading Profiles", className="mb-3"),
            dash_table.DataTable(
                data=member_summary.head(20).to_dict('records'),
                columns=[
                    {"name": "Member", "id": "full_name"},
                    {"name": "Party", "id": "party"},
                    {"name": "State", "id": "state"},
                    {"name": "Avg Suspicion", "id": "Avg Suspicion", "type": "numeric", "format": ".2f"},
                    {"name": "Total Volume", "id": "Total Volume", "type": "numeric", "format": "$,.0f"},
                    {"name": "Trades", "id": "Trade Count", "type": "numeric"},
                    {"name": "Avg Delay (days)", "id": "Avg Filing Delay", "type": "numeric", "format": ".1f"}
                ],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                        'if': {
                            'filter_query': '{Avg Suspicion} > 5',
                            'column_id': 'Avg Suspicion'
                        },
                        'backgroundColor': '#ffebee',
                        'color': 'black',
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                page_size=15,
                sort_action="native"
            )
        ])
    
    def _create_market_tab(self, analysis_data: Optional[pd.DataFrame] = None) -> html.Div:
        """Create the market impact analysis tab."""
        if analysis_data is None or analysis_data.empty:
            return html.Div([
                dbc.Alert("No market data available.", color="warning", className="text-center")
            ])
        
        # Symbol analysis
        symbol_analysis = analysis_data.groupby('symbol').agg({
            'suspicion_score': 'mean',
            'amount_mid': 'sum',
            'bioguide_id': 'nunique'
        }).round(2)
        symbol_analysis.columns = ['Avg Suspicion', 'Total Volume', 'Unique Members']
        symbol_analysis = symbol_analysis.sort_values('Total Volume', ascending=False).head(10)
        
        fig_symbols = px.bar(
            symbol_analysis.reset_index(),
            x='symbol',
            y='Total Volume',
            title='Most Traded Symbols by Volume'
        )
        
        return html.Div([
            html.H4("📈 Market Impact Analysis", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_symbols)
                ], width=8),
                dbc.Col([
                    html.H5("Top Traded Symbols"),
                    dash_table.DataTable(
                        data=symbol_analysis.reset_index().to_dict('records'),
                        columns=[{"name": i, "id": i} for i in symbol_analysis.reset_index().columns],
                        style_cell={'textAlign': 'left'},
                        page_size=10
                    )
                ], width=4)
            ])
        ])
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('trading-data-store', 'data'),
             Output('analysis-data-store', 'data'),
             Output('high-risk-count', 'children'),
             Output('total-volume', 'children'),
             Output('active-members', 'children'),
             Output('avg-suspicion', 'children')],
            [Input('refresh-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_data(n_clicks, n_intervals):
            """Update dashboard data."""
            try:
                # Run analysis
                analysis_df, alerts_df = self.detector.run_full_analysis()
                
                if analysis_df.empty:
                    return {}, {}, "0", "$0", "0", "0.0"
                
                # Calculate metrics
                high_risk_count = len(analysis_df[analysis_df['suspicion_score'] >= 7.0])
                total_volume = analysis_df['amount_mid'].sum()
                active_members = analysis_df['bioguide_id'].nunique()
                avg_suspicion = analysis_df['suspicion_score'].mean()
                
                return (
                    analysis_df.to_dict('records'),
                    analysis_df.to_dict('records'),
                    str(high_risk_count),
                    f"${total_volume:,.0f}",
                    str(active_members),
                    f"{avg_suspicion:.1f}"
                )
                
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                return {}, {}, "Error", "Error", "Error", "Error"
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('analysis-data-store', 'data')]
        )
        def update_tab_content(active_tab, analysis_data):
            """Update tab content based on selection."""
            try:
                if not analysis_data:
                    return dbc.Alert("Loading data...", color="info", className="text-center")
                
                df = pd.DataFrame(analysis_data)
                
                if active_tab == "alerts":
                    return self._create_alerts_tab(df)
                elif active_tab == "analysis":
                    return self._create_analysis_tab(df)
                elif active_tab == "network":
                    return self._create_network_tab()
                elif active_tab == "profiles":
                    return self._create_profiles_tab(df)
                elif active_tab == "market":
                    return self._create_market_tab(df)
                else:
                    return html.Div("Tab content not implemented yet.")
                    
            except Exception as e:
                logger.error(f"Error updating tab content: {e}")
                return dbc.Alert(f"Error loading content: {str(e)}", color="danger")
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server."""
        logger.info(f"Starting Congressional Trading Intelligence Dashboard on port {port}")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')

def main():
    """Main execution function."""
    logger.info("Initializing Congressional Trading Intelligence Dashboard...")
    
    dashboard = CongressionalDashboard()
    dashboard.run_server(debug=True, port=8050)

if __name__ == "__main__":
    main()