#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Interactive Network Visualizations
Advanced network graphs for analyzing relationships between members, committees, and trades.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_cytoscape as cyto

logger = logging.getLogger(__name__)

@dataclass
class NetworkNode:
    """Data model for network nodes."""
    id: str
    label: str
    node_type: str  # member, committee, stock, bill
    size: float
    color: str
    data: Dict[str, Any]

@dataclass
class NetworkEdge:
    """Data model for network edges."""
    source: str
    target: str
    weight: float
    edge_type: str  # trade, membership, cosponsorship
    label: str
    data: Dict[str, Any]

@dataclass
class NetworkLayout:
    """Network layout configuration."""
    algorithm: str  # force, circular, hierarchical, cluster
    parameters: Dict[str, Any]

class CongressionalNetworkBuilder:
    """Builds network graphs from congressional data."""
    
    def __init__(self):
        """Initialize network builder."""
        self.graph = nx.Graph()
        self.nodes = []
        self.edges = []
        
        # Color schemes for different node types
        self.colors = {
            'member_R': '#FF6B6B',      # Republican - Red
            'member_D': '#4ECDC4',      # Democrat - Blue/Teal
            'member_I': '#45B7D1',      # Independent - Light Blue
            'committee': '#96CEB4',      # Committee - Green
            'stock': '#FFEAA7',         # Stock - Yellow
            'bill': '#DDA0DD',          # Bill - Purple
            'sector': '#F8B500'         # Sector - Orange
        }
    
    def build_member_trading_network(self, trading_data: pd.DataFrame, 
                                   member_data: pd.DataFrame) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """
        Build network showing trading relationships between members.
        
        Args:
            trading_data: DataFrame with trading data
            member_data: DataFrame with member information
            
        Returns:
            Tuple of (nodes, edges) for the network
        """
        logger.info("Building member trading network")
        
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # Add member nodes
        member_nodes = {}
        for _, member in member_data.iterrows():
            member_id = member['bioguide_id']
            party = member.get('party', 'I')
            
            # Calculate member's total trading volume
            member_trades = trading_data[trading_data['bioguide_id'] == member_id]
            total_volume = member_trades['amount_mid'].sum() if len(member_trades) > 0 else 0
            
            node = NetworkNode(
                id=member_id,
                label=f"{member['first_name']} {member['last_name']}",
                node_type=f"member_{party}",
                size=max(10, min(50, np.log10(total_volume + 1) * 5)),  # Size based on volume
                color=self.colors.get(f'member_{party}', self.colors['member_I']),
                data={
                    'party': party,
                    'state': member.get('state', ''),
                    'chamber': member.get('chamber', ''),
                    'total_trades': len(member_trades),
                    'total_volume': total_volume,
                    'leadership': member.get('leadership_position', '')
                }
            )
            
            self.nodes.append(node)
            member_nodes[member_id] = node
            self.graph.add_node(member_id, **asdict(node))
        
        # Add edges based on shared stock holdings
        self._add_shared_stock_edges(trading_data, member_nodes)
        
        # Add edges based on coordinated trading (same stock, similar timing)
        self._add_coordinated_trading_edges(trading_data, member_nodes)
        
        logger.info(f"Built network with {len(self.nodes)} nodes and {len(self.edges)} edges")
        return self.nodes, self.edges
    
    def _add_shared_stock_edges(self, trading_data: pd.DataFrame, member_nodes: Dict[str, NetworkNode]):
        """Add edges between members who trade the same stocks."""
        # Group trades by symbol
        for symbol in trading_data['symbol'].unique():
            symbol_trades = trading_data[trading_data['symbol'] == symbol]
            symbol_members = symbol_trades['bioguide_id'].unique()
            
            # Create edges between all pairs of members who traded this symbol
            for i, member1 in enumerate(symbol_members):
                for member2 in symbol_members[i+1:]:
                    if member1 in member_nodes and member2 in member_nodes:
                        
                        # Calculate edge weight based on trading similarity
                        member1_trades = symbol_trades[symbol_trades['bioguide_id'] == member1]
                        member2_trades = symbol_trades[symbol_trades['bioguide_id'] == member2]
                        
                        weight = self._calculate_trading_similarity(member1_trades, member2_trades)
                        
                        if weight > 0.1:  # Only add edges with meaningful similarity
                            edge = NetworkEdge(
                                source=member1,
                                target=member2,
                                weight=weight,
                                edge_type='shared_stock',
                                label=f"Both traded {symbol}",
                                data={
                                    'symbol': symbol,
                                    'similarity_score': weight,
                                    'member1_trades': len(member1_trades),
                                    'member2_trades': len(member2_trades)
                                }
                            )
                            
                            self.edges.append(edge)
                            self.graph.add_edge(member1, member2, weight=weight, **asdict(edge))
    
    def _add_coordinated_trading_edges(self, trading_data: pd.DataFrame, member_nodes: Dict[str, NetworkNode]):
        """Add edges for potentially coordinated trading (same time + same stock)."""
        trading_data['transaction_date'] = pd.to_datetime(trading_data['transaction_date'])
        
        # Look for trades within 7 days of each other
        time_window = timedelta(days=7)
        
        for symbol in trading_data['symbol'].unique():
            symbol_trades = trading_data[trading_data['symbol'] == symbol].sort_values('transaction_date')
            
            for i, trade1 in symbol_trades.iterrows():
                for j, trade2 in symbol_trades.iterrows():
                    if i >= j:  # Avoid duplicates and self-comparison
                        continue
                    
                    member1 = trade1['bioguide_id']
                    member2 = trade2['bioguide_id']
                    
                    if member1 == member2:  # Same member
                        continue
                    
                    # Check if trades are within time window
                    time_diff = abs((trade2['transaction_date'] - trade1['transaction_date']).total_seconds())
                    if time_diff <= time_window.total_seconds():
                        
                        # Calculate coordination strength
                        coord_strength = self._calculate_coordination_strength(trade1, trade2, time_diff)
                        
                        if coord_strength > 0.3:  # Minimum threshold for coordination
                            edge = NetworkEdge(
                                source=member1,
                                target=member2,
                                weight=coord_strength,
                                edge_type='coordinated_trading',
                                label=f"Coordinated {symbol} trades",
                                data={
                                    'symbol': symbol,
                                    'coordination_strength': coord_strength,
                                    'time_diff_days': time_diff / 86400,
                                    'trade1_date': trade1['transaction_date'].isoformat(),
                                    'trade2_date': trade2['transaction_date'].isoformat()
                                }
                            )
                            
                            self.edges.append(edge)
                            # Add to graph with higher weight for coordinated trades
                            if self.graph.has_edge(member1, member2):
                                self.graph[member1][member2]['weight'] += coord_strength * 2
                            else:
                                self.graph.add_edge(member1, member2, weight=coord_strength * 2, **asdict(edge))
    
    def _calculate_trading_similarity(self, trades1: pd.DataFrame, trades2: pd.DataFrame) -> float:
        """Calculate similarity between two members' trading patterns."""
        if len(trades1) == 0 or len(trades2) == 0:
            return 0.0
        
        # Simple similarity based on:
        # 1. Trade direction (buy/sell)
        # 2. Trade timing
        # 3. Trade amounts
        
        similarity_scores = []
        
        # Direction similarity
        if len(trades1) > 0 and len(trades2) > 0:
            buy_ratio1 = (trades1['transaction_type'] == 'Purchase').mean()
            buy_ratio2 = (trades2['transaction_type'] == 'Purchase').mean()
            direction_sim = 1.0 - abs(buy_ratio1 - buy_ratio2)
            similarity_scores.append(direction_sim)
        
        # Amount similarity (using coefficient of variation)
        amounts1 = trades1['amount_mid'].dropna()
        amounts2 = trades2['amount_mid'].dropna()
        
        if len(amounts1) > 0 and len(amounts2) > 0:
            cv1 = amounts1.std() / amounts1.mean() if amounts1.mean() > 0 else 1.0
            cv2 = amounts2.std() / amounts2.mean() if amounts2.mean() > 0 else 1.0
            amount_sim = 1.0 - min(1.0, abs(cv1 - cv2))
            similarity_scores.append(amount_sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.1
    
    def _calculate_coordination_strength(self, trade1: pd.Series, trade2: pd.Series, time_diff: float) -> float:
        """Calculate coordination strength between two trades."""
        # Base score from timing (closer = higher score)
        max_time_diff = 7 * 86400  # 7 days in seconds
        time_score = 1.0 - (time_diff / max_time_diff)
        
        # Direction bonus (same direction = higher coordination)
        direction_bonus = 0.5 if trade1['transaction_type'] == trade2['transaction_type'] else 0.0
        
        # Amount similarity bonus
        amount1 = trade1.get('amount_mid', 0)
        amount2 = trade2.get('amount_mid', 0)
        
        if amount1 > 0 and amount2 > 0:
            amount_ratio = min(amount1, amount2) / max(amount1, amount2)
            amount_bonus = amount_ratio * 0.3
        else:
            amount_bonus = 0.0
        
        coordination_strength = time_score + direction_bonus + amount_bonus
        return min(1.0, coordination_strength)
    
    def build_committee_network(self, member_data: pd.DataFrame, 
                              committee_data: pd.DataFrame,
                              membership_data: pd.DataFrame) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """
        Build network showing committee membership relationships.
        
        Args:
            member_data: DataFrame with member information
            committee_data: DataFrame with committee information  
            membership_data: DataFrame with committee memberships
            
        Returns:
            Tuple of (nodes, edges) for the network
        """
        logger.info("Building committee network")
        
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # Add member nodes
        for _, member in member_data.iterrows():
            member_id = member['bioguide_id']
            party = member.get('party', 'I')
            
            node = NetworkNode(
                id=member_id,
                label=f"{member['first_name']} {member['last_name']}",
                node_type=f"member_{party}",
                size=15,
                color=self.colors.get(f'member_{party}', self.colors['member_I']),
                data={
                    'party': party,
                    'state': member.get('state', ''),
                    'chamber': member.get('chamber', ''),
                    'leadership': member.get('leadership_position', '')
                }
            )
            
            self.nodes.append(node)
            self.graph.add_node(member_id, **asdict(node))
        
        # Add committee nodes
        committee_nodes = {}
        for _, committee in committee_data.iterrows():
            committee_id = committee['thomas_id']
            
            # Count committee members
            committee_members = membership_data[membership_data['committee_id'] == committee.get('id', '')]
            member_count = len(committee_members)
            
            node = NetworkNode(
                id=committee_id,
                label=committee['name'],
                node_type='committee',
                size=max(20, min(60, member_count * 3)),  # Size based on membership
                color=self.colors['committee'],
                data={
                    'chamber': committee.get('chamber', ''),
                    'committee_type': committee.get('committee_type', ''),
                    'member_count': member_count
                }
            )
            
            self.nodes.append(node)
            committee_nodes[committee_id] = node
            self.graph.add_node(committee_id, **asdict(node))
        
        # Add membership edges
        for _, membership in membership_data.iterrows():
            member_id = membership['bioguide_id']
            committee_id = membership.get('committee_thomas_id', '')  # Adjust field name as needed
            
            if committee_id in committee_nodes:
                role = membership.get('role', 'Member')
                
                # Weight based on role
                weight = 3.0 if role == 'Chair' else (2.0 if role == 'Ranking Member' else 1.0)
                
                edge = NetworkEdge(
                    source=member_id,
                    target=committee_id,
                    weight=weight,
                    edge_type='membership',
                    label=f"{role} of committee",
                    data={
                        'role': role,
                        'start_date': membership.get('start_date', ''),
                        'end_date': membership.get('end_date', ''),
                        'is_current': membership.get('is_current', True)
                    }
                )
                
                self.edges.append(edge)
                self.graph.add_edge(member_id, committee_id, weight=weight, **asdict(edge))
        
        logger.info(f"Built committee network with {len(self.nodes)} nodes and {len(self.edges)} edges")
        return self.nodes, self.edges
    
    def detect_communities(self) -> Dict[str, List[str]]:
        """Detect communities in the network using modularity optimization."""
        if len(self.graph.nodes) == 0:
            return {}
        
        logger.info("Detecting communities in network")
        
        # Use Louvain method for community detection
        communities = community.greedy_modularity_communities(self.graph)
        
        community_dict = {}
        for i, comm in enumerate(communities):
            community_dict[f"community_{i}"] = list(comm)
        
        logger.info(f"Detected {len(communities)} communities")
        return community_dict
    
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality metrics for network nodes."""
        if len(self.graph.nodes) == 0:
            return {}
        
        logger.info("Calculating centrality metrics")
        
        centrality_metrics = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.graph)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Combine into single dictionary
            for node in self.graph.nodes:
                centrality_metrics[node] = {
                    'degree': degree_centrality.get(node, 0.0),
                    'betweenness': betweenness_centrality.get(node, 0.0),
                    'closeness': closeness_centrality.get(node, 0.0),
                    'pagerank': pagerank.get(node, 0.0)
                }
        
        except Exception as e:
            logger.error(f"Error calculating centrality metrics: {e}")
        
        return centrality_metrics

class InteractiveNetworkDashboard:
    """Creates interactive network visualization dashboard using Dash and Cytoscape."""
    
    def __init__(self, app_name: str = "congressional_network"):
        """Initialize interactive dashboard."""
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        self.app_name = app_name
        self.network_builder = CongressionalNetworkBuilder()
        
        # Load Cytoscape stylesheet
        cyto.load_extra_layouts()
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Congressional Trading Network Analysis", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.Label("Network Type:"),
                    dcc.Dropdown(
                        id='network-type-dropdown',
                        options=[
                            {'label': 'Member Trading Network', 'value': 'trading'},
                            {'label': 'Committee Network', 'value': 'committee'},
                            {'label': 'Stock Correlation Network', 'value': 'stock'}
                        ],
                        value='trading'
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Label("Layout Algorithm:"),
                    dcc.Dropdown(
                        id='layout-dropdown',
                        options=[
                            {'label': 'Force-directed', 'value': 'cose'},
                            {'label': 'Circular', 'value': 'circle'},
                            {'label': 'Grid', 'value': 'grid'},
                            {'label': 'Hierarchical', 'value': 'dagre'}
                        ],
                        value='cose'
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Label("Node Size By:"),
                    dcc.Dropdown(
                        id='node-size-dropdown',
                        options=[
                            {'label': 'Trading Volume', 'value': 'volume'},
                            {'label': 'Number of Trades', 'value': 'count'},
                            {'label': 'Centrality Score', 'value': 'centrality'},
                            {'label': 'Equal Size', 'value': 'equal'}
                        ],
                        value='volume'
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Button('Refresh Network', id='refresh-button', n_clicks=0,
                              style={'marginTop': 25})
                ], className='three columns')
                
            ], className='row', style={'marginBottom': 20}),
            
            # Network visualization
            html.Div([
                cyto.Cytoscape(
                    id='network-graph',
                    layout={'name': 'cose'},
                    style={'width': '100%', 'height': '600px'},
                    elements=[],
                    stylesheet=[
                        # Node styles
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'background-color': 'data(color)',
                                'width': 'data(size)',
                                'height': 'data(size)',
                                'font-size': '8px',
                                'text-valign': 'center',
                                'text-halign': 'center'
                            }
                        },
                        # Edge styles
                        {
                            'selector': 'edge',
                            'style': {
                                'width': 'data(weight)',
                                'line-color': '#ccc',
                                'target-arrow-color': '#ccc',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier'
                            }
                        },
                        # Selected styles
                        {
                            'selector': ':selected',
                            'style': {
                                'background-color': '#ff6b6b',
                                'line-color': '#ff6b6b',
                                'border-color': '#ff6b6b',
                                'border-width': 3
                            }
                        }
                    ]
                )
            ], className='row'),
            
            # Information panels
            html.Div([
                html.Div([
                    html.H4("Node Information"),
                    html.Div(id='node-info', children="Select a node to see details")
                ], className='six columns'),
                
                html.Div([
                    html.H4("Network Statistics"),
                    html.Div(id='network-stats', children="Network statistics will appear here")
                ], className='six columns')
                
            ], className='row', style={'marginTop': 20}),
            
            # Community detection results
            html.Div([
                html.H4("Community Detection"),
                html.Div(id='community-info', children="Communities will be detected automatically")
            ], className='row', style={'marginTop': 20})
        ])
    
    def setup_callbacks(self):
        """Set up interactive callbacks."""
        
        @self.app.callback(
            [Output('network-graph', 'elements'),
             Output('network-stats', 'children'),
             Output('community-info', 'children')],
            [Input('refresh-button', 'n_clicks'),
             Input('network-type-dropdown', 'value'),
             Input('layout-dropdown', 'value'),
             Input('node-size-dropdown', 'value')]
        )
        def update_network(n_clicks, network_type, layout, node_size_by):
            # Generate sample data (in production, this would query the database)
            sample_data = self._generate_sample_data(network_type)
            
            if network_type == 'trading':
                nodes, edges = self.network_builder.build_member_trading_network(
                    sample_data['trading'], sample_data['members']
                )
            elif network_type == 'committee':
                nodes, edges = self.network_builder.build_committee_network(
                    sample_data['members'], sample_data['committees'], sample_data['memberships']
                )
            else:
                nodes, edges = [], []  # Stock network not implemented in this example
            
            # Convert to Cytoscape format
            elements = []
            
            # Add nodes
            for node in nodes:
                elements.append({
                    'data': {
                        'id': node.id,
                        'label': node.label,
                        'color': node.color,
                        'size': node.size,
                        **node.data
                    }
                })
            
            # Add edges
            for edge in edges:
                elements.append({
                    'data': {
                        'source': edge.source,
                        'target': edge.target,
                        'weight': max(1, edge.weight * 5),  # Scale for visibility
                        'label': edge.label,
                        **edge.data
                    }
                })
            
            # Calculate network statistics
            stats = self._calculate_network_stats(nodes, edges)
            stats_text = html.Div([
                html.P(f"Nodes: {stats['node_count']}"),
                html.P(f"Edges: {stats['edge_count']}"),
                html.P(f"Density: {stats['density']:.3f}"),
                html.P(f"Avg Degree: {stats['avg_degree']:.2f}")
            ])
            
            # Detect communities
            communities = self.network_builder.detect_communities()
            community_text = html.Div([
                html.P(f"Communities detected: {len(communities)}"),
                html.Ul([
                    html.Li(f"Community {i}: {len(members)} members") 
                    for i, (name, members) in enumerate(communities.items())
                ])
            ])
            
            return elements, stats_text, community_text
        
        @self.app.callback(
            Output('node-info', 'children'),
            [Input('network-graph', 'selectedNodeData')]
        )
        def display_node_info(selected_nodes):
            if not selected_nodes:
                return "Select a node to see details"
            
            node = selected_nodes[0]
            
            info_items = []
            for key, value in node.items():
                if key not in ['id', 'color', 'size']:
                    info_items.append(html.P(f"{key.replace('_', ' ').title()}: {value}"))
            
            return html.Div(info_items)
    
    def _generate_sample_data(self, network_type: str) -> Dict[str, pd.DataFrame]:
        """Generate sample data for visualization (replace with actual database queries)."""
        np.random.seed(42)
        
        # Sample members
        members_data = []
        parties = ['R', 'D', 'I']
        chambers = ['House', 'Senate']
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
        
        for i in range(50):
            members_data.append({
                'bioguide_id': f'M{i:03d}',
                'first_name': f'First{i}',
                'last_name': f'Last{i}',
                'party': np.random.choice(parties, p=[0.45, 0.53, 0.02]),
                'chamber': np.random.choice(chambers, p=[0.8, 0.2]),
                'state': np.random.choice(states),
                'leadership_position': np.random.choice([None, 'Chair', 'Ranking'], p=[0.9, 0.05, 0.05])
            })
        
        # Sample trading data
        trading_data = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        for i in range(200):
            member_id = f'M{np.random.randint(0, 50):03d}'
            symbol = np.random.choice(symbols)
            trade_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            
            trading_data.append({
                'bioguide_id': member_id,
                'symbol': symbol,
                'transaction_date': trade_date,
                'transaction_type': np.random.choice(['Purchase', 'Sale']),
                'amount_mid': np.random.lognormal(10, 1)
            })
        
        # Sample committees
        committees_data = []
        committee_names = [
            'Financial Services', 'Banking', 'Ways and Means', 'Appropriations',
            'Energy and Commerce', 'Armed Services', 'Judiciary', 'Intelligence'
        ]
        
        for i, name in enumerate(committee_names):
            committees_data.append({
                'id': i,
                'thomas_id': f'C{i:03d}',
                'name': name,
                'chamber': np.random.choice(['House', 'Senate']),
                'committee_type': 'Standing'
            })
        
        # Sample memberships
        memberships_data = []
        for i in range(100):
            memberships_data.append({
                'bioguide_id': f'M{np.random.randint(0, 50):03d}',
                'committee_id': np.random.randint(0, len(committee_names)),
                'committee_thomas_id': f'C{np.random.randint(0, len(committee_names)):03d}',
                'role': np.random.choice(['Member', 'Chair', 'Ranking Member'], p=[0.9, 0.05, 0.05]),
                'start_date': datetime.now() - timedelta(days=np.random.randint(0, 1460)),
                'is_current': True
            })
        
        return {
            'members': pd.DataFrame(members_data),
            'trading': pd.DataFrame(trading_data),
            'committees': pd.DataFrame(committees_data),
            'memberships': pd.DataFrame(memberships_data)
        }
    
    def _calculate_network_stats(self, nodes: List[NetworkNode], edges: List[NetworkEdge]) -> Dict[str, Any]:
        """Calculate basic network statistics."""
        node_count = len(nodes)
        edge_count = len(edges)
        
        if node_count > 1:
            max_possible_edges = node_count * (node_count - 1) / 2
            density = edge_count / max_possible_edges if max_possible_edges > 0 else 0
            avg_degree = (2 * edge_count) / node_count
        else:
            density = 0
            avg_degree = 0
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'density': density,
            'avg_degree': avg_degree
        }
    
    def run_server(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = True):
        """Run the dashboard server."""
        logger.info(f"Starting network visualization dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def main():
    """Test function for network visualization."""
    logging.basicConfig(level=logging.INFO)
    
    # Create and run dashboard
    dashboard = InteractiveNetworkDashboard()
    
    print("Starting Congressional Trading Network Dashboard...")
    print("Open your browser to http://127.0.0.1:8050 to view the dashboard")
    
    dashboard.run_server(debug=True)

if __name__ == "__main__":
    main()