#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Phase 2
Network Analysis for Committee-Trading Correlations

This module implements network analysis to identify relationships between
committee memberships, trading patterns, and influence networks.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Any
import json
from pathlib import Path

# Network Analysis
import networkx as nx
from networkx.algorithms import community

# Database and Data Processing
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml

# Visualization and Analysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    """
    Advanced network analysis for congressional trading intelligence.
    Analyzes relationships between members, committees, and trading patterns.
    """
    
    def __init__(self, config_path: str = "config/database.yml"):
        """Initialize the network analyzer."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.db_config = self.config.get('development', {}).get('database', {})
        
        # Network graphs
        self.member_network = nx.Graph()
        self.committee_network = nx.Graph()
        self.trading_network = nx.Graph()
        self.influence_network = nx.DiGraph()
        
        # Analysis results
        self.network_metrics = {}
        self.community_structure = {}
        self.influence_scores = {}
        
        # Output directory
        self.output_dir = Path("analysis/network")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def load_network_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data for network analysis.
        
        Returns:
            Tuple of (members_df, committees_df, trades_df)
        """
        logger.info("Loading network analysis data...")
        
        try:
            conn = self._get_db_connection()
            
            # Load members data
            members_query = """
            SELECT 
                bioguide_id,
                full_name,
                party,
                chamber,
                state,
                served_from,
                (SELECT COUNT(*) FROM trades WHERE bioguide_id = m.bioguide_id) as total_trades,
                (SELECT SUM(amount_mid) FROM trades WHERE bioguide_id = m.bioguide_id) as total_volume
            FROM members m
            """
            members_df = pd.read_sql_query(members_query, conn)
            
            # Load committee relationships
            committees_query = """
            SELECT 
                c.id as committee_id,
                c.thomas_id,
                c.name as committee_name,
                c.chamber,
                c.committee_type,
                cm.bioguide_id,
                cm.role,
                m.full_name,
                m.party,
                m.state
            FROM committees c
            LEFT JOIN committee_memberships cm ON c.id = cm.committee_id
            LEFT JOIN members m ON cm.bioguide_id = m.bioguide_id
            """
            committees_df = pd.read_sql_query(committees_query, conn)
            
            # Load trading data with enhanced features
            trades_query = """
            SELECT 
                t.id,
                t.bioguide_id,
                m.full_name,
                m.party,
                m.chamber,
                m.state,
                t.symbol,
                t.transaction_type,
                t.transaction_date,
                t.filing_date,
                t.amount_mid,
                t.filing_delay_days,
                sp.close_price,
                -- Calculate performance metrics
                CASE WHEN t.transaction_type = 'Purchase' THEN
                    (sp2.close_price - sp.close_price) / sp.close_price * 100
                ELSE
                    (sp.close_price - sp2.close_price) / sp.close_price * 100
                END as performance_30d
            FROM trades t
            JOIN members m ON t.bioguide_id = m.bioguide_id
            LEFT JOIN stock_prices sp ON t.symbol = sp.symbol 
                AND sp.date <= t.transaction_date
                AND sp.date = (
                    SELECT MAX(date) FROM stock_prices 
                    WHERE symbol = t.symbol AND date <= t.transaction_date
                )
            LEFT JOIN stock_prices sp2 ON t.symbol = sp2.symbol 
                AND sp2.date >= t.transaction_date + INTERVAL '30 days'
                AND sp2.date = (
                    SELECT MIN(date) FROM stock_prices 
                    WHERE symbol = t.symbol AND date >= t.transaction_date + INTERVAL '30 days'
                )
            WHERE t.amount_mid IS NOT NULL
            ORDER BY t.transaction_date DESC
            """
            trades_df = pd.read_sql_query(trades_query, conn)
            
            conn.close()
            
            logger.info(f"Loaded {len(members_df)} members, {len(committees_df)} committee relationships, "
                       f"{len(trades_df)} trades")
            
            return members_df, committees_df, trades_df
            
        except Exception as e:
            logger.error(f"Failed to load network data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def build_member_network(self, members_df: pd.DataFrame, committees_df: pd.DataFrame) -> nx.Graph:
        """
        Build network of congressional members based on committee co-memberships.
        
        Args:
            members_df: Members data
            committees_df: Committee membership data
            
        Returns:
            NetworkX graph of member relationships
        """
        logger.info("Building congressional member network...")
        
        # Create member network
        G = nx.Graph()
        
        # Add member nodes with attributes
        for _, member in members_df.iterrows():
            G.add_node(
                member['bioguide_id'],
                name=member['full_name'],
                party=member['party'],
                chamber=member['chamber'],
                state=member['state'],
                total_trades=member.get('total_trades', 0),
                total_volume=member.get('total_volume', 0)
            )
        
        # Add edges based on committee co-memberships
        committee_members = committees_df.groupby('committee_id')['bioguide_id'].apply(list)
        
        for committee_id, members in committee_members.items():
            if len(members) > 1:
                # Create edges between all committee members
                for i, member1 in enumerate(members):
                    for member2 in members[i+1:]:
                        if member1 and member2:  # Handle null values
                            if G.has_edge(member1, member2):
                                # Increment weight for multiple shared committees
                                G[member1][member2]['weight'] += 1
                                G[member1][member2]['committees'].append(committee_id)
                            else:
                                G.add_edge(member1, member2, weight=1, committees=[committee_id])
        
        self.member_network = G
        logger.info(f"Built member network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def build_trading_network(self, trades_df: pd.DataFrame) -> nx.Graph:
        """
        Build network based on trading patterns and correlations.
        
        Args:
            trades_df: Trading data
            
        Returns:
            NetworkX graph of trading relationships
        """
        logger.info("Building trading pattern network...")
        
        G = nx.Graph()
        
        # Create trading similarity matrix
        member_trading_profiles = self._create_trading_profiles(trades_df)
        
        # Calculate similarities between members based on trading patterns
        members = list(member_trading_profiles.keys())
        similarities = {}
        
        for i, member1 in enumerate(members):
            for member2 in members[i+1:]:
                similarity = self._calculate_trading_similarity(
                    member_trading_profiles[member1],
                    member_trading_profiles[member2]
                )
                
                if similarity > 0.3:  # Threshold for meaningful similarity
                    similarities[(member1, member2)] = similarity
        
        # Add nodes and edges to graph
        for member in members:
            profile = member_trading_profiles[member]
            G.add_node(
                member,
                total_trades=profile['total_trades'],
                avg_amount=profile['avg_amount'],
                preferred_symbols=profile['top_symbols'][:3],
                avg_performance=profile['avg_performance']
            )
        
        for (member1, member2), similarity in similarities.items():
            G.add_edge(member1, member2, similarity=similarity, weight=similarity)
        
        self.trading_network = G
        logger.info(f"Built trading network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _create_trading_profiles(self, trades_df: pd.DataFrame) -> Dict:
        """Create trading profiles for each member."""
        profiles = {}
        
        for bioguide_id in trades_df['bioguide_id'].unique():
            member_trades = trades_df[trades_df['bioguide_id'] == bioguide_id]
            
            # Calculate trading statistics
            symbol_counts = member_trades['symbol'].value_counts()
            transaction_types = member_trades['transaction_type'].value_counts()
            
            profile = {
                'bioguide_id': bioguide_id,
                'total_trades': len(member_trades),
                'avg_amount': member_trades['amount_mid'].mean(),
                'total_volume': member_trades['amount_mid'].sum(),
                'top_symbols': symbol_counts.head(10).index.tolist(),
                'symbol_counts': symbol_counts.to_dict(),
                'transaction_types': transaction_types.to_dict(),
                'avg_filing_delay': member_trades['filing_delay_days'].mean(),
                'avg_performance': member_trades['performance_30d'].mean() if 'performance_30d' in member_trades.columns else 0,
                'party': member_trades['party'].iloc[0] if len(member_trades) > 0 else None,
                'chamber': member_trades['chamber'].iloc[0] if len(member_trades) > 0 else None
            }
            
            profiles[bioguide_id] = profile
        
        return profiles
    
    def _calculate_trading_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Calculate similarity between two trading profiles."""
        similarity_score = 0.0
        
        # Symbol overlap similarity
        symbols1 = set(profile1['top_symbols'])
        symbols2 = set(profile2['top_symbols'])
        if symbols1 and symbols2:
            symbol_similarity = len(symbols1.intersection(symbols2)) / len(symbols1.union(symbols2))
            similarity_score += symbol_similarity * 0.4
        
        # Trading frequency similarity
        freq_ratio = min(profile1['total_trades'], profile2['total_trades']) / max(profile1['total_trades'], profile2['total_trades'])
        similarity_score += freq_ratio * 0.3
        
        # Amount similarity
        amount_ratio = min(profile1['avg_amount'], profile2['avg_amount']) / max(profile1['avg_amount'], profile2['avg_amount'])
        similarity_score += amount_ratio * 0.2
        
        # Party similarity (bonus)
        if profile1['party'] == profile2['party']:
            similarity_score += 0.1
        
        return similarity_score
    
    def analyze_committee_trading_correlations(self, committees_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between committee memberships and trading patterns.
        
        Args:
            committees_df: Committee membership data
            trades_df: Trading data
            
        Returns:
            Dictionary of correlation analysis results
        """
        logger.info("Analyzing committee-trading correlations...")
        
        correlations = {}
        
        # Get committees with members
        committees_with_members = committees_df.dropna(subset=['bioguide_id'])
        
        for committee_id in committees_with_members['committee_id'].unique():
            committee_info = committees_df[committees_df['committee_id'] == committee_id].iloc[0]
            committee_members = committees_with_members[
                committees_with_members['committee_id'] == committee_id
            ]['bioguide_id'].tolist()
            
            if len(committee_members) < 2:
                continue
            
            # Get trades by committee members
            committee_trades = trades_df[trades_df['bioguide_id'].isin(committee_members)]
            
            if len(committee_trades) == 0:
                continue
            
            # Analyze trading patterns
            analysis = {
                'committee_name': committee_info['committee_name'],
                'committee_type': committee_info['committee_type'],
                'chamber': committee_info['chamber'],
                'member_count': len(committee_members),
                'total_trades': len(committee_trades),
                'total_volume': committee_trades['amount_mid'].sum(),
                'avg_trade_amount': committee_trades['amount_mid'].mean(),
                'top_symbols': committee_trades['symbol'].value_counts().head(5).to_dict(),
                'avg_filing_delay': committee_trades['filing_delay_days'].mean(),
                'transaction_types': committee_trades['transaction_type'].value_counts().to_dict()
            }
            
            # Calculate sector correlations
            sector_correlations = self._analyze_sector_correlations(
                committee_info['committee_name'], 
                committee_trades
            )
            analysis['sector_correlations'] = sector_correlations
            
            correlations[committee_id] = analysis
        
        logger.info(f"Analyzed correlations for {len(correlations)} committees")
        return correlations
    
    def _analyze_sector_correlations(self, committee_name: str, trades_df: pd.DataFrame) -> Dict:
        """Analyze correlations between committee jurisdiction and trading sectors."""
        # Define sector mappings (simplified)
        sector_keywords = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'CVS'],
            'Energy': ['XOM', 'CVX', 'COP'],
            'Defense': ['LMT', 'RTX', 'BA', 'NOC']
        }
        
        committee_keywords = {
            'financial': 'Financial',
            'banking': 'Financial',
            'technology': 'Technology',
            'intelligence': 'Technology',
            'energy': 'Energy',
            'health': 'Healthcare',
            'armed': 'Defense',
            'defense': 'Defense'
        }
        
        # Determine expected sector based on committee name
        expected_sector = None
        committee_lower = committee_name.lower()
        for keyword, sector in committee_keywords.items():
            if keyword in committee_lower:
                expected_sector = sector
                break
        
        if not expected_sector:
            return {'expected_sector': None, 'correlation_strength': 0.0}
        
        # Calculate actual trading in expected sector
        expected_symbols = set(sector_keywords.get(expected_sector, []))
        committee_symbols = set(trades_df['symbol'].unique())
        
        overlap = len(expected_symbols.intersection(committee_symbols))
        correlation_strength = overlap / len(expected_symbols) if expected_symbols else 0.0
        
        return {
            'expected_sector': expected_sector,
            'correlation_strength': correlation_strength,
            'overlapping_symbols': list(expected_symbols.intersection(committee_symbols)),
            'total_trades_in_sector': len(trades_df[trades_df['symbol'].isin(expected_symbols)])
        }
    
    def calculate_influence_scores(self) -> Dict[str, float]:
        """
        Calculate influence scores for members based on network position and trading activity.
        
        Returns:
            Dictionary of influence scores by member
        """
        logger.info("Calculating member influence scores...")
        
        influence_scores = {}
        
        if self.member_network.number_of_nodes() == 0:
            logger.warning("Member network not built yet")
            return influence_scores
        
        # Calculate network centrality measures
        centrality_measures = {
            'degree': nx.degree_centrality(self.member_network),
            'betweenness': nx.betweenness_centrality(self.member_network),
            'closeness': nx.closeness_centrality(self.member_network),
            'eigenvector': nx.eigenvector_centrality(self.member_network, max_iter=1000)
        }
        
        # Calculate composite influence score
        for member in self.member_network.nodes():
            node_data = self.member_network.nodes[member]
            
            # Network-based influence (40%)
            network_influence = (
                centrality_measures['degree'].get(member, 0) * 0.3 +
                centrality_measures['betweenness'].get(member, 0) * 0.3 +
                centrality_measures['closeness'].get(member, 0) * 0.2 +
                centrality_measures['eigenvector'].get(member, 0) * 0.2
            )
            
            # Trading-based influence (40%)
            total_trades = node_data.get('total_trades', 0)
            total_volume = node_data.get('total_volume', 0)
            
            # Normalize trading metrics
            max_trades = max([n.get('total_trades', 0) for n in self.member_network.nodes().values()])
            max_volume = max([n.get('total_volume', 0) for n in self.member_network.nodes().values()])
            
            trading_influence = 0
            if max_trades > 0 and max_volume > 0:
                trading_influence = (
                    (total_trades / max_trades) * 0.5 +
                    (total_volume / max_volume) * 0.5
                )
            
            # Position-based influence (20%)
            position_influence = 0
            party = node_data.get('party', '')
            chamber = node_data.get('chamber', '')
            
            # Leadership bonus (simplified)
            name = node_data.get('name', '').lower()
            if any(title in name for title in ['speaker', 'leader', 'chair']):
                position_influence = 1.0
            elif party in ['D', 'R'] and chamber == 'Senate':
                position_influence = 0.7
            elif party in ['D', 'R']:
                position_influence = 0.5
            
            # Composite score
            influence_scores[member] = (
                network_influence * 0.4 +
                trading_influence * 0.4 +
                position_influence * 0.2
            )
        
        self.influence_scores = influence_scores
        logger.info(f"Calculated influence scores for {len(influence_scores)} members")
        return influence_scores
    
    def detect_communities(self) -> Dict:
        """
        Detect communities in the member network.
        
        Returns:
            Dictionary of community analysis results
        """
        logger.info("Detecting communities in member network...")
        
        if self.member_network.number_of_nodes() == 0:
            logger.warning("Member network not built yet")
            return {}
        
        # Detect communities using multiple algorithms
        communities = {}
        
        # Louvain community detection
        try:
            louvain_communities = community.greedy_modularity_communities(self.member_network)
            communities['louvain'] = {
                'algorithm': 'Louvain',
                'num_communities': len(louvain_communities),
                'communities': [list(comm) for comm in louvain_communities],
                'modularity': community.modularity(self.member_network, louvain_communities)
            }
        except Exception as e:
            logger.warning(f"Louvain community detection failed: {e}")
        
        # Label propagation
        try:
            label_prop_communities = community.label_propagation_communities(self.member_network)
            communities['label_propagation'] = {
                'algorithm': 'Label Propagation',
                'num_communities': len(label_prop_communities),
                'communities': [list(comm) for comm in label_prop_communities],
                'modularity': community.modularity(self.member_network, label_prop_communities)
            }
        except Exception as e:
            logger.warning(f"Label propagation community detection failed: {e}")
        
        # Analyze community composition
        for alg_name, comm_data in communities.items():
            community_analysis = []
            
            for i, comm in enumerate(comm_data['communities']):
                party_dist = {}
                chamber_dist = {}
                
                for member in comm:
                    node_data = self.member_network.nodes.get(member, {})
                    party = node_data.get('party', 'Unknown')
                    chamber = node_data.get('chamber', 'Unknown')
                    
                    party_dist[party] = party_dist.get(party, 0) + 1
                    chamber_dist[chamber] = chamber_dist.get(chamber, 0) + 1
                
                community_analysis.append({
                    'community_id': i,
                    'size': len(comm),
                    'members': comm,
                    'party_distribution': party_dist,
                    'chamber_distribution': chamber_dist
                })
            
            comm_data['community_analysis'] = community_analysis
        
        self.community_structure = communities
        logger.info(f"Detected communities using {len(communities)} algorithms")
        return communities
    
    def generate_network_report(self) -> Dict:
        """
        Generate comprehensive network analysis report.
        
        Returns:
            Dictionary containing network analysis results
        """
        logger.info("Generating comprehensive network analysis report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'network_statistics': {},
            'influence_analysis': {},
            'community_analysis': {},
            'correlation_analysis': {},
            'key_findings': []
        }
        
        # Network statistics
        if self.member_network.number_of_nodes() > 0:
            report['network_statistics'] = {
                'member_network': {
                    'nodes': self.member_network.number_of_nodes(),
                    'edges': self.member_network.number_of_edges(),
                    'density': nx.density(self.member_network),
                    'average_clustering': nx.average_clustering(self.member_network),
                    'connected_components': nx.number_connected_components(self.member_network)
                }
            }
        
        if self.trading_network.number_of_nodes() > 0:
            report['network_statistics']['trading_network'] = {
                'nodes': self.trading_network.number_of_nodes(),
                'edges': self.trading_network.number_of_edges(),
                'density': nx.density(self.trading_network)
            }
        
        # Influence analysis
        if self.influence_scores:
            sorted_influence = sorted(self.influence_scores.items(), key=lambda x: x[1], reverse=True)
            report['influence_analysis'] = {
                'top_influential_members': sorted_influence[:10],
                'average_influence_score': np.mean(list(self.influence_scores.values())),
                'influence_distribution': {
                    'high_influence_count': len([s for s in self.influence_scores.values() if s > 0.7]),
                    'medium_influence_count': len([s for s in self.influence_scores.values() if 0.3 < s <= 0.7]),
                    'low_influence_count': len([s for s in self.influence_scores.values() if s <= 0.3])
                }
            }
        
        # Community analysis
        if self.community_structure:
            report['community_analysis'] = self.community_structure
        
        # Generate key findings
        findings = []
        
        if self.member_network.number_of_nodes() > 0:
            density = nx.density(self.member_network)
            if density > 0.5:
                findings.append("High network density indicates strong committee collaboration")
            elif density < 0.1:
                findings.append("Low network density suggests fragmented committee structure")
        
        if self.influence_scores:
            top_influence = max(self.influence_scores.values())
            if top_influence > 0.8:
                findings.append("Highly concentrated influence detected in network")
        
        report['key_findings'] = findings
        
        # Save report
        report_path = self.output_dir / f"network_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Network analysis report saved to {report_path}")
        return report
    
    def run_full_network_analysis(self) -> Dict:
        """
        Run complete network analysis pipeline.
        
        Returns:
            Comprehensive analysis results
        """
        logger.info("Running full network analysis pipeline...")
        
        # Load data
        members_df, committees_df, trades_df = self.load_network_data()
        
        if members_df.empty:
            logger.error("No data available for network analysis")
            return {}
        
        # Build networks
        self.build_member_network(members_df, committees_df)
        self.build_trading_network(trades_df)
        
        # Analyze correlations
        correlations = self.analyze_committee_trading_correlations(committees_df, trades_df)
        
        # Calculate influence scores
        self.calculate_influence_scores()
        
        # Detect communities
        self.detect_communities()
        
        # Generate comprehensive report
        report = self.generate_network_report()
        report['correlation_analysis'] = correlations
        
        logger.info("Full network analysis completed successfully")
        return report

def main():
    """Main execution function."""
    logger.info("Starting Congressional Trading Network Analysis...")
    
    analyzer = NetworkAnalyzer()
    
    # Run full network analysis
    results = analyzer.run_full_network_analysis()
    
    if results:
        # Display summary
        logger.info("=== NETWORK ANALYSIS SUMMARY ===")
        
        net_stats = results.get('network_statistics', {})
        if 'member_network' in net_stats:
            member_net = net_stats['member_network']
            logger.info(f"Member Network: {member_net['nodes']} nodes, {member_net['edges']} edges")
            logger.info(f"Network Density: {member_net['density']:.3f}")
        
        influence_analysis = results.get('influence_analysis', {})
        if 'top_influential_members' in influence_analysis:
            logger.info("\nTop 5 Most Influential Members:")
            for member_id, score in influence_analysis['top_influential_members'][:5]:
                member_data = analyzer.member_network.nodes.get(member_id, {})
                name = member_data.get('name', member_id)
                logger.info(f"  {name}: {score:.3f}")
        
        community_analysis = results.get('community_analysis', {})
        if 'louvain' in community_analysis:
            louvain = community_analysis['louvain']
            logger.info(f"\nCommunity Structure: {louvain['num_communities']} communities detected")
            logger.info(f"Modularity Score: {louvain['modularity']:.3f}")
        
        key_findings = results.get('key_findings', [])
        if key_findings:
            logger.info("\nKey Findings:")
            for finding in key_findings:
                logger.info(f"  • {finding}")
    
    logger.info("Network analysis completed successfully!")

if __name__ == "__main__":
    main()