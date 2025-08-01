# Phase 2 Network Analysis - Technical Specification

**Component**: Network Analysis & Visualization  
**Phase**: 2 - Intelligence & Analytics  
**Status**: Implemented  
**Version**: 2.0  

## Overview

The Network Analysis component transforms congressional trading data into interactive relationship maps, revealing hidden connections, trading clusters, and influence patterns among members of Congress. This specification details the graph algorithms, visualization techniques, and community detection methods implemented in Phase 2.

## Technical Architecture

### Core Components
- **Network Builder** (`src/visualizations/network_graph.py`)
- **Community Detection** (Louvain algorithm with optimization)
- **Centrality Analysis** (Multiple centrality metrics)
- **Interactive Visualization** (D3.js + Cytoscape integration)
- **React Dashboard Integration** (`src/pages/Network/Network.tsx`)

### Graph Theory Foundation
The system employs multiple graph representations to capture different relationship types:

1. **Member Trading Network**: Nodes = Members, Edges = Shared stock holdings
2. **Committee Overlap Network**: Nodes = Members, Edges = Committee co-membership
3. **Stock Correlation Network**: Nodes = Stocks, Edges = Members who hold both
4. **Temporal Pattern Network**: Nodes = Members, Edges = Synchronized trading timing

## Implementation Details

### 1. Network Construction

#### 1.1 Congressional Network Builder
**File**: `src/visualizations/network_graph.py`

**Class Architecture**:
```python
class CongressionalNetworkBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self.member_data = {}
        self.trading_data = {}
        self.committee_data = {}
        
    def build_member_trading_network(self, trading_data, member_data):
        """Build network based on shared stock holdings"""
        return self._construct_trading_graph(trading_data, member_data)
    
    def build_committee_network(self, committee_data):
        """Build network based on committee co-membership"""
        return self._construct_committee_graph(committee_data)
```

#### 1.2 Trading Network Construction
**Algorithm**: Shared stock holdings with weighted edges

```python
def _construct_trading_graph(self, trading_data, member_data):
    # Create nodes for each congressional member
    for member in member_data:
        self.graph.add_node(
            member['bioguide_id'],
            name=member['full_name'],
            party=member['party'],
            chamber=member['chamber'],
            node_type='member'
        )
    
    # Calculate shared stock holdings
    member_stocks = self._group_stocks_by_member(trading_data)
    
    # Create edges based on stock overlap
    for member1, stocks1 in member_stocks.items():
        for member2, stocks2 in member_stocks.items():
            if member1 != member2:
                shared_stocks = set(stocks1) & set(stocks2)
                if len(shared_stocks) > 0:
                    # Weight by number of shared stocks and trading volume
                    weight = self._calculate_edge_weight(
                        shared_stocks, trading_data, member1, member2
                    )
                    
                    if weight >= self.min_weight_threshold:
                        self.graph.add_edge(
                            member1, member2,
                            weight=weight,
                            shared_stocks=list(shared_stocks),
                            edge_type='trading_similarity'
                        )
    
    return self.graph
```

#### 1.3 Edge Weight Calculation
**Multi-factor weighting system**:

```python
def _calculate_edge_weight(self, shared_stocks, trading_data, member1, member2):
    weight_factors = {
        'stock_overlap': 0.4,      # Number of shared stocks
        'volume_similarity': 0.3,   # Similar trading volumes
        'timing_correlation': 0.2,  # Synchronized trading timing
        'direction_alignment': 0.1  # Buy/sell pattern similarity
    }
    
    # Stock overlap factor
    overlap_score = len(shared_stocks) / max(
        len(self.member_stocks[member1]), 
        len(self.member_stocks[member2])
    )
    
    # Volume similarity factor
    volume_similarity = self._calculate_volume_similarity(
        trading_data, member1, member2, shared_stocks
    )
    
    # Timing correlation factor
    timing_correlation = self._calculate_timing_correlation(
        trading_data, member1, member2, shared_stocks
    )
    
    # Direction alignment factor
    direction_alignment = self._calculate_direction_alignment(
        trading_data, member1, member2, shared_stocks
    )
    
    # Combined weighted score
    final_weight = (
        weight_factors['stock_overlap'] * overlap_score +
        weight_factors['volume_similarity'] * volume_similarity +
        weight_factors['timing_correlation'] * timing_correlation +
        weight_factors['direction_alignment'] * direction_alignment
    )
    
    return final_weight
```

### 2. Community Detection

#### 2.1 Louvain Algorithm Implementation
**Purpose**: Identify clusters of congressional members with similar trading patterns

```python
class CommunityDetector:
    def __init__(self, resolution=1.0):
        self.resolution = resolution
        self.communities = {}
        
    def detect_communities(self, graph):
        """Apply Louvain algorithm for community detection"""
        # Initial community assignment (each node in its own community)
        communities = {node: i for i, node in enumerate(graph.nodes())}
        
        # Iterative optimization
        improved = True
        while improved:
            improved = False
            
            for node in graph.nodes():
                # Calculate modularity gain for moving node to neighbor communities
                best_community, best_gain = self._find_best_community_move(
                    graph, node, communities
                )
                
                if best_gain > 0:
                    communities[node] = best_community
                    improved = True
        
        return self._format_communities(communities, graph)
```

#### 2.2 Modularity Optimization
**Metric**: Newman's modularity for community quality assessment

```python
def _calculate_modularity(self, graph, communities):
    """Calculate Newman's modularity Q"""
    m = graph.number_of_edges()  # Total number of edges
    modularity = 0.0
    
    for community_id in set(communities.values()):
        community_nodes = [node for node, comm in communities.items() 
                          if comm == community_id]
        
        # Internal edges within community
        internal_edges = 0
        total_degree = 0
        
        for node in community_nodes:
            total_degree += graph.degree(node)
            
            for neighbor in graph.neighbors(node):
                if neighbor in community_nodes:
                    internal_edges += 1
        
        internal_edges /= 2  # Each edge counted twice
        
        # Modularity contribution from this community
        modularity += (internal_edges / m) - ((total_degree / (2 * m)) ** 2)
    
    return modularity
```

#### 2.3 Community Characterization
**Analysis**: Identify common characteristics within each community

```python
def characterize_communities(self, communities, member_data, trading_data):
    """Analyze community characteristics"""
    community_profiles = {}
    
    for comm_id, members in communities.items():
        # Party distribution
        party_dist = self._calculate_party_distribution(members, member_data)
        
        # Chamber distribution
        chamber_dist = self._calculate_chamber_distribution(members, member_data)
        
        # Common stock preferences
        common_stocks = self._find_common_stocks(members, trading_data)
        
        # Trading volume statistics
        volume_stats = self._calculate_volume_statistics(members, trading_data)
        
        # Geographic distribution
        state_distribution = self._calculate_state_distribution(members, member_data)
        
        # Committee overlap analysis
        committee_overlap = self._analyze_committee_overlap(members, member_data)
        
        community_profiles[comm_id] = {
            'name': self._generate_community_name(common_stocks, party_dist),
            'members': members,
            'party_distribution': party_dist,
            'chamber_distribution': chamber_dist,
            'common_stocks': common_stocks,
            'volume_statistics': volume_stats,
            'state_distribution': state_distribution,
            'committee_overlap': committee_overlap,
            'cohesion_score': self._calculate_cohesion_score(members, trading_data)
        }
    
    return community_profiles
```

### 3. Centrality Analysis

#### 3.1 Multiple Centrality Metrics
**Purpose**: Identify most influential members in different network contexts

```python
class CentralityAnalyzer:
    def __init__(self, graph):
        self.graph = graph
        
    def calculate_all_centralities(self):
        """Calculate multiple centrality metrics"""
        centralities = {
            'betweenness': nx.betweenness_centrality(self.graph, weight='weight'),
            'closeness': nx.closeness_centrality(self.graph, distance='weight'),
            'eigenvector': nx.eigenvector_centrality(self.graph, weight='weight'),
            'degree': nx.degree_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph, weight='weight')
        }
        
        # Combine centralities into composite influence score
        composite_scores = self._calculate_composite_influence(centralities)
        
        return centralities, composite_scores
```

#### 3.2 Betweenness Centrality
**Interpretation**: Members who serve as bridges between different trading groups

```python
def analyze_betweenness_centrality(self, centrality_scores):
    """Analyze bridge members in trading network"""
    # Sort members by betweenness centrality
    sorted_members = sorted(
        centrality_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    bridge_members = []
    for member_id, score in sorted_members[:10]:  # Top 10 bridge members
        member_info = self.get_member_info(member_id)
        
        # Analyze which communities this member connects
        connected_communities = self._find_connected_communities(member_id)
        
        bridge_members.append({
            'member': member_info,
            'betweenness_score': score,
            'connected_communities': connected_communities,
            'bridge_strength': len(connected_communities)
        })
    
    return bridge_members
```

#### 3.3 Eigenvector Centrality
**Interpretation**: Members connected to other highly connected members

```python
def analyze_eigenvector_centrality(self, centrality_scores):
    """Identify members with high-influence connections"""
    # Members with high eigenvector centrality are connected to other influential members
    influential_members = []
    
    for member_id, score in sorted(centrality_scores.items(), 
                                  key=lambda x: x[1], reverse=True)[:15]:
        member_info = self.get_member_info(member_id)
        
        # Analyze neighbor influence
        neighbor_influence = self._calculate_neighbor_influence(member_id)
        
        influential_members.append({
            'member': member_info,
            'eigenvector_score': score,
            'neighbor_influence': neighbor_influence,
            'influence_network_size': len(list(self.graph.neighbors(member_id)))
        })
    
    return influential_members
```

### 4. Interactive Visualization

#### 4.1 D3.js Integration
**React Component**: `src/pages/Network/Network.tsx`

**Visualization Features**:
- **Force-directed layout** with customizable physics parameters
- **Interactive nodes** with hover effects and click events
- **Dynamic filtering** by party, chamber, community, or centrality
- **Zoom and pan** controls for network navigation
- **Edge bundling** for complex networks with many connections

```tsx
// Network visualization configuration
const networkConfig = {
  layout: {
    type: 'force-directed',
    gravity: 0.1,
    linkDistance: 100,
    linkStrength: 0.5,
    chargeStrength: -300
  },
  nodes: {
    size: (node) => 10 + (node.centrality * 20),
    color: (node) => getPartyColor(node.party),
    label: (node) => showLabels ? node.name : ''
  },
  edges: {
    width: (edge) => 1 + (edge.weight * 5),
    opacity: 0.6,
    color: '#999999'
  }
};
```

#### 4.2 Cytoscape.js Integration
**Purpose**: Advanced graph layouts and analysis tools

```javascript
// Cytoscape configuration for complex network analysis
const cytoscapeElements = [
  // Nodes
  ...members.map(member => ({
    data: {
      id: member.bioguide_id,
      label: member.full_name,
      party: member.party,
      chamber: member.chamber,
      centrality: member.centrality_score
    },
    classes: `member ${member.party.toLowerCase()}`
  })),
  
  // Edges
  ...connections.map(connection => ({
    data: {
      id: `${connection.source}-${connection.target}`,
      source: connection.source,
      target: connection.target,
      weight: connection.weight,
      shared_stocks: connection.shared_stocks
    },
    classes: 'trading-connection'
  }))
];

// Layout algorithms
const layoutOptions = {
  'force-directed': { name: 'cose', animate: true },
  'circular': { name: 'circle', radius: 200 },
  'hierarchical': { name: 'dagre', rankDir: 'TB' },
  'community-based': { name: 'cola', flow: { axis: 'y' } }
};
```

### 5. Network Analysis Algorithms

#### 5.1 Shortest Path Analysis
**Purpose**: Measure relationship distances between members

```python
def analyze_shortest_paths(self, source_member, target_member=None):
    """Calculate shortest paths between members"""
    if target_member:
        # Single shortest path
        try:
            path = nx.shortest_path(
                self.graph, source_member, target_member, weight='weight'
            )
            path_length = nx.shortest_path_length(
                self.graph, source_member, target_member, weight='weight'
            )
            
            return {
                'path': path,
                'length': path_length,
                'intermediary_members': path[1:-1]
            }
        except nx.NetworkXNoPath:
            return {'path': None, 'length': float('inf')}
    
    else:
        # All shortest paths from source
        paths = nx.single_source_shortest_path_length(
            self.graph, source_member, weight='weight'
        )
        
        return sorted(paths.items(), key=lambda x: x[1])
```

#### 5.2 Clustering Coefficient
**Purpose**: Measure local network density and group cohesion

```python
def calculate_clustering_metrics(self):
    """Calculate clustering coefficients for network analysis"""
    # Overall clustering coefficient
    global_clustering = nx.average_clustering(self.graph, weight='weight')
    
    # Individual clustering coefficients
    local_clustering = nx.clustering(self.graph, weight='weight')
    
    # Community-level clustering
    community_clustering = {}
    for comm_id, members in self.communities.items():
        subgraph = self.graph.subgraph(members)
        community_clustering[comm_id] = nx.average_clustering(subgraph, weight='weight')
    
    return {
        'global_clustering': global_clustering,
        'local_clustering': local_clustering,
        'community_clustering': community_clustering,
        'highly_clustered_members': sorted(
            local_clustering.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
    }
```

#### 5.3 Network Density Analysis
**Purpose**: Assess overall connectivity and relationship strength

```python
def analyze_network_density(self):
    """Calculate network density metrics"""
    # Basic density (actual edges / possible edges)
    basic_density = nx.density(self.graph)
    
    # Weighted density (sum of weights / max possible weighted sum)
    total_weight = sum(data['weight'] for _, _, data in self.graph.edges(data=True))
    max_possible_weight = self.graph.number_of_edges()  # Assuming max weight = 1
    weighted_density = total_weight / max_possible_weight if max_possible_weight > 0 else 0
    
    # Component analysis
    components = list(nx.connected_components(self.graph))
    largest_component_size = len(max(components, key=len)) if components else 0
    
    # Density by party
    party_densities = {}
    for party in ['D', 'R', 'I']:
        party_members = [node for node, data in self.graph.nodes(data=True) 
                        if data.get('party') == party]
        if len(party_members) > 1:
            party_subgraph = self.graph.subgraph(party_members)
            party_densities[party] = nx.density(party_subgraph)
    
    return {
        'basic_density': basic_density,
        'weighted_density': weighted_density,
        'number_of_components': len(components),
        'largest_component_size': largest_component_size,
        'party_densities': party_densities,
        'connectivity_ratio': largest_component_size / self.graph.number_of_nodes()
    }
```

### 6. Real-time Network Updates

#### 6.1 Dynamic Network Modification
**Purpose**: Update network structure as new trading data arrives

```python
class DynamicNetworkUpdater:
    def __init__(self, network_builder):
        self.network_builder = network_builder
        self.update_queue = []
        
    def add_new_trade(self, trade_data):
        """Process new trading disclosure and update network"""
        member_id = trade_data['bioguide_id']
        ticker = trade_data['ticker']
        
        # Update member's stock portfolio
        self._update_member_portfolio(member_id, ticker, trade_data)
        
        # Recalculate affected edges
        affected_edges = self._find_affected_edges(member_id, ticker)
        
        # Update edge weights
        for edge in affected_edges:
            new_weight = self._recalculate_edge_weight(edge)
            self.network_builder.graph.edges[edge]['weight'] = new_weight
        
        # Check for new edges
        potential_new_connections = self._find_potential_new_connections(member_id, ticker)
        
        for connection in potential_new_connections:
            if self._meets_connection_threshold(connection):
                self._add_new_edge(connection)
        
        # Queue for community recalculation
        self.update_queue.append({
            'type': 'trade_update',
            'member_id': member_id,
            'affected_edges': affected_edges,
            'timestamp': datetime.utcnow()
        })
```

#### 6.2 Incremental Community Updates
**Purpose**: Efficiently update communities without full recalculation

```python
def incremental_community_update(self, updated_edges):
    """Update communities incrementally based on edge changes"""
    # Identify communities affected by edge changes
    affected_communities = set()
    
    for edge in updated_edges:
        node1, node2 = edge
        comm1 = self.node_to_community[node1]
        comm2 = self.node_to_community[node2]
        affected_communities.add(comm1)
        affected_communities.add(comm2)
    
    # Recalculate modularity for affected communities only
    for community_id in affected_communities:
        community_nodes = self.communities[community_id]
        
        # Check if any nodes should move to different communities
        for node in community_nodes:
            best_community, modularity_gain = self._find_best_community_move(
                self.graph, node, self.node_to_community
            )
            
            if modularity_gain > self.modularity_threshold:
                self._move_node_to_community(node, best_community)
    
    # Update community characterizations
    self._update_community_profiles(affected_communities)
```

### 7. Performance Optimization

#### 7.1 Graph Algorithms Optimization
**Techniques for handling large networks (535+ members)**

```python
class OptimizedNetworkAnalyzer:
    def __init__(self, use_parallel=True, chunk_size=50):
        self.use_parallel = use_parallel
        self.chunk_size = chunk_size
        
    def parallel_centrality_calculation(self, graph):
        """Calculate centralities in parallel for large networks"""
        if self.use_parallel and graph.number_of_nodes() > 100:
            # Split nodes into chunks for parallel processing
            node_chunks = [
                list(graph.nodes())[i:i + self.chunk_size] 
                for i in range(0, graph.number_of_nodes(), self.chunk_size)
            ]
            
            # Parallel betweenness centrality calculation
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self._calculate_chunk_betweenness, graph, chunk)
                    for chunk in node_chunks
                ]
                
                centrality_results = {}
                for future in as_completed(futures):
                    chunk_results = future.result()
                    centrality_results.update(chunk_results)
            
            return centrality_results
        else:
            # Standard sequential calculation for smaller networks
            return nx.betweenness_centrality(graph, weight='weight')
```

#### 7.2 Memory Management
**Strategies for efficient large-graph processing**

```python
def optimize_memory_usage(self):
    """Optimize memory usage for large network analysis"""
    # Use sparse matrix representations
    adjacency_matrix = nx.adjacency_matrix(self.graph, weight='weight')
    
    # Clear unnecessary node/edge attributes during computation
    essential_attributes = {'weight', 'party', 'chamber'}
    
    for node, data in self.graph.nodes(data=True):
        for attr in list(data.keys()):
            if attr not in essential_attributes:
                del data[attr]
    
    # Use memory-efficient algorithms
    self.graph = self._convert_to_efficient_representation(self.graph)
    
    # Garbage collection after major operations
    import gc
    gc.collect()
```

### 8. Validation and Testing

#### 8.1 Network Quality Metrics
**Validation of network construction accuracy**

```python
def validate_network_quality(self):
    """Validate network construction and analysis quality"""
    validation_results = {
        'edge_weight_distribution': self._analyze_edge_weight_distribution(),
        'community_quality': self._validate_community_detection(),
        'centrality_correlation': self._test_centrality_correlations(),
        'network_stability': self._test_network_stability()
    }
    
    return validation_results

def _validate_community_detection(self):
    """Validate community detection quality"""
    # Modularity score (higher is better)
    modularity_score = self._calculate_modularity(self.graph, self.communities)
    
    # Silhouette analysis for community cohesion
    silhouette_scores = self._calculate_community_silhouette_scores()
    
    # Community size distribution
    community_sizes = [len(members) for members in self.communities.values()]
    
    return {
        'modularity': modularity_score,
        'average_silhouette': np.mean(silhouette_scores),
        'community_size_variance': np.var(community_sizes),
        'number_of_communities': len(self.communities)
    }
```

#### 8.2 Algorithmic Testing
**Unit tests for network algorithms**

```python
import unittest

class TestNetworkAnalysis(unittest.TestCase):
    def setUp(self):
        self.network_builder = CongressionalNetworkBuilder()
        self.sample_data = self._generate_sample_data()
        
    def test_edge_weight_calculation(self):
        """Test edge weight calculation accuracy"""
        shared_stocks = ['AAPL', 'MSFT']
        weight = self.network_builder._calculate_edge_weight(
            shared_stocks, self.sample_data, 'member1', 'member2'
        )
        
        self.assertGreaterEqual(weight, 0)
        self.assertLessEqual(weight, 1)
        
    def test_community_detection(self):
        """Test community detection algorithm"""
        graph = self._create_test_graph()
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)
        
        # Validate community structure
        self.assertGreater(len(communities), 1)
        self.assertLess(len(communities), graph.number_of_nodes())
        
    def test_centrality_calculations(self):
        """Test centrality metric calculations"""
        graph = self._create_test_graph()
        analyzer = CentralityAnalyzer(graph)
        centralities, composite = analyzer.calculate_all_centralities()
        
        # Validate centrality scores
        for metric, scores in centralities.items():
            for node, score in scores.items():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)
```

This comprehensive network analysis specification provides the foundation for understanding congressional trading relationships, identifying influence patterns, and revealing hidden connections in the political-financial ecosystem. The combination of advanced graph algorithms, interactive visualizations, and real-time updates creates a powerful tool for transparency and accountability research.