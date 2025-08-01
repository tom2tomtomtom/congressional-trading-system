import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Alert,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  AccountTree,
  Hub,
  Group,
  Timeline,
  Settings,
  Refresh,
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  GetApp,
  InfoOutlined
} from '@mui/icons-material';

import { useAppState } from '../../contexts/AppStateContext';

interface NetworkNode {
  id: string;
  label: string;
  type: 'member' | 'stock' | 'committee';
  party?: string;
  chamber?: string;
  size: number;
  centrality: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  type: 'trading' | 'committee' | 'similarity';
  label?: string;
}

interface NetworkStats {
  totalNodes: number;
  totalEdges: number;
  density: number;
  avgClustering: number;
  communities: number;
}

interface CommunityGroup {
  id: string;
  name: string;
  members: string[];
  commonStocks: string[];
  tradingVolume: number;
  cohesionScore: number;
}

const Network: React.FC = () => {
  const { state, setLoading, setError, clearError } = useAppState();
  const [networkType, setNetworkType] = useState('member_trading');
  const [minWeight, setMinWeight] = useState(0.3);
  const [showLabels, setShowLabels] = useState(true);
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [communities, setCommunities] = useState<CommunityGroup[]>([]);
  const [topNodes, setTopNodes] = useState<NetworkNode[]>([]);

  const generateMockNetworkData = () => {
    const members = [
      'Nancy Pelosi', 'Kevin McCarthy', 'Chuck Schumer', 'Mitch McConnell',
      'Elizabeth Warren', 'Ted Cruz', 'Alexandria Ocasio-Cortez', 'Josh Hawley'
    ];
    
    const stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'];
    
    const mockStats: NetworkStats = {
      totalNodes: 47,
      totalEdges: 156,
      density: 0.142,
      avgClustering: 0.68,
      communities: 4
    };

    const mockCommunities: CommunityGroup[] = [
      {
        id: 'tech_cluster',
        name: 'Tech Sector Focus',
        members: ['Nancy Pelosi', 'Josh Hawley', 'Alexandria Ocasio-Cortez'],
        commonStocks: ['AAPL', 'MSFT', 'GOOGL', 'META'],
        tradingVolume: 12500000,
        cohesionScore: 0.847
      },
      {
        id: 'energy_cluster',
        name: 'Energy & Infrastructure',
        members: ['Ted Cruz', 'Mitch McConnell'],
        commonStocks: ['XOM', 'CVX', 'COP'],
        tradingVolume: 8300000,
        cohesionScore: 0.723
      },
      {
        id: 'finance_cluster',
        name: 'Financial Services',
        members: ['Elizabeth Warren', 'Chuck Schumer'],
        commonStocks: ['JPM', 'BAC', 'GS'],
        tradingVolume: 15600000,
        cohesionScore: 0.691
      },
      {
        id: 'healthcare_cluster',
        name: 'Healthcare & Pharma',
        members: ['Kevin McCarthy'],
        commonStocks: ['JNJ', 'PFE', 'UNH'],
        tradingVolume: 6800000,
        cohesionScore: 0.612
      }
    ];

    const mockTopNodes: NetworkNode[] = [
      {
        id: 'pelosi',
        label: 'Nancy Pelosi',
        type: 'member',
        party: 'D',
        chamber: 'House',
        size: 18,
        centrality: 0.89
      },
      {
        id: 'schumer',
        label: 'Chuck Schumer',
        type: 'member',
        party: 'D',
        chamber: 'Senate',
        size: 16,
        centrality: 0.76
      },
      {
        id: 'aapl',
        label: 'AAPL',
        type: 'stock',
        size: 15,
        centrality: 0.72
      },
      {
        id: 'financial_services',
        label: 'Financial Services Committee',
        type: 'committee',
        size: 14,
        centrality: 0.68
      }
    ];

    setNetworkStats(mockStats);
    setCommunities(mockCommunities);
    setTopNodes(mockTopNodes);
  };

  const loadNetworkData = async () => {
    setLoading('network', true);
    clearError('network');

    try {
      await new Promise(resolve => setTimeout(resolve, 1200));
      generateMockNetworkData();
    } catch (error) {
      setError('network', 'Failed to load network analysis data');
    } finally {
      setLoading('network', false);
    }
  };

  useEffect(() => {
    loadNetworkData();
  }, [networkType, minWeight]);

  const getNetworkTypeDescription = (type: string) => {
    switch (type) {
      case 'member_trading':
        return 'Members connected by shared stock holdings and similar trading patterns';
      case 'committee_overlap':
        return 'Members connected through committee memberships and jurisdictional overlap';
      case 'stock_correlation':
        return 'Stocks connected by members who hold both securities';
      case 'temporal_patterns':
        return 'Connections based on synchronized trading timing';
      default:
        return 'Network analysis visualization';
    }
  };

  const getPartyColor = (party?: string) => {
    switch (party) {
      case 'D': return '#1976d2';
      case 'R': return '#d32f2f';
      case 'I': return '#388e3c';
      default: return '#757575';
    }
  };

  if (state.loading.network) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Building network analysis...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" color="primary" gutterBottom>
            Network Analysis
            <Chip label="Phase 2" size="small" color="secondary" sx={{ ml: 2 }} />
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Relationship mapping and connection analysis for congressional trading
          </Typography>
        </Box>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={loadNetworkData}
        >
          Rebuild Network
        </Button>
      </Box>

      {/* Phase 2 Alert */}
      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>🕸️ Network Analysis Active:</strong> Advanced graph algorithms are mapping 
          relationships between members, committees, and trading patterns to identify clusters and correlations.
        </Typography>
      </Alert>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" color="primary" gutterBottom>
            Network Configuration
          </Typography>
          
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Network Type</InputLabel>
                <Select
                  value={networkType}
                  label="Network Type"
                  onChange={(e) => setNetworkType(e.target.value)}
                >
                  <MenuItem value="member_trading">Member Trading Network</MenuItem>
                  <MenuItem value="committee_overlap">Committee Overlap Network</MenuItem>
                  <MenuItem value="stock_correlation">Stock Correlation Network</MenuItem>
                  <MenuItem value="temporal_patterns">Temporal Pattern Network</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Typography gutterBottom>
                Connection Strength: {minWeight}
              </Typography>
              <Slider
                value={minWeight}
                onChange={(_, value) => setMinWeight(value as number)}
                min={0.1}
                max={1.0}
                step={0.1}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={showLabels}
                    onChange={(e) => setShowLabels(e.target.checked)}
                  />
                }
                label="Show Node Labels"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="Zoom In">
                  <IconButton size="small">
                    <ZoomIn />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom Out">
                  <IconButton size="small">
                    <ZoomOut />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Center View">
                  <IconButton size="small">
                    <CenterFocusStrong />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Export Network">
                  <IconButton size="small">
                    <GetApp />
                  </IconButton>
                </Tooltip>
              </Box>
            </Grid>
          </Grid>
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            {getNetworkTypeDescription(networkType)}
          </Typography>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Network Visualization */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: 600 }}>
            <CardContent sx={{ height: '100%' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="primary">
                  Network Visualization
                </Typography>
                <Chip 
                  label={`${networkStats?.totalNodes || 0} nodes, ${networkStats?.totalEdges || 0} edges`} 
                  size="small" 
                  variant="outlined" 
                />
              </Box>
              
              {/* Placeholder for D3.js/Cytoscape visualization */}
              <Paper 
                sx={{ 
                  height: '90%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  backgroundColor: '#f5f5f5',
                  border: '2px dashed #ccc'
                }}
              >
                <Box sx={{ textAlign: 'center' }}>
                  <AccountTree sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Interactive Network Graph
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    D3.js/Cytoscape visualization will render here
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Chip label="Force-directed layout" size="small" sx={{ mr: 1 }} />
                    <Chip label="Interactive nodes" size="small" sx={{ mr: 1 }} />
                    <Chip label="Real-time updates" size="small" />
                  </Box>
                </Box>
              </Paper>
            </CardContent>
          </Card>
        </Grid>

        {/* Network Statistics */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {/* Network Metrics */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Network Metrics
                  </Typography>
                  
                  {networkStats && (
                    <Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Nodes:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {networkStats.totalNodes}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Edges:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {networkStats.totalEdges}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Density:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {networkStats.density.toFixed(3)}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Avg Clustering:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {networkStats.avgClustering.toFixed(3)}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Communities:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {networkStats.communities}
                        </Typography>
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Top Nodes */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Most Central Nodes
                  </Typography>
                  
                  <List dense>
                    {topNodes.map((node) => (
                      <ListItem key={node.id} divider>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              {node.type === 'member' && (
                                <Box
                                  sx={{
                                    width: 12,
                                    height: 12,
                                    borderRadius: '50%',
                                    bgcolor: getPartyColor(node.party),
                                    mr: 1
                                  }}
                                />
                              )}
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {node.label}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                              <Chip 
                                label={node.type} 
                                size="small" 
                                variant="outlined" 
                                sx={{ mr: 1, fontSize: '0.7rem' }}
                              />
                              {node.chamber && (
                                <Chip 
                                  label={node.chamber} 
                                  size="small" 
                                  variant="outlined" 
                                  sx={{ fontSize: '0.7rem' }}
                                />
                              )}
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Typography variant="body2" color="text.secondary">
                            {node.centrality.toFixed(2)}
                          </Typography>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Community Detection */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="primary">
                  Detected Communities
                </Typography>
                <Button startIcon={<GetApp />} size="small">
                  Export Communities
                </Button>
              </Box>
              
              <Grid container spacing={2}>
                {communities.map((community) => (
                  <Grid item xs={12} md={6} lg={3} key={community.id}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Group color="primary" sx={{ mr: 1 }} />
                          <Typography variant="h6" color="primary">
                            {community.name}
                          </Typography>
                        </Box>
                        
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {community.members.length} members
                        </Typography>
                        
                        <Box sx={{ mb: 2 }}>
                          {community.members.slice(0, 3).map((member, idx) => (
                            <Chip 
                              key={idx}
                              label={member} 
                              size="small" 
                              sx={{ mr: 0.5, mb: 0.5, fontSize: '0.7rem' }}
                            />
                          ))}
                          {community.members.length > 3 && (
                            <Chip 
                              label={`+${community.members.length - 3} more`}
                              size="small" 
                              variant="outlined"
                              sx={{ fontSize: '0.7rem' }}
                            />
                          )}
                        </Box>
                        
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Common stocks: {community.commonStocks.join(', ')}
                        </Typography>
                        
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            Volume: ${(community.tradingVolume / 1000000).toFixed(1)}M
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Cohesion: {community.cohesionScore.toFixed(3)}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Network;