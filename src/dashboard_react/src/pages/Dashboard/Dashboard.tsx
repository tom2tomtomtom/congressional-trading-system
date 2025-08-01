import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  TrendingUp,
  People,
  AccountBalance,
  Psychology,
  Refresh,
  InfoOutlined
} from '@mui/icons-material';

import { useAppState } from '../../contexts/AppStateContext';

// Mock data interfaces
interface DashboardMetrics {
  totalMembers: number;
  totalTrades: number;
  totalVolume: number;
  avgTradeSize: number;
  suspiciousAlerts: number;
  mlPredictions: number;
  lastUpdated: string;
}

interface RecentActivity {
  id: string;
  type: 'trade' | 'alert' | 'prediction';
  member: string;
  description: string;
  amount?: number;
  timestamp: string;
  severity?: 'low' | 'medium' | 'high';
}

const Dashboard: React.FC = () => {
  const { state, setLoading, setError, clearError } = useAppState();
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  // Mock data generation
  const generateMockData = (): { metrics: DashboardMetrics; activity: RecentActivity[] } => {
    const members = [
      'Nancy Pelosi', 'Kevin McCarthy', 'Chuck Schumer', 'Mitch McConnell',
      'Alexandria Ocasio-Cortez', 'Ted Cruz', 'Elizabeth Warren', 'Josh Hawley'
    ];

    return {
      metrics: {
        totalMembers: 535,
        totalTrades: 12847,
        totalVolume: 287650000,
        avgTradeSize: 22380,
        suspiciousAlerts: 23,
        mlPredictions: 156,
        lastUpdated: new Date().toISOString()
      },
      activity: Array.from({ length: 8 }, (_, i) => ({
        id: `activity_${i}`,
        type: ['trade', 'alert', 'prediction'][Math.floor(Math.random() * 3)] as 'trade' | 'alert' | 'prediction',
        member: members[Math.floor(Math.random() * members.length)],
        description: [
          'Purchased AAPL stock',
          'Sold TSLA holdings',
          'Options trade detected',
          'Unusual timing pattern',
          'Committee jurisdiction overlap',
          'High volume alert',
          'Predicted trade probability',
          'News sentiment correlation'
        ][Math.floor(Math.random() * 8)],
        amount: Math.floor(Math.random() * 500000) + 10000,
        timestamp: new Date(Date.now() - Math.random() * 86400000 * 7).toISOString(),
        severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as 'low' | 'medium' | 'high'
      }))
    };
  };

  const loadData = async () => {
    setLoading('trades', true);
    clearError('trades');

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockData = generateMockData();
      setMetrics(mockData.metrics);
      setRecentActivity(mockData.activity);
    } catch (error) {
      setError('trades', 'Failed to load dashboard data');
    } finally {
      setLoading('trades', false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  useEffect(() => {
    loadData();
  }, []);

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'trade': return <TrendingUp />;
      case 'alert': return <InfoOutlined />;
      case 'prediction': return <Psychology />;
      default: return <InfoOutlined />;
    }
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  if (state.loading.trades && !metrics) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Loading dashboard data...
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
            Congressional Trading Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time intelligence and analytics for congressional trading transparency
          </Typography>
        </Box>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? 'Refreshing...' : 'Refresh Data'}
        </Button>
      </Box>

      {/* Phase 2 Features Alert */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>🚀 Phase 2 Active:</strong> Advanced ML analytics, network analysis, and real-time 
          intelligence monitoring are now available. Explore the new features in the Analytics, 
          Network, and Intelligence sections.
        </Typography>
      </Alert>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <People color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  Members Tracked
                </Typography>
              </Box>
              <Typography variant="h3" color="text.primary" gutterBottom>
                {metrics ? formatNumber(metrics.totalMembers) : '---'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                All 535 congressional members
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUp color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  Total Trades
                </Typography>
              </Box>
              <Typography variant="h3" color="text.primary" gutterBottom>
                {metrics ? formatNumber(metrics.totalTrades) : '---'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                STOCK Act disclosures
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AccountBalance color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="primary">
                  Total Volume
                </Typography>
              </Box>
              <Typography variant="h3" color="text.primary" gutterBottom>
                {metrics ? formatCurrency(metrics.totalVolume) : '---'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Aggregate trading volume
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Psychology color="secondary" sx={{ mr: 1 }} />
                <Typography variant="h6" color="secondary">
                  ML Predictions
                </Typography>
                <Chip label="Phase 2" size="small" color="secondary" sx={{ ml: 1 }} />
              </Box>
              <Typography variant="h3" color="text.primary" gutterBottom>
                {metrics ? formatNumber(metrics.mlPredictions) : '---'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active predictions this week
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activity and Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="primary">
                  Recent Activity
                </Typography>
                <Tooltip title="Real-time trading activity and system alerts">
                  <IconButton size="small">
                    <InfoOutlined fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              
              {recentActivity.length > 0 ? (
                <Box>
                  {recentActivity.map((activity) => (
                    <Box
                      key={activity.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        py: 2,
                        borderBottom: '1px solid #f0f0f0',
                        '&:last-child': { borderBottom: 'none' }
                      }}
                    >
                      <Box sx={{ mr: 2, color: 'text.secondary' }}>
                        {getActivityIcon(activity.type)}
                      </Box>
                      
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {activity.member}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {activity.description}
                          {activity.amount && ` - ${formatCurrency(activity.amount)}`}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ textAlign: 'right' }}>
                        {activity.severity && (
                          <Chip
                            label={activity.severity.toUpperCase()}
                            size="small"
                            color={getSeverityColor(activity.severity) as any}
                            sx={{ mb: 0.5 }}
                          />
                        )}
                        <Typography variant="caption" color="text.secondary" display="block">
                          {new Date(activity.timestamp).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                  No recent activity to display
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" color="secondary" gutterBottom>
                🚨 Suspicious Alerts
              </Typography>
              <Typography variant="h3" color="secondary">
                {metrics ? metrics.suspiciousAlerts : '---'}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Anomalies detected this week
              </Typography>
              <Button variant="outlined" size="small" sx={{ mt: 2 }} fullWidth>
                View All Alerts
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                📊 System Status
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Data Pipeline
                </Typography>
                <LinearProgress variant="determinate" value={95} color="success" />
                <Typography variant="caption" color="success.main">
                  95% - Operational
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  ML Models
                </Typography>
                <LinearProgress variant="determinate" value={87} color="primary" />
                <Typography variant="caption" color="primary.main">
                  87% - Training
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  API Health
                </Typography>
                <LinearProgress variant="determinate" value={100} color="success" />
                <Typography variant="caption" color="success.main">
                  100% - Healthy
                </Typography>
              </Box>

              <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                Last updated: {metrics ? new Date(metrics.lastUpdated).toLocaleTimeString() : '---'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;