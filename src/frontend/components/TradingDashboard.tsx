/**
 * Congressional Trading Intelligence Dashboard - Main Component
 * Advanced React TypeScript implementation with real-time data
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Container, Grid, Paper, Typography, Card, CardContent,
  Alert, Chip, IconButton, Tabs, Tab, CircularProgress,
  Button, Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, MenuItem, FormControl, InputLabel, Select
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  TrendingUp as TrendingUpIcon,
  Person as PersonIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  FilterList as FilterIcon
} from '@mui/icons-material';
import { 
  Chart as ChartJS, 
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, ArcElement 
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format, subDays } from 'date-fns';

import { apiClient } from '../services/apiClient';
import { TradeAlert, Member, Trade, TradeStatistics } from '../types/api';
import { useAuth } from '../contexts/AuthContext';
import { useNotifications } from '../hooks/useNotifications';
import { AlertsList } from './AlertsList';
import { TradesList } from './TradesList';
import { MemberProfiles } from './MemberProfiles';
import { NetworkAnalysis } from './NetworkAnalysis';
import { RealTimeMonitor } from './RealTimeMonitor';

// Register Chart.js components
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, ArcElement
);

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role=\"tabpanel\"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const TradingDashboard: React.FC = () => {
  const { user } = useAuth();
  const { showNotification } = useNotifications();
  const queryClient = useQueryClient();
  
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [dateRange, setDateRange] = useState({
    startDate: format(subDays(new Date(), 30), 'yyyy-MM-dd'),
    endDate: format(new Date(), 'yyyy-MM-dd')
  });
  const [filters, setFilters] = useState({
    member: '',
    symbol: '',
    minAmount: '',
    alertLevel: ''
  });
  const [filterDialogOpen, setFilterDialogOpen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // API Queries
  const { data: statistics, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ['trade-statistics'],
    queryFn: () => apiClient.get<TradeStatistics>('/trades/statistics'),
    refetchInterval: autoRefresh ? 30000 : false, // 30 seconds
    staleTime: 60000 // 1 minute
  });

  const { data: alerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['alerts', { level: filters.alertLevel, ...dateRange }],
    queryFn: () => apiClient.get<TradeAlert[]>('/trades/alerts', {
      params: {
        level: filters.alertLevel || undefined,
        date_from: dateRange.startDate,
        date_to: dateRange.endDate,
        per_page: 50
      }
    }),
    refetchInterval: autoRefresh ? 10000 : false // 10 seconds
  });

  const { data: recentTrades, isLoading: tradesLoading } = useQuery({
    queryKey: ['recent-trades', { ...filters, ...dateRange }],
    queryFn: () => apiClient.get<Trade[]>('/trades', {
      params: {
        per_page: 20,
        sort: 'transaction_date',
        order: 'desc',
        date_from: dateRange.startDate,
        date_to: dateRange.endDate,
        member_id: filters.member || undefined,
        symbol: filters.symbol || undefined,
        amount_min: filters.minAmount || undefined
      }
    }),
    refetchInterval: autoRefresh ? 30000 : false
  });

  // Manual refresh mutation
  const refreshMutation = useMutation({
    mutationFn: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['trade-statistics'] }),
        queryClient.invalidateQueries({ queryKey: ['alerts'] }),
        queryClient.invalidateQueries({ queryKey: ['recent-trades'] })
      ]);
    },
    onSuccess: () => {
      showNotification('Dashboard refreshed successfully', 'success');
    },
    onError: () => {
      showNotification('Failed to refresh dashboard', 'error');
    }
  });

  // Tab change handler
  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  }, []);

  // Filter handlers
  const handleApplyFilters = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['alerts'] });
    queryClient.invalidateQueries({ queryKey: ['recent-trades'] });
    setFilterDialogOpen(false);
    showNotification('Filters applied successfully', 'success');
  }, [queryClient, showNotification]);

  const handleClearFilters = useCallback(() => {
    setFilters({
      member: '',
      symbol: '',
      minAmount: '',
      alertLevel: ''
    });
    setDateRange({
      startDate: format(subDays(new Date(), 30), 'yyyy-MM-dd'),
      endDate: format(new Date(), 'yyyy-MM-dd')
    });
  }, []);

  // Chart data preparations
  const suspicionScoreData = useMemo(() => {
    if (!statistics?.data) return null;
    
    return {
      labels: ['Low (0-3)', 'Medium (3-6)', 'High (6-8)', 'Extreme (8-10)'],
      datasets: [{
        label: 'Suspicion Score Distribution',
        data: [
          statistics.data.alerts?.low_count || 0,
          statistics.data.alerts?.medium_count || 0,
          statistics.data.alerts?.high_count || 0,
          statistics.data.alerts?.extreme_count || 0
        ],
        backgroundColor: [
          '#4CAF50', // Green for low
          '#FF9800', // Orange for medium  
          '#F44336', // Red for high
          '#9C27B0'  // Purple for extreme
        ]
      }]
    };
  }, [statistics]);

  const volumeTrendData = useMemo(() => {
    if (!statistics?.data?.volume_trend) return null;
    
    return {
      labels: statistics.data.volume_trend.map((item: any) => item.date),
      datasets: [{
        label: 'Daily Trading Volume ($)',
        data: statistics.data.volume_trend.map((item: any) => item.volume),
        borderColor: '#2196F3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        tension: 0.4
      }]
    };
  }, [statistics]);

  // Auto-refresh effect
  useEffect(() => {
    const interval = setInterval(() => {
      if (autoRefresh && !refreshMutation.isPending) {
        refreshMutation.mutate();
      }
    }, 60000); // 1 minute

    return () => clearInterval(interval);
  }, [autoRefresh, refreshMutation]);

  if (statsError) {
    return (
      <Container>
        <Alert severity=\"error\" sx={{ mt: 2 }}>
          Failed to load dashboard data. Please try refreshing the page.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth=\"xl\" sx={{ py: 3 }}>
      {/* Header */}
      <Box display=\"flex\" justifyContent=\"space-between\" alignItems=\"center\" mb={3}>
        <Typography variant=\"h4\" component=\"h1\" fontWeight=\"bold\">
          Congressional Trading Intelligence
        </Typography>
        <Box display=\"flex\" gap={1}>
          <Button
            variant=\"outlined\"
            startIcon={<FilterIcon />}
            onClick={() => setFilterDialogOpen(true)}
          >
            Filters
          </Button>
          <IconButton 
            onClick={() => refreshMutation.mutate()}
            disabled={refreshMutation.isPending}
            title=\"Refresh Dashboard\"
          >
            {refreshMutation.isPending ? (
              <CircularProgress size={24} />
            ) : (
              <RefreshIcon />
            )}
          </IconButton>
        </Box>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display=\"flex\" alignItems=\"center\" justifyContent=\"space-between\">
                <Box>
                  <Typography color=\"textSecondary\" gutterBottom variant=\"body2\">
                    High-Risk Alerts
                  </Typography>
                  <Typography variant=\"h4\" component=\"div\">
                    {statsLoading ? (
                      <CircularProgress size={24} />
                    ) : (
                      statistics?.data?.alerts?.high_priority_count || 0
                    )}
                  </Typography>
                </Box>
                <WarningIcon color=\"error\" fontSize=\"large\" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display=\"flex\" alignItems=\"center\" justifyContent=\"space-between\">
                <Box>
                  <Typography color=\"textSecondary\" gutterBottom variant=\"body2\">
                    Total Volume (30d)
                  </Typography>
                  <Typography variant=\"h4\" component=\"div\">
                    {statsLoading ? (
                      <CircularProgress size={24} />
                    ) : (
                      `$${(statistics?.data?.overview?.total_volume / 1000000).toFixed(1)}M`
                    )}
                  </Typography>
                </Box>
                <TrendingUpIcon color=\"success\" fontSize=\"large\" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display=\"flex\" alignItems=\"center\" justifyContent=\"space-between\">
                <Box>
                  <Typography color=\"textSecondary\" gutterBottom variant=\"body2\">
                    Active Members
                  </Typography>
                  <Typography variant=\"h4\" component=\"div\">
                    {statsLoading ? (
                      <CircularProgress size={24} />
                    ) : (
                      statistics?.data?.overview?.active_members || 0
                    )}
                  </Typography>
                </Box>
                <PersonIcon color=\"info\" fontSize=\"large\" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display=\"flex\" alignItems=\"center\" justifyContent=\"space-between\">
                <Box>
                  <Typography color=\"textSecondary\" gutterBottom variant=\"body2\">
                    Avg Suspicion Score
                  </Typography>
                  <Typography variant=\"h4\" component=\"div\">
                    {statsLoading ? (
                      <CircularProgress size={24} />
                    ) : (
                      (statistics?.data?.alerts?.average_suspicion_score || 0).toFixed(1)
                    )}
                  </Typography>
                </Box>
                <AssessmentIcon color=\"warning\" fontSize=\"large\" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant=\"h6\" gutterBottom>
              Trading Volume Trend
            </Typography>
            {volumeTrendData ? (
              <Line 
                data={volumeTrendData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { position: 'top' as const }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      ticks: {
                        callback: function(value) {
                          return '$' + (Number(value) / 1000000).toFixed(1) + 'M';
                        }
                      }
                    }
                  }
                }}
                height={350}
              />
            ) : (
              <Box display=\"flex\" justifyContent=\"center\" alignItems=\"center\" height={350}>
                <CircularProgress />
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant=\"h6\" gutterBottom>
              Risk Distribution
            </Typography>
            {suspicionScoreData ? (
              <Doughnut 
                data={suspicionScoreData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { position: 'bottom' as const }
                  }
                }}
                height={350}
              />
            ) : (
              <Box display=\"flex\" justifyContent=\"center\" alignItems=\"center\" height={350}>
                <CircularProgress />
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Main Content Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          aria-label=\"dashboard tabs\"
          variant=\"scrollable\"
          scrollButtons=\"auto\"
        >
          <Tab label=\"🚨 Real-Time Alerts\" />
          <Tab label=\"📊 Trading Analysis\" />
          <Tab label=\"🕸️ Network Insights\" />
          <Tab label=\"👥 Member Profiles\" />
          <Tab label=\"⚡ Live Monitor\" />
        </Tabs>

        <TabPanel value={activeTab} index={0}>
          <AlertsList 
            alerts={alerts?.data?.alerts || []}
            loading={alertsLoading}
            onRefresh={() => queryClient.invalidateQueries({ queryKey: ['alerts'] })}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <TradesList 
            trades={recentTrades?.data?.trades || []}
            loading={tradesLoading}
            statistics={statistics?.data}
            onRefresh={() => queryClient.invalidateQueries({ queryKey: ['recent-trades'] })}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <NetworkAnalysis />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <MemberProfiles />
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          <RealTimeMonitor 
            autoRefresh={autoRefresh}
            onToggleAutoRefresh={() => setAutoRefresh(!autoRefresh)}
          />
        </TabPanel>
      </Paper>

      {/* Filter Dialog */}
      <Dialog 
        open={filterDialogOpen} 
        onClose={() => setFilterDialogOpen(false)}
        maxWidth=\"sm\"
        fullWidth
      >
        <DialogTitle>Filter Dashboard Data</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label=\"Start Date\"
                type=\"date\"
                value={dateRange.startDate}
                onChange={(e) => setDateRange(prev => ({ ...prev, startDate: e.target.value }))}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label=\"End Date\"
                type=\"date\"
                value={dateRange.endDate}
                onChange={(e) => setDateRange(prev => ({ ...prev, endDate: e.target.value }))}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label=\"Stock Symbol\"
                value={filters.symbol}
                onChange={(e) => setFilters(prev => ({ ...prev, symbol: e.target.value }))}
                placeholder=\"e.g., AAPL\"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label=\"Min Amount ($)\"
                type=\"number\"
                value={filters.minAmount}
                onChange={(e) => setFilters(prev => ({ ...prev, minAmount: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Alert Level</InputLabel>
                <Select
                  value={filters.alertLevel}
                  label=\"Alert Level\"
                  onChange={(e) => setFilters(prev => ({ ...prev, alertLevel: e.target.value }))}
                >
                  <MenuItem value=\"\">All Levels</MenuItem>
                  <MenuItem value=\"low\">Low</MenuItem>
                  <MenuItem value=\"medium\">Medium</MenuItem>
                  <MenuItem value=\"high\">High</MenuItem>
                  <MenuItem value=\"critical\">Critical</MenuItem>
                  <MenuItem value=\"extreme\">Extreme</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClearFilters}>Clear All</Button>
          <Button onClick={() => setFilterDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleApplyFilters} variant=\"contained\">
            Apply Filters
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};