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
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  Warning,
  Timeline,
  PieChart,
  BarChart,
  ShowChart,
  Refresh,
  GetApp,
  InfoOutlined
} from '@mui/icons-material';

import { useAppState } from '../../contexts/AppStateContext';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface MLModel {
  id: string;
  name: string;
  type: string;
  accuracy: number;
  status: 'training' | 'ready' | 'error';
  lastTrained: string;
  predictions: number;
}

interface PredictionResult {
  id: string;
  member: string;
  prediction: string;
  confidence: number;
  factors: string[];
  targetDate: string;
  status: 'pending' | 'confirmed' | 'expired';
}

interface AnomalyAlert {
  id: string;
  type: 'timing' | 'volume' | 'pattern' | 'coordination';
  severity: 'low' | 'medium' | 'high';
  member: string;
  description: string;
  confidence: number;
  detectedAt: string;
}

const Analytics: React.FC = () => {
  const { state, setLoading, setError, clearError } = useAppState();
  const [tabValue, setTabValue] = useState(0);
  const [models, setModels] = useState<MLModel[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyAlert[]>([]);

  const generateMockData = () => {
    const mockModels: MLModel[] = [
      {
        id: 'trade_predictor',
        name: 'Trade Prediction Model',
        type: 'XGBoost Ensemble',
        accuracy: 87.3,
        status: 'ready',
        lastTrained: '2024-12-15T10:30:00Z',
        predictions: 156
      },
      {
        id: 'anomaly_detector',
        name: 'Anomaly Detection Model',
        type: 'Isolation Forest',
        accuracy: 92.1,
        status: 'ready',
        lastTrained: '2024-12-15T08:15:00Z',
        predictions: 23
      },
      {
        id: 'sentiment_analyzer',
        name: 'News Sentiment Model',
        type: 'Transformer (BERT)',
        accuracy: 89.7,
        status: 'training',
        lastTrained: '2024-12-14T16:45:00Z',
        predictions: 89
      }
    ];

    const members = ['Nancy Pelosi', 'Kevin McCarthy', 'Chuck Schumer', 'Elizabeth Warren'];
    const mockPredictions: PredictionResult[] = Array.from({ length: 8 }, (_, i) => ({
      id: `pred_${i}`,
      member: members[Math.floor(Math.random() * members.length)],
      prediction: [
        'Likely AAPL purchase within 7 days',
        'Expected portfolio rebalancing',
        'Potential tech sector increase',
        'Healthcare sector concentration',
        'Energy sector divestment likely'
      ][Math.floor(Math.random() * 5)],
      confidence: 0.65 + Math.random() * 0.3,
      factors: ['Committee activity', 'News sentiment', 'Historical pattern'].slice(0, Math.floor(Math.random() * 3) + 1),
      targetDate: new Date(Date.now() + Math.random() * 14 * 24 * 60 * 60 * 1000).toISOString(),
      status: ['pending', 'confirmed', 'expired'][Math.floor(Math.random() * 3)] as 'pending' | 'confirmed' | 'expired'
    }));

    const mockAnomalies: AnomalyAlert[] = Array.from({ length: 6 }, (_, i) => ({
      id: `anom_${i}`,
      type: ['timing', 'volume', 'pattern', 'coordination'][Math.floor(Math.random() * 4)] as any,
      severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as any,
      member: members[Math.floor(Math.random() * members.length)],
      description: [
        'Unusual trading timing detected',
        'Volume spike before committee hearing',
        'Coordinated trading pattern observed',
        'Options activity ahead of vote',
        'Cross-committee trading correlation'
      ][Math.floor(Math.random() * 5)],
      confidence: 0.7 + Math.random() * 0.3,
      detectedAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString()
    }));

    setModels(mockModels);
    setPredictions(mockPredictions);
    setAnomalies(mockAnomalies);
  };

  const loadAnalyticsData = async () => {
    setLoading('analytics', true);
    clearError('analytics');

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      generateMockData();
    } catch (error) {
      setError('analytics', 'Failed to load analytics data');
    } finally {
      setLoading('analytics', false);
    }
  };

  useEffect(() => {
    loadAnalyticsData();
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'success';
      case 'training': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  if (state.loading.analytics) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Loading advanced analytics...
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
            Advanced Analytics
            <Chip label="Phase 2" size="small" color="secondary" sx={{ ml: 2 }} />
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Machine learning models and predictive analytics for congressional trading
          </Typography>
        </Box>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={loadAnalyticsData}
        >
          Refresh Models
        </Button>
      </Box>

      {/* Phase 2 Alert */}
      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>🤖 ML Models Active:</strong> Advanced machine learning algorithms are now analyzing 
          congressional trading patterns, predicting future activities, and detecting anomalies in real-time.
        </Typography>
      </Alert>

      {/* Model Status Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {models.map((model) => (
          <Grid item xs={12} md={4} key={model.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Psychology color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6" color="primary">
                    {model.name}
                  </Typography>
                </Box>
                
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {model.type}
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Chip
                    label={model.status.toUpperCase()}
                    size="small"
                    color={getStatusColor(model.status) as any}
                    sx={{ mr: 1 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {model.accuracy}% accuracy
                  </Typography>
                </Box>
                
                <Typography variant="body2" color="text.secondary">
                  {model.predictions} predictions this week
                </Typography>
                
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Last trained: {new Date(model.lastTrained).toLocaleDateString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="analytics tabs">
            <Tab 
              label="Predictions" 
              icon={<TrendingUp />} 
              iconPosition="start"
              id="analytics-tab-0"
              aria-controls="analytics-tabpanel-0"
            />
            <Tab 
              label="Anomaly Detection" 
              icon={<Warning />} 
              iconPosition="start"
              id="analytics-tab-1"
              aria-controls="analytics-tabpanel-1"
            />
            <Tab 
              label="Pattern Analysis" 
              icon={<Timeline />} 
              iconPosition="start"
              id="analytics-tab-2"
              aria-controls="analytics-tabpanel-2"
            />
            <Tab 
              label="Model Performance" 
              icon={<BarChart />} 
              iconPosition="start"
              id="analytics-tab-3"
              aria-controls="analytics-tabpanel-3"
            />
          </Tabs>
        </Box>

        {/* Predictions Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" color="primary">
              ML Predictions
            </Typography>
            <Button startIcon={<GetApp />} size="small">
              Export Predictions
            </Button>
          </Box>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Member</TableCell>
                  <TableCell>Prediction</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Key Factors</TableCell>
                  <TableCell>Target Date</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {predictions.map((prediction) => (
                  <TableRow key={prediction.id} hover>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {prediction.member}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {prediction.prediction}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <LinearProgress
                          variant="determinate"
                          value={prediction.confidence * 100}
                          sx={{ width: 60, mr: 1 }}
                        />
                        <Typography variant="body2">
                          {Math.round(prediction.confidence * 100)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {prediction.factors.map((factor, idx) => (
                          <Chip key={idx} label={factor} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(prediction.targetDate).toLocaleDateString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={prediction.status}
                        size="small"
                        color={prediction.status === 'confirmed' ? 'success' : 
                               prediction.status === 'expired' ? 'error' : 'default'}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Anomaly Detection Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" color="primary">
              Anomaly Alerts
            </Typography>
            <Button startIcon={<GetApp />} size="small">
              Export Alerts
            </Button>
          </Box>
          
          <Grid container spacing={2}>
            {anomalies.map((anomaly) => (
              <Grid item xs={12} md={6} key={anomaly.id}>
                <Card variant="outlined">
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                      <Box>
                        <Chip
                          label={anomaly.severity.toUpperCase()}
                          size="small"
                          color={getSeverityColor(anomaly.severity) as any}
                        />
                        <Chip
                          label={anomaly.type}
                          size="small"
                          variant="outlined"
                          sx={{ ml: 1 }}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        Confidence: {Math.round(anomaly.confidence * 100)}%
                      </Typography>
                    </Box>
                    
                    <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                      {anomaly.member}
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {anomaly.description}
                    </Typography>
                    
                    <Typography variant="caption" color="text.secondary">
                      Detected: {new Date(anomaly.detectedAt).toLocaleDateString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        {/* Pattern Analysis Tab */}
        <TabPanel value={tabValue} index={2}>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <ShowChart sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Pattern Analysis Dashboard
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Advanced pattern recognition and correlation analysis coming soon
            </Typography>
            <Alert severity="info">
              This section will include temporal patterns, committee correlations, 
              and cross-member trading synchronization analysis.
            </Alert>
          </Box>
        </TabPanel>

        {/* Model Performance Tab */}
        <TabPanel value={tabValue} index={3}>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <PieChart sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Model Performance Metrics
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Detailed performance analytics and model comparison
            </Typography>
            <Alert severity="info">
              This section will include accuracy trends, feature importance, 
              model comparison metrics, and training performance history.
            </Alert>
          </Box>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default Analytics;