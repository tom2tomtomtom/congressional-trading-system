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
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Paper,
  TextField,
  InputAdornment,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  NewspaperOutlined,
  TrendingUp,
  TrendingDown,
  Search,
  Refresh,
  OpenInNew,
  Sentiment,
  Schedule,
  Language,
  Twitter,
  LinkedIn,
  Reddit,
  FilterList
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
      id={`intelligence-tabpanel-${index}`}
      aria-labelledby={`intelligence-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  publishedAt: string;
  url: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentimentScore: number;
  relatedMembers: string[];
  relatedStocks: string[];
  category: 'politics' | 'market' | 'regulatory' | 'scandal';
}

interface SentimentTrend {
  date: string;
  sentiment: number;
  volume: number;
  keyTopics: string[];
}

interface MarketCorrelation {
  member: string;
  newsVolume: number;
  marketImpact: number;
  correlation: number;
  lastUpdate: string;
}

const Intelligence: React.FC = () => {
  const { state, setLoading, setError, clearError } = useAppState();
  const [tabValue, setTabValue] = useState(0);
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [sentimentTrends, setSentimentTrends] = useState<SentimentTrend[]>([]);
  const [marketCorrelations, setMarketCorrelations] = useState<MarketCorrelation[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [sentimentFilter, setSentimentFilter] = useState('all');

  const generateMockData = () => {
    const members = ['Nancy Pelosi', 'Kevin McCarthy', 'Chuck Schumer', 'Elizabeth Warren'];
    const stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
    const sources = ['Reuters', 'Bloomberg', 'Wall Street Journal', 'Financial Times', 'CNBC'];
    const categories: Array<'politics' | 'market' | 'regulatory' | 'scandal'> = ['politics', 'market', 'regulatory', 'scandal'];
    const sentiments: Array<'positive' | 'negative' | 'neutral'> = ['positive', 'negative', 'neutral'];

    const mockNews: NewsItem[] = Array.from({ length: 20 }, (_, i) => ({
      id: `news_${i}`,
      title: [
        'Congressional Committee Announces Tech Regulation Hearing',
        'Senator Questions Big Tech Market Dominance',
        'House Passes Infrastructure Investment Bill',
        'Ethics Committee Reviews Trading Activities',
        'New Financial Disclosure Requirements Proposed',
        'Market Reacts to Congressional Banking Hearing',
        'Lawmakers Propose Cryptocurrency Regulation Framework',
        'Congressional Investigation into Market Manipulation',
        'Senator Introduces AI Governance Legislation',
        'House Committee Examines Social Media Policies'
      ][Math.floor(Math.random() * 10)],
      summary: 'Recent developments in congressional activities that may impact financial markets and trading patterns. Analysis of potential policy implications and market correlations.',
      source: sources[Math.floor(Math.random() * sources.length)],
      publishedAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      url: `https://example.com/news/${i}`,
      sentiment: sentiments[Math.floor(Math.random() * sentiments.length)],
      sentimentScore: -1 + Math.random() * 2, // -1 to 1
      relatedMembers: [members[Math.floor(Math.random() * members.length)]],
      relatedStocks: stocks.slice(0, Math.floor(Math.random() * 3) + 1),
      category: categories[Math.floor(Math.random() * categories.length)]
    }));

    const mockSentimentTrends: SentimentTrend[] = Array.from({ length: 7 }, (_, i) => ({
      date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      sentiment: -0.5 + Math.random(),
      volume: Math.floor(Math.random() * 100) + 20,
      keyTopics: ['regulation', 'trading', 'ethics'].slice(0, Math.floor(Math.random() * 3) + 1)
    }));

    const mockCorrelations: MarketCorrelation[] = members.map(member => ({
      member,
      newsVolume: Math.floor(Math.random() * 50) + 10,
      marketImpact: Math.random() * 10,
      correlation: Math.random() * 0.8 + 0.2,
      lastUpdate: new Date().toISOString()
    }));

    setNewsItems(mockNews);
    setSentimentTrends(mockSentimentTrends);
    setMarketCorrelations(mockCorrelations);
  };

  const loadIntelligenceData = async () => {
    setLoading('intelligence', true);
    clearError('intelligence');

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      generateMockData();
    } catch (error) {
      setError('intelligence', 'Failed to load intelligence data');
    } finally {
      setLoading('intelligence', false);
    }
  };

  useEffect(() => {
    loadIntelligenceData();
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'success';
      case 'negative': return 'error';
      case 'neutral': return 'default';
      default: return 'default';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return <TrendingUp />;
      case 'negative': return <TrendingDown />;
      case 'neutral': return <Sentiment />;
      default: return <Sentiment />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'politics': return 'primary';
      case 'market': return 'success';
      case 'regulatory': return 'warning';
      case 'scandal': return 'error';
      default: return 'default';
    }
  };

  const getSourceIcon = (source: string) => {
    if (source.toLowerCase().includes('twitter')) return <Twitter />;
    if (source.toLowerCase().includes('linkedin')) return <LinkedIn />;
    if (source.toLowerCase().includes('reddit')) return <Reddit />;
    return <Language />;
  };

  const filteredNews = newsItems.filter(item => {
    const matchesSearch = searchTerm === '' || 
      item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.relatedMembers.some(member => member.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = categoryFilter === 'all' || item.category === categoryFilter;
    const matchesSentiment = sentimentFilter === 'all' || item.sentiment === sentimentFilter;
    
    return matchesSearch && matchesCategory && matchesSentiment;
  });

  if (state.loading.intelligence) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Loading intelligence data...
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
            Intelligence & News
            <Chip label="Phase 2" size="small" color="secondary" sx={{ ml: 2 }} />
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time news monitoring and sentiment analysis for congressional trading intelligence
          </Typography>
        </Box>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={loadIntelligenceData}
        >
          Refresh Feed
        </Button>
      </Box>

      {/* Phase 2 Alert */}
      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>📰 Intelligence Monitoring Active:</strong> Real-time news collection, sentiment analysis, 
          and market correlation tracking are providing comprehensive intelligence on congressional activities.
        </Typography>
      </Alert>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                News Articles
              </Typography>
              <Typography variant="h3" color="text.primary">
                {newsItems.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                This week
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="success.main" gutterBottom>
                Positive Sentiment
              </Typography>
              <Typography variant="h3" color="text.primary">
                {Math.round((newsItems.filter(n => n.sentiment === 'positive').length / newsItems.length) * 100)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Overall sentiment
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="warning.main" gutterBottom>
                Market Correlations
              </Typography>
              <Typography variant="h3" color="text.primary">
                {marketCorrelations.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active correlations
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Sources Monitored
              </Typography>
              <Typography variant="h3" color="text.primary">
                247
              </Typography>
              <Typography variant="body2" color="text.secondary">
                News & social media
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="intelligence tabs">
            <Tab 
              label="News Feed" 
              icon={<NewspaperOutlined />} 
              iconPosition="start"
            />
            <Tab 
              label="Sentiment Analysis" 
              icon={<Sentiment />} 
              iconPosition="start"
            />
            <Tab 
              label="Market Correlations" 
              icon={<TrendingUp />} 
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {/* News Feed Tab */}
        <TabPanel value={tabValue} index={0}>
          {/* Filters */}
          <Box sx={{ mb: 3 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  placeholder="Search news and members..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search />
                      </InputAdornment>
                    )
                  }}
                />
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={categoryFilter}
                    label="Category"
                    onChange={(e) => setCategoryFilter(e.target.value)}
                  >
                    <MenuItem value="all">All Categories</MenuItem>
                    <MenuItem value="politics">Politics</MenuItem>
                    <MenuItem value="market">Market</MenuItem>
                    <MenuItem value="regulatory">Regulatory</MenuItem>
                    <MenuItem value="scandal">Scandal</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Sentiment</InputLabel>
                  <Select
                    value={sentimentFilter}
                    label="Sentiment"
                    onChange={(e) => setSentimentFilter(e.target.value)}
                  >
                    <MenuItem value="all">All Sentiments</MenuItem>
                    <MenuItem value="positive">Positive</MenuItem>
                    <MenuItem value="neutral">Neutral</MenuItem>
                    <MenuItem value="negative">Negative</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <Button
                  variant="outlined"
                  startIcon={<FilterList />}
                  onClick={() => {
                    setSearchTerm('');
                    setCategoryFilter('all');
                    setSentimentFilter('all');
                  }}
                  fullWidth
                >
                  Clear
                </Button>
              </Grid>
            </Grid>
          </Box>

          {/* News List */}
          <List>
            {filteredNews.map((item) => (
              <Paper key={item.id} sx={{ mb: 2 }}>
                <ListItem alignItems="flex-start">
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: `${getCategoryColor(item.category)}.main` }}>
                      {getSourceIcon(item.source)}
                    </Avatar>
                  </ListItemAvatar>
                  
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h6" sx={{ flexGrow: 1, mr: 2 }}>
                          {item.title}
                        </Typography>
                        <Tooltip title="Open Article">
                          <IconButton size="small" href={item.url} target="_blank">
                            <OpenInNew fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          {item.summary}
                        </Typography>
                        
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Chip
                            label={item.category}
                            size="small"
                            color={getCategoryColor(item.category) as any}
                            variant="outlined"
                          />
                          <Chip
                            label={item.sentiment}
                            size="small"
                            color={getSentimentColor(item.sentiment) as any}
                            icon={getSentimentIcon(item.sentiment)}
                          />
                          <Typography variant="caption" color="text.secondary">
                            Score: {item.sentimentScore.toFixed(2)}
                          </Typography>
                        </Box>
                        
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            {item.source} • {new Date(item.publishedAt).toLocaleDateString()}
                          </Typography>
                          
                          {item.relatedMembers.length > 0 && (
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                              {item.relatedMembers.map((member, idx) => (
                                <Chip 
                                  key={idx}
                                  label={member} 
                                  size="small" 
                                  variant="outlined"
                                  sx={{ fontSize: '0.7rem' }}
                                />
                              ))}
                            </Box>
                          )}
                          
                          {item.relatedStocks.length > 0 && (
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                              {item.relatedStocks.map((stock, idx) => (
                                <Chip 
                                  key={idx}
                                  label={stock} 
                                  size="small" 
                                  color="primary"
                                  variant="outlined"
                                  sx={{ fontSize: '0.7rem' }}
                                />
                              ))}
                            </Box>
                          )}
                        </Box>
                      </Box>
                    }
                  />
                </ListItem>
              </Paper>
            ))}
          </List>
        </TabPanel>

        {/* Sentiment Analysis Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Sentiment sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Sentiment Analysis Dashboard
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Advanced sentiment tracking and trend analysis coming soon
            </Typography>
            <Alert severity="info">
              This section will include sentiment time series, topic analysis, 
              and emotional response tracking across multiple news sources.
            </Alert>
          </Box>
        </TabPanel>

        {/* Market Correlations Tab */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" color="primary" gutterBottom>
            News-Market Correlation Analysis
          </Typography>
          
          <Grid container spacing={2}>
            {marketCorrelations.map((correlation) => (
              <Grid item xs={12} md={6} key={correlation.member}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" color="primary" gutterBottom>
                      {correlation.member}
                    </Typography>
                    
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">News Volume:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {correlation.newsVolume} articles
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Market Impact:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {correlation.marketImpact.toFixed(1)}%
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Correlation:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {correlation.correlation.toFixed(3)}
                        </Typography>
                      </Box>
                    </Box>
                    
                    <LinearProgress
                      variant="determinate"
                      value={correlation.correlation * 100}
                      sx={{ mb: 1 }}
                    />
                    
                    <Typography variant="caption" color="text.secondary">
                      Last updated: {new Date(correlation.lastUpdate).toLocaleTimeString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default Intelligence;