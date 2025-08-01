import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Alert,
  Tooltip,
  IconButton,
  Avatar
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Search,
  FilterList,
  GetApp,
  InfoOutlined,
  Timeline
} from '@mui/icons-material';

import { TradingDisclosure } from '../../types';
import { useAppState } from '../../contexts/AppStateContext';

const Trading: React.FC = () => {
  const { state, setLoading, setError, clearError } = useAppState();
  const [trades, setTrades] = useState<TradingDisclosure[]>([]);
  const [filteredTrades, setFilteredTrades] = useState<TradingDisclosure[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [sortBy, setSortBy] = useState('transaction_date');

  // Generate mock trading data
  const generateMockTrades = (): TradingDisclosure[] => {
    const members = [
      'Nancy Pelosi', 'Kevin McCarthy', 'Chuck Schumer', 'Mitch McConnell',
      'Alexandria Ocasio-Cortez', 'Ted Cruz', 'Elizabeth Warren', 'Josh Hawley',
      'Mitt Romney', 'Bernie Sanders'
    ];
    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'UBER'];
    const types: Array<'Purchase' | 'Sale' | 'Exchange'> = ['Purchase', 'Sale', 'Exchange'];
    
    return Array.from({ length: 100 }, (_, i) => {
      const transactionType = types[Math.floor(Math.random() * types.length)];
      const amount = Math.floor(Math.random() * 500000) + 1000;
      
      return {
        disclosure_id: `D${i.toString().padStart(6, '0')}`,
        bioguide_id: `M${Math.floor(Math.random() * 50).toString().padStart(6, '0')}`,
        member_name: members[Math.floor(Math.random() * members.length)],
        transaction_date: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        disclosure_date: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        ticker: symbols[Math.floor(Math.random() * symbols.length)],
        asset_description: `${symbols[Math.floor(Math.random() * symbols.length)]} Common Stock`,
        asset_type: 'Stock',
        transaction_type: transactionType,
        amount_range: `$${Math.floor(amount / 1000)}K - $${Math.floor(amount / 1000) + 50}K`,
        amount_estimate: amount,
        comment: Math.random() > 0.8 ? 'SP' : undefined,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
    });
  };

  const loadTrades = async () => {
    setLoading('trades', true);
    clearError('trades');

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1200));
      
      const mockTrades = generateMockTrades();
      setTrades(mockTrades);
      setFilteredTrades(mockTrades);
    } catch (error) {
      setError('trades', 'Failed to load trading disclosures');
    } finally {
      setLoading('trades', false);
    }
  };

  const filterTrades = () => {
    let filtered = trades;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(trade =>
        trade.member_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        trade.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
        trade.asset_description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter(trade => trade.transaction_type === typeFilter);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'transaction_date':
          return new Date(b.transaction_date).getTime() - new Date(a.transaction_date).getTime();
        case 'amount_estimate':
          return (b.amount_estimate || 0) - (a.amount_estimate || 0);
        case 'member_name':
          return a.member_name.localeCompare(b.member_name);
        default:
          return 0;
      }
    });

    setFilteredTrades(filtered);
  };

  useEffect(() => {
    loadTrades();
  }, []);

  useEffect(() => {
    filterTrades();
  }, [searchTerm, typeFilter, sortBy, trades]);

  const getTransactionColor = (type: string) => {
    switch (type) {
      case 'Purchase': return 'success';
      case 'Sale': return 'error';
      case 'Exchange': return 'warning';
      default: return 'default';
    }
  };

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case 'Purchase': return <TrendingUp />;
      case 'Sale': return <TrendingDown />;
      case 'Exchange': return <Timeline />;
      default: return <InfoOutlined />;
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const getInitials = (name: string) => {
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  };

  if (state.loading.trades && trades.length === 0) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Loading trading disclosures...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" color="primary" gutterBottom>
          Trading Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          STOCK Act disclosures and trading pattern analysis
        </Typography>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Total Disclosures
              </Typography>
              <Typography variant="h3" color="text.primary">
                {trades.length.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="success.main" gutterBottom>
                Purchases
              </Typography>
              <Typography variant="h3" color="text.primary">
                {trades.filter(t => t.transaction_type === 'Purchase').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="error.main" gutterBottom>
                Sales
              </Typography>
              <Typography variant="h3" color="text.primary">
                {trades.filter(t => t.transaction_type === 'Sale').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Total Volume
              </Typography>
              <Typography variant="h3" color="text.primary">
                {formatCurrency(trades.reduce((sum, t) => sum + (t.amount_estimate || 0), 0))}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                placeholder="Search by member, ticker, or asset..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                }}
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Transaction Type</InputLabel>
                <Select
                  value={typeFilter}
                  label="Transaction Type"
                  onChange={(e) => setTypeFilter(e.target.value)}
                >
                  <MenuItem value="all">All Types</MenuItem>
                  <MenuItem value="Purchase">Purchase</MenuItem>
                  <MenuItem value="Sale">Sale</MenuItem>
                  <MenuItem value="Exchange">Exchange</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={sortBy}
                  label="Sort By"
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <MenuItem value="transaction_date">Transaction Date</MenuItem>
                  <MenuItem value="amount_estimate">Amount</MenuItem>
                  <MenuItem value="member_name">Member Name</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={2}>
              <Button
                variant="outlined"
                startIcon={<GetApp />}
                fullWidth
                onClick={() => {
                  console.log('Export trades');
                }}
              >
                Export
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <Alert severity="info" sx={{ mb: 2 }}>
        Showing {filteredTrades.length} of {trades.length} trading disclosures
      </Alert>

      {/* Trading Table */}
      <Card>
        <CardContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Member</TableCell>
                  <TableCell>Transaction</TableCell>
                  <TableCell>Asset</TableCell>
                  <TableCell>Amount</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Disclosure</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredTrades.slice(0, 50).map((trade) => (
                  <TableRow key={trade.disclosure_id} hover>
                    {/* Member */}
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Avatar
                          sx={{
                            width: 32,
                            height: 32,
                            bgcolor: 'primary.main',
                            mr: 2,
                            fontSize: '0.75rem'
                          }}
                        >
                          {getInitials(trade.member_name)}
                        </Avatar>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {trade.member_name}
                        </Typography>
                      </Box>
                    </TableCell>

                    {/* Transaction */}
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{ mr: 1, color: `${getTransactionColor(trade.transaction_type)}.main` }}>
                          {getTransactionIcon(trade.transaction_type)}
                        </Box>
                        <Chip
                          label={trade.transaction_type}
                          size="small"
                          color={getTransactionColor(trade.transaction_type) as any}
                          variant="outlined"
                        />
                      </Box>
                    </TableCell>

                    {/* Asset */}
                    <TableCell>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {trade.ticker}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" noWrap>
                          {trade.asset_description}
                        </Typography>
                      </Box>
                    </TableCell>

                    {/* Amount */}
                    <TableCell>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {trade.amount_estimate ? formatCurrency(trade.amount_estimate) : trade.amount_range}
                        </Typography>
                        {trade.comment && (
                          <Chip label={trade.comment} size="small" variant="outlined" sx={{ mt: 0.5 }} />
                        )}
                      </Box>
                    </TableCell>

                    {/* Transaction Date */}
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(trade.transaction_date).toLocaleDateString()}
                      </Typography>
                    </TableCell>

                    {/* Disclosure Date */}
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(trade.disclosure_date).toLocaleDateString()}
                      </Typography>
                    </TableCell>

                    {/* Actions */}
                    <TableCell align="center">
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => {
                            console.log('View trade details:', trade.disclosure_id);
                          }}
                        >
                          <InfoOutlined fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {filteredTrades.length > 50 && (
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Button variant="outlined">
                Load More ({filteredTrades.length - 50} remaining)
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* No Results */}
      {filteredTrades.length === 0 && !state.loading.trades && (
        <Card sx={{ mt: 2 }}>
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No trading disclosures found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Try adjusting your search criteria or filters
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default Trading;